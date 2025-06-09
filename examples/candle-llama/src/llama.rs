//! Llama inference implementation.
//!
//! See ["LLaMA: Open and Efficient Foundation Language Models"](https://arxiv.org/abs/2302.13971)
//!
//! Implementation based on Hugging Face's [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)

use candle::{DType, Device, IndexOp, Tensor as CandleTensor};
use candle_nn::{embedding, Embedding, Module, VarBuilder};
use candle_transformers::models::with_tracing::{linear_no_bias as linear, Linear, RmsNorm};
use glowstick::{
    num::{U0, U1, U2, U3, U32, U8192},
    Shape2, Shape3, Shape4,
};
use glowstick_candle::{cat, expand, matmul, narrow, reshape, softmax, transpose, Error, Tensor};
use std::{collections::HashMap, f32::consts::PI};

use crate::shape::{A, B, C, H, K, KV, N, Q, S};

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub enum Llama3RopeType {
    #[serde(rename = "llama3")]
    Llama3,
    #[default]
    #[serde(rename = "default")]
    Default,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct Llama3RopeConfig {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_type: Llama3RopeType,
}
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum LlamaEosToks {
    Single(u32),
    Multiple(Vec<u32>),
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    pub eos_token_id: Option<LlamaEosToks>,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: Option<bool>,
}

impl LlamaConfig {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
}

fn default_rope() -> f32 {
    10_000.0
}

impl LlamaConfig {
    pub fn into_config(self, use_flash_attn: bool) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads(),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            use_flash_attn,
            eos_token_id: self.eos_token_id,
            rope_scaling: self.rope_scaling,
            max_position_embeddings: self.max_position_embeddings,
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub use_flash_attn: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub eos_token_id: Option<LlamaEosToks>,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

type CacheTensor = Tensor<Shape4<B, K, N, H>>;
#[derive(Clone)]
pub struct Cache {
    masks: HashMap<usize, Tensor<Shape2<N, N>>>,
    pub use_kv_cache: bool,
    kvs: Vec<Option<(CacheTensor, CacheTensor)>>,
    cos: Tensor<Shape2<U8192, U32>>,
    sin: Tensor<Shape2<U8192, U32>>,
    device: Device,
}

fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

impl Cache {
    pub fn new(
        use_kv_cache: bool,
        dtype: DType,
        config: &Config,
        device: &Device,
    ) -> Result<Self, Error> {
        // precompute freqs_cis
        let theta = match &config.rope_scaling {
            None
            | Some(Llama3RopeConfig {
                rope_type: Llama3RopeType::Default,
                ..
            }) => calculate_default_inv_freq(config),
            Some(rope_scaling) => {
                let low_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.low_freq_factor;
                let high_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.high_freq_factor;

                calculate_default_inv_freq(config)
                    .into_iter()
                    .map(|freq| {
                        let wavelen = 2. * PI / freq;
                        if wavelen < high_freq_wavelen {
                            freq
                        } else if wavelen > low_freq_wavelen {
                            freq / rope_scaling.factor
                        } else {
                            let smooth = (rope_scaling.original_max_position_embeddings as f32
                                / wavelen
                                - rope_scaling.low_freq_factor)
                                / (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor);
                            (1. - smooth) * freq / rope_scaling.factor + smooth * freq
                        }
                    })
                    .collect::<Vec<_>>()
            }
        };

        let theta = CandleTensor::new(theta, device)?;

        let idx_theta = CandleTensor::arange(0, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        // This is different from the paper, see:
        // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        let cos: Tensor<Shape2<U8192, U32>> = idx_theta.cos()?.to_dtype(dtype)?.try_into()?;
        let sin: Tensor<Shape2<U8192, U32>> = idx_theta.sin()?.to_dtype(dtype)?.try_into()?;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; config.num_hidden_layers],
            device: device.clone(),
            cos,
            sin,
        })
    }

    fn mask(&mut self, t: usize) -> Result<Tensor<Shape2<N, N>>, Error> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask: Tensor<Shape2<N, N>> =
                CandleTensor::from_slice(&mask, (t, t), &self.device)?.try_into()?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }
}

#[derive(Debug, Clone)]
struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    use_flash_attn: bool,
    span: tracing::Span,
    span_rot: tracing::Span,
    max_position_embeddings: usize,
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor<Shape4<B, N, A, H>>,
    k: &Tensor<Shape4<B, N, A, H>>,
    v: &Tensor<Shape4<B, N, A, H>>,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor<Shape4<B, N, A, H>>, Error> {
    candle_flash_attn::flash_attn(q.inner(), k.inner(), v.inner(), softmax_scale, causal)?
        .try_into()
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(
    _: &Tensor<Shape4<B, N, A, H>>,
    _: &Tensor<Shape4<B, N, A, H>>,
    _: &Tensor<Shape4<B, N, A, H>>,
    _: f32,
    _: bool,
) -> Result<Tensor<Shape4<B, N, A, H>>, Error> {
    unimplemented!("compile with '--features flash-attn'")
}

impl CausalSelfAttention {
    fn apply_attn_head_rotary_emb(
        &self,
        x: &Tensor<Shape4<B, A, N, H>>,
        index_pos: usize,
        cache: &Cache,
    ) -> Result<Tensor<Shape4<B, A, N, H>>, Error> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _, seq_len, _hidden_size) = x.inner().dims4()?;
        let cos = narrow!(&cache.cos, U0: [{ index_pos }, { seq_len }] => N)?;
        let sin = narrow!(&cache.sin, U0: [{ index_pos }, { seq_len }] => N)?;
        candle_nn::rotary_emb::rope(x.inner(), cos.inner(), sin.inner())?.try_into()
    }

    fn apply_kv_head_rotary_emb(
        &self,
        x: &Tensor<Shape4<B, K, N, H>>,
        index_pos: usize,
        cache: &Cache,
    ) -> Result<Tensor<Shape4<B, K, N, H>>, Error> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _, seq_len, _hidden_size) = x.inner().dims4()?;
        let cos = narrow!(&cache.cos, U0: [{ index_pos }, { seq_len }] => N)?;
        let sin = narrow!(&cache.sin, U0: [{ index_pos }, { seq_len }] => N)?;
        candle_nn::rotary_emb::rope(x.inner(), cos.inner(), sin.inner())?.try_into()
    }

    fn forward(
        &self,
        x: &Tensor<Shape3<B, N, S>>,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor<Shape3<B, N, S>>, Error> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, _hidden_size) = x.inner().dims3()?;
        let q: Tensor<Shape3<B, N, Q>> = self.q_proj.forward(x.inner())?.try_into()?;
        let k: Tensor<Shape3<B, N, KV>> = self.k_proj.forward(x.inner())?.try_into()?;
        let v: Tensor<Shape3<B, N, KV>> = self.v_proj.forward(x.inner())?.try_into()?;

        let q =
            transpose!(reshape!(&q, [() => B, { seq_len } => N, A, H])?, U1:U2)?.contiguous()?;
        let k =
            transpose!(reshape!(&k, [() => B, { seq_len } => N, K, H])?, U1:U2)?.contiguous()?;
        let mut v =
            transpose!(reshape!(&v, [() => B, { seq_len } => N, K, H])?, U1:U2)?.contiguous()?;

        let q = self.apply_attn_head_rotary_emb(&q, index_pos, cache)?;
        let mut k = self.apply_kv_head_rotary_emb(&k, index_pos, cache)?;

        if cache.use_kv_cache {
            if let Some((cache_k, cache_v)) = &cache.kvs[block_idx] {
                k = cat!(vec![cache_k, &k].as_slice(), U2 => N)?.contiguous()?;
                v = cat!(vec![cache_v, &v].as_slice(), U2 => N)?.contiguous()?;
                let k_seq_len = k.inner().dims()[2];
                if k_seq_len > self.max_position_embeddings {
                    k = narrow!(
                        &k,
                        U2: [{ k_seq_len - self.max_position_embeddings }, self.max_position_embeddings] => N
                    )?;
                }
                let v_seq_len = v.inner().dims()[2];
                if v_seq_len > 2 * self.max_position_embeddings {
                    v = narrow!(
                        &v,
                        U2: [{ v_seq_len - self.max_position_embeddings }, self.max_position_embeddings] => N
                    )?;
                }
            }
            cache.kvs[block_idx] = Some((k.clone(), v.clone()))
        }

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let y = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = transpose!(q, U1:U2)?;
            let k = transpose!(k, U1:U2)?;
            let v = transpose!(v, U1:U2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            transpose!(flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?, U1:U2)
        } else {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            let att = (matmul!(q, transpose!(k, U2:U3)?)? / (self.head_dim as f64).sqrt())?;
            let att = if seq_len == 1 {
                att
            } else {
                let mask = expand!(&cache.mask(seq_len)?, &att)?;
                masked_fill(&att, &mask, f32::NEG_INFINITY)?
            };

            let att = softmax!(att, U3)?;
            // Convert to contiguous as matmul doesn't support strided vs for now.
            matmul!(att, &v.contiguous()?)?.to_dtype(in_dtype)
        }?;
        let y = transpose!(y, U1:U2)?;
        let y = reshape!(y, [{ b_sz } => B, { seq_len } => N, S])?;
        let y = self.o_proj.forward(y.inner())?;
        y.try_into()
    }

    fn repeat_kv(
        &self,
        x: Tensor<Shape4<B, K, N, H>>,
    ) -> Result<Tensor<Shape4<B, A, N, H>>, Error> {
        candle_transformers::utils::repeat_kv(
            x.into_inner(),
            self.num_attention_heads / self.num_key_value_heads,
        )?
        .try_into()
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self, candle::Error> {
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = linear(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            use_flash_attn: cfg.use_flash_attn,
            span,
            span_rot,
            max_position_embeddings: cfg.max_position_embeddings,
        })
    }
}

fn masked_fill(
    on_false: &Tensor<Shape4<B, A, N, N>>,
    mask: &Tensor<Shape4<B, A, N, N>>,
    on_true: f32,
) -> Result<Tensor<Shape4<B, A, N, N>>, Error> {
    let shape = mask.inner().shape();
    let on_true =
        CandleTensor::new(on_true, on_false.inner().device())?.broadcast_as(shape.dims())?;
    let m = mask
        .inner()
        .where_cond(&on_true, on_false.inner())?
        .try_into()?;
    Ok(m)
}

#[derive(Debug, Clone)]
struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
    span: tracing::Span,
}

impl Mlp {
    fn forward(&self, x: &Tensor<Shape3<B, N, S>>) -> Result<Tensor<Shape3<B, N, S>>, Error> {
        let _enter = self.span.enter();
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x.inner())?)?
            * self.c_fc2.forward(x.inner())?)?;
        self.c_proj.forward(&x)?.try_into()
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self, candle::Error> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = linear(h_size, i_size, vb.pp("gate_proj"))?;
        let c_fc2 = linear(h_size, i_size, vb.pp("up_proj"))?;
        let c_proj = linear(i_size, h_size, vb.pp("down_proj"))?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
            span,
        })
    }
}

#[derive(Debug, Clone)]
struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
    span: tracing::Span,
}

impl Block {
    fn forward(
        &self,
        x: &Tensor<Shape3<B, N, S>>,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor<Shape3<B, N, S>>, Error> {
        let _enter = self.span.enter();
        let residual = x;
        let x: Tensor<Shape3<B, N, S>> = self.rms_1.forward(x.inner())?.try_into()?;
        let x = (&self.attn.forward(&x, index_pos, block_idx, cache)? + residual)?;
        let x = (&self
            .mlp
            .forward(&self.rms_2.forward(x.inner())?.try_into()?)?
            + x)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self, candle::Error> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        let rms_1 = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
            span,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
    pub eos_tokens: Option<LlamaEosToks>,
}

impl Llama {
    pub fn forward(
        &self,
        x: &Tensor<Shape2<B, N>>,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor<Shape2<B, C>>, Error> {
        let (_b_sz, seq_len) = x.inner().dims2()?;
        let mut x: Tensor<Shape3<B, N, S>> = self.wte.forward(x.inner())?.try_into()?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }
        let x = self.ln_f.forward(x.inner())?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?.try_into()?;
        Ok(logits)
    }

    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self, candle::Error> {
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(wte.embeddings().clone(), None)
        } else {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let ln_f = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), cfg).unwrap())
            .collect();

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            eos_tokens: cfg.eos_token_id.clone(),
        })
    }
}
