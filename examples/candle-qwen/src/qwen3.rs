use candle::{DType, Device, Module};
use candle_nn::{kv_cache::KvCache, Activation, VarBuilder};
use candle_transformers::{
    models::with_tracing::{linear_b, linear_no_bias, Linear, RmsNorm},
    utils::repeat_kv,
};
use glowstick::{
    num::{U0, U1, U1024, U12, U128, U16, U2, U2048, U3, U8},
    Shape2, Shape3, Shape4,
};
use glowstick_candle::tensor::Tensor;
use glowstick_candle::{broadcast_add, flatten, matmul, narrow, reshape, transpose};
use std::sync::Arc;

#[allow(unused)]
use glowstick::debug_tensor;

use super::Error;
use crate::shape::*;

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub max_window_layers: usize,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub use_sliding_window: bool,
    pub hidden_act: Activation,
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen3RotaryEmbedding {
    sin: candle::Tensor,
    cos: candle::Tensor,
}

impl Qwen3RotaryEmbedding {
    pub(crate) fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self, Error> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq =
            candle::Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = candle::Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    #[allow(clippy::type_complexity)]
    fn apply(
        &self,
        q: &Tensor<Shape4<N, U16, L, U128>>,
        k: &Tensor<Shape4<N, U8, L, U128>>,
        offset: usize,
    ) -> Result<
        (
            Tensor<Shape4<N, U16, L, U128>>,
            Tensor<Shape4<N, U8, L, U128>>,
        ),
        Error,
    > {
        let (_, _, seq_len, _) = q.inner().dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(q.contiguous()?.inner(), &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(k.contiguous()?.inner(), &cos, &sin)?;
        Ok((q_embed.try_into()?, k_embed.try_into()?))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen3MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Qwen3MLP {
    pub(crate) fn new(cfg: &Config, vb: VarBuilder) -> Result<Self, Error> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Qwen3MLP {
    fn forward(&self, x: &candle::Tensor) -> candle::Result<candle::Tensor> {
        let lhs = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = x.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
#[allow(unused)]
pub(crate) struct Qwen3Attention {
    // projections
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    // norms
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    // hyper params
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    // utils
    rotary_emb: Arc<Qwen3RotaryEmbedding>,
    kv_cache: KvCache,
}

impl Qwen3Attention {
    pub(crate) fn new(
        cfg: &Config,
        rotary_emb: Arc<Qwen3RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self, Error> {
        if cfg.use_sliding_window {
            return Err(Error::Candle(candle::Error::Msg(
                "sliding window is not suppored".to_string(),
            )));
        }

        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;

        let q_proj = linear_b(
            cfg.hidden_size,
            num_heads * head_dim,
            cfg.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            num_heads * head_dim,
            cfg.hidden_size,
            cfg.attention_bias,
            vb.pp("o_proj"),
        )?;

        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        // Necessary because the hidden_size in the config isn't always accurate
        let hidden_size = head_dim * cfg.num_attention_heads;

        let kv_cache = KvCache::new(2, cfg.max_position_embeddings);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
            rotary_emb,
            kv_cache,
        })
    }

    pub(crate) fn forward(
        &mut self,
        x: &Tensor<Shape3<N, L, U1024>>,
        attn_mask: Option<&Tensor<Shape4<N, U1, L, U12>>>,
        offset: usize,
    ) -> Result<Tensor<Shape3<N, L, U1024>>, Error> {
        // 1. Proj
        let q: Tensor<Shape3<N, L, U2048>> = self.q_proj.forward(x.inner())?.try_into()?;
        let k: Tensor<Shape3<N, L, U1024>> = self.k_proj.forward(x.inner())?.try_into()?;
        let v: Tensor<Shape3<N, L, U1024>> = self.v_proj.forward(x.inner())?.try_into()?;

        // 2. Reshape
        let (_, l, _) = q.inner().dims3()?;
        let q = reshape!(&q, [() => N, { l } => L, U16, U128])?;
        let q = transpose!(&q, U1:U2)?;
        let k = reshape!(&k, [() => N, { l } => L, U8, U128])?;
        let k = transpose!(&k, U1:U2)?;
        let v = reshape!(&v, [() => N, { l } => L, U8, U128])?;
        let v = transpose!(&v, U1:U2)?;

        // 3. Per‑head RMSNorm
        let q_flat = flatten!(&q, [U0, U2])?;
        let k_flat = flatten!(&k, [U0, U2])?;
        let q_flat: Tensor<Shape2<Bhl, U128>> = self.q_norm.forward(q_flat.inner())?.try_into()?;
        let k_flat: Tensor<Shape2<Bhl, U128>> = self.k_norm.forward(k_flat.inner())?.try_into()?;
        let q = reshape!(&q_flat, [() => N, U16, { l } => L, U128])?;
        let k = reshape!(&k_flat, [() => N, U8, { l } => L, U128])?;

        // 4. RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // 5. Accumulate KV cache
        let (k, v) = self
            .kv_cache
            .append(k.contiguous()?.inner(), v.contiguous()?.inner())?;

        // 6. GQA repeat_kv
        let k: Tensor<Shape4<N, U16, L, U128>> = repeat_kv(k, self.num_kv_groups)?.try_into()?;
        let v: Tensor<Shape4<N, U16, L, U128>> = repeat_kv(v, self.num_kv_groups)?.try_into()?;

        // 7. Attention score
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = transpose!(&k, U2:U3)?;
        let scores = matmul!(&q, scores)?;
        let mut scores = (scores * scale)?;
        if let Some(m) = attn_mask {
            scores = broadcast_add!(scores, m)?;
        }
        let probs: Tensor<Shape4<N, U16, L, L>> =
            candle_nn::ops::softmax_last_dim(scores.inner())?.try_into()?;
        let ctx = matmul!(probs, &v)?;

        // 8. Output proj
        let ctx = transpose!(&ctx, U1:U2)?;
        let ctx = reshape!(&ctx, [() => N, { l } => L, U2048])?;
        Ok(ctx.inner().apply(&self.o_proj)?.try_into()?)
    }

    pub(crate) fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: Qwen3MLP,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &Config, rotary: Arc<Qwen3RotaryEmbedding>, vb: VarBuilder) -> Result<Self, Error> {
        let self_attn = Qwen3Attention::new(cfg, rotary, vb.pp("self_attn"))?;
        let mlp = Qwen3MLP::new(cfg, vb.pp("mlp"))?;
        let ln1 = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let ln2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor<Shape3<N, L, U1024>>,
        mask: Option<&Tensor<Shape4<N, U1, L, U12>>>,
        offset: usize,
    ) -> Result<Tensor<Shape3<N, L, U1024>>, Error> {
        let h: Tensor<Shape3<N, L, U1024>> = self.ln1.forward(x.inner())?.try_into()?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2: Tensor<Shape3<N, L, U1024>> = self.ln2.forward(x.inner())?.try_into()?;
        let h2: Tensor<Shape3<N, L, U1024>> = h2.inner().apply(&self.mlp)?.try_into()?;
        Ok((&x + h2)?)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self, Error> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let rotary = Arc::new(Qwen3RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, rotary.clone(), vb_l.pp(i))?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn clear_kv_cache(&mut self) {
        for l in &mut self.layers {
            l.clear_kv_cache();
        }
    }

    fn causal_mask(
        &self,
        b: usize,
        tgt: usize,
        offset: usize,
        sw: Option<usize>,
    ) -> Result<Tensor<Shape4<N, U1, L, U12>>, Error> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| {
                    let past_ok = j <= i + offset;
                    let sw_ok = match sw {
                        Some(w) => (i + offset) as i64 - j as i64 <= w as i64,
                        None => true,
                    };
                    if past_ok && sw_ok {
                        0.
                    } else {
                        minf
                    }
                })
            })
            .collect();
        Ok(
            candle::Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?
                .to_dtype(self.dtype)?
                .try_into()?,
        )
    }

    pub fn forward(
        &mut self,
        input: &Tensor<Shape2<N, L>>,
        offset: usize,
    ) -> Result<Tensor<Shape3<N, L, U1024>>, Error> {
        let (n, l) = input.inner().dims2()?;
        let mut h: Tensor<Shape3<N, L, U1024>> =
            self.embed_tokens.forward(input.inner())?.try_into()?;

        let causal = if l == 1 {
            None
        } else {
            Some(self.causal_mask(n, l, offset, None)?)
        };

        for layer in &mut self.layers {
            h = layer.forward(&h, causal.as_ref(), offset)?;
        }
        Ok(self.norm.forward(h.inner())?.try_into()?)
    }
}

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    base: Model,
    lm_head: Linear,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self, Error> {
        let base = Model::new(cfg, vb.clone())?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(base.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self { base, lm_head })
    }

    pub fn forward(
        &mut self,
        input: &Tensor<Shape2<N, L>>,
        offset: usize,
    ) -> Result<Tensor<Shape3<N, U1, U151936>>, Error> {
        let (_, l) = input.inner().dims2()?;
        let t = self.base.forward(input, offset)?;
        let t = narrow!(&t, U1: [l - 1, U1])?;
        Ok(t.inner().apply(&self.lm_head)?.try_into()?)
    }

    #[allow(unused)]
    pub fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache();
    }
}
