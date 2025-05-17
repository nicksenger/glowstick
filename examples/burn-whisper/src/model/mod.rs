pub mod cache;
pub mod load;
pub mod timestamps;

use std::num::NonZeroUsize;

use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        self,
        conv::{Conv1d, Conv1dConfig},
        PaddingConfig1d,
    },
    tensor::{backend::Backend, module::embedding, Distribution, Tensor as BTensor},
};
use cache::TensorCache;
use glowstick::{
    num::{U0, U1, U2, U3, U32, U384, U448, U6, U64, U80},
    Shape2, Shape3,
};

#[allow(unused)]
use glowstick::debug_tensor;

use crate::shape::*;
use crate::{expand, flatten, matmul, narrow, reshape, softmax, transpose, unsqueeze};

pub const MAX_TARGET_POSITIONS: usize = 448;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Timestamps not enabled")]
    TimestampsNotEnabled,

    #[error("Timestamp generation failed: {0}")]
    Timestamps(#[from] timestamps::Error),

    #[error("Tensor error: {0}")]
    Tensor(#[from] crate::tensor::Error),
}

#[derive(Config, Debug)]
pub struct WhisperConfig {
    audio_encoder_config: AudioEncoderConfig,
    text_decoder_config: TextDecoderConfig,
}

impl WhisperConfig {
    pub fn init<B: Backend>(&self, tensor_device_ref: &B::Device, timestamps: bool) -> Whisper<B> {
        let n_audio_state = self.audio_encoder_config.n_audio_state;
        let n_text_state = self.text_decoder_config.n_text_state;

        assert!(
            n_audio_state == n_text_state,
            "Audio encoder state size {} must be equal to text decoder state size {}.",
            n_audio_state,
            n_text_state
        );

        let encoder = self.audio_encoder_config.init(tensor_device_ref);
        let decoder = self.text_decoder_config.init(tensor_device_ref, timestamps);

        Whisper { encoder, decoder }
    }
}

#[derive(Module, Debug)]
pub struct Whisper<B: Backend> {
    encoder: AudioEncoder<B>,
    pub(crate) decoder: TextDecoder<B>,
}

impl<B: Backend> Whisper<B> {
    pub fn forward(
        &mut self,
        mel: Rank3Tensor<BB, U80, U3000, B>,
        tokens: Rank2IntTensor<BB, L, B>,
        enable_self_attn_kv_cache: bool,
        attention_output: Option<&mut [TensorCache<B, 4>]>,
        flush: bool,
    ) -> Result<Rank3Tensor<BB, L, U51865, B>, Error> {
        self.decoder.forward(
            tokens,
            self.encoder.forward(mel, flush)?,
            enable_self_attn_kv_cache,
            attention_output,
            flush,
        )
    }

    pub fn forward_encoder(
        &mut self,
        mel: Rank3Tensor<BB, U80, U3000, B>,
        flush: bool,
    ) -> Result<Rank3Tensor<BB, U1500, U384, B>, Error> {
        self.encoder.forward(mel, flush)
    }

    pub fn forward_decoder(
        &mut self,
        tokens: Rank2IntTensor<BB, L, B>,
        encoder_output: Rank3Tensor<BB, U1500, U384, B>,
        enable_self_attn_kv_cache: bool,
        attention_output: Option<&mut [TensorCache<B, 4>]>,
        flush: bool,
    ) -> Result<Rank3Tensor<BB, L, U51865, B>, Error> {
        self.decoder.forward(
            tokens,
            encoder_output,
            enable_self_attn_kv_cache,
            attention_output,
            flush,
        )
    }

    pub fn encoder_ctx_size(&self) -> usize {
        self.encoder.ctx_size()
    }
    pub fn encoder_mel_size(&self) -> usize {
        self.encoder.n_mels
    }

    pub fn decoder_ctx_size(&self) -> usize {
        self.decoder.ctx_size()
    }

    pub fn reset_kv_cache(&mut self) {
        self.encoder.reset_kv_cache();
        self.decoder.reset_kv_cache();
    }

    pub fn dtw_timestamps(
        &mut self,
        attn_outs: Vec<Rank4Tensor<BB, U6, U32, U1500, B>>,
        alignment_heads: timestamps::AlignmentHeads,
        filter_width: NonZeroUsize,
        n_frames: usize,
        n_start_tokens: usize,
        batch_indices: impl IntoIterator<Item = usize>,
    ) -> Result<Vec<timestamps::Raw>, Error> {
        self.reset_kv_cache();
        if attn_outs.is_empty() {
            return Err(Error::TimestampsNotEnabled);
        }

        let timestamps = alignment_heads.extract_timestamps(
            &attn_outs,
            filter_width,
            n_frames,
            n_start_tokens,
            batch_indices,
        )?;

        Ok(timestamps)
    }

    pub fn clear_attention_outputs(&mut self) {
        let _ = self.take_attention_outputs();
    }

    fn take_attention_outputs(&mut self) -> Vec<BTensor<B, 4>> {
        self.decoder
            .blocks
            .iter_mut()
            .map(|layer| &mut layer.cross_attn)
            .filter_map(|attn| match &mut attn.output_attentions {
                Some(attns) => attns.take(),
                _ => None,
            })
            .collect()
    }
}

#[derive(Config, Debug)]
pub struct TextDecoderConfig {
    n_vocab: usize,
    n_text_ctx: usize,
    n_text_state: usize,
    n_text_head: usize,
    n_text_layer: usize,
}

impl TextDecoderConfig {
    pub fn init<B: Backend>(
        &self,
        tensor_device_ref: &B::Device,
        timestamps: bool,
    ) -> TextDecoder<B> {
        let token_embedding = Param::from_tensor(BTensor::random(
            [self.n_vocab, self.n_text_state],
            Distribution::Normal(0.0, 1.0),
            tensor_device_ref,
        ));
        let positional_embedding = Param::from_tensor(BTensor::random(
            [self.n_text_ctx, self.n_text_state],
            Distribution::Normal(0.0, 1.0),
            tensor_device_ref,
        ));
        let blocks: Vec<_> = (0..self.n_text_layer)
            .map(|_| {
                ResidualDecoderAttentionBlockConfig::new(self.n_text_state, self.n_text_head)
                    .init(tensor_device_ref, timestamps)
            })
            .collect();
        let ln = nn::LayerNormConfig::new(self.n_text_state).init(tensor_device_ref);

        let mask: Vec<_> = (0..MAX_TARGET_POSITIONS)
            .flat_map(|i| {
                (0..MAX_TARGET_POSITIONS).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 })
            })
            .collect();
        let mask = BTensor::<B, 1>::from_floats(mask.as_slice(), tensor_device_ref)
            .reshape([MAX_TARGET_POSITIONS, MAX_TARGET_POSITIONS]);
        let mask = Param::from_tensor(mask);

        let n_vocab = self.n_vocab;
        let n_text_ctx = self.n_text_ctx;

        TextDecoder {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            mask,
            n_vocab,
            n_text_ctx,
        }
    }
}

#[derive(Module, Debug)]
pub struct TextDecoder<B: Backend> {
    token_embedding: Param<BTensor<B, 2>>,
    positional_embedding: Param<BTensor<B, 2>>,
    pub(crate) blocks: Vec<ResidualDecoderAttentionBlock<B>>,
    ln: nn::LayerNorm<B>,
    mask: Param<BTensor<B, 2>>,
    n_vocab: usize,
    n_text_ctx: usize,
}

impl<B: Backend> TextDecoder<B> {
    fn forward(
        &mut self,
        x: Rank2IntTensor<BB, L, B>,
        xa: Rank3Tensor<BB, U1500, U384, B>,
        enable_self_attn_kv_cache: bool,
        mut attention_output: Option<&mut [TensorCache<B, 4>]>,
        flush: bool,
    ) -> Result<Rank3Tensor<BB, L, U51865, B>, Error> {
        let batch_size = x.inner().dims()[0];
        let offset = flush
            .then_some(0)
            .or(self
                .blocks
                .first()
                .and_then(|b| b.attn.kv_cache.as_ref())
                .map(|(k, _)| k.dims()[1]))
            .unwrap_or_default();

        let x = if offset == 0 {
            x
        } else {
            narrow!(x, U1, offset, U1).transmute()
        };

        let positional_embedding: Rank2Tensor<U448, U384, B> =
            self.positional_embedding.val().try_into()?;
        let positional_embedding = narrow!(positional_embedding, U0, offset, U1);
        let positional_embedding = unsqueeze!(positional_embedding, U0);
        let mut x = embedding(self.token_embedding.val(), x.into_inner())
            + positional_embedding.into_inner();

        for (i, block) in self.blocks.iter_mut().enumerate() {
            let attention_output = attention_output.as_mut().map(|outputs| &mut outputs[i]);
            x = block
                .forward(
                    x.try_into()?,
                    xa.clone().transmute(),
                    self.mask.val().try_into()?,
                    enable_self_attn_kv_cache,
                    attention_output,
                    flush,
                )?
                .into_inner();
        }

        let x: Rank3Tensor<BB, L, U384, B> = self.ln.forward(x).try_into()?;
        let w: Rank2Tensor<U51865, U384, B> = self.token_embedding.val().try_into()?;
        let w = transpose!(w, U0, U1);
        let w = unsqueeze!(w, U0);
        let w: Rank3Tensor<BB, U384, U51865, B> =
            w.into_inner().repeat(&[batch_size, 0, 0]).try_into()?;

        Ok(matmul!(x, w))
    }

    fn ctx_size(&self) -> usize {
        self.n_text_ctx
    }

    fn reset_kv_cache(&mut self) {
        for block in self.blocks.iter_mut() {
            block.reset_kv_cache();
        }
    }
}

#[derive(Config, Debug)]
pub struct AudioEncoderConfig {
    n_mels: usize,
    n_audio_ctx: usize,
    n_audio_state: usize,
    n_audio_head: usize,
    n_audio_layer: usize,
}

impl AudioEncoderConfig {
    pub fn init<B: Backend>(&self, tensor_device_ref: &B::Device) -> AudioEncoder<B> {
        let conv1 = Conv1dConfig::new(self.n_mels, self.n_audio_state, 3)
            .with_padding(PaddingConfig1d::Explicit(1))
            .init(tensor_device_ref);
        let gelu1 = nn::Gelu::new();
        let conv2 = Conv1dConfig::new(self.n_audio_state, self.n_audio_state, 3)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_stride(2)
            .init(tensor_device_ref);
        let gelu2 = nn::Gelu::new();
        let blocks: Vec<_> = (0..self.n_audio_layer)
            .map(|_| {
                ResidualEncoderAttentionBlockConfig::new(self.n_audio_state, self.n_audio_head)
                    .init(tensor_device_ref)
            })
            .collect();
        let ln_post = nn::LayerNormConfig::new(self.n_audio_state).init(tensor_device_ref);
        let positional_embedding = Param::from_tensor(BTensor::random(
            [self.n_audio_ctx, self.n_audio_state],
            Distribution::Normal(0.0, 1.0),
            tensor_device_ref,
        ));
        let n_mels = self.n_mels;
        let n_audio_ctx = self.n_audio_ctx;

        AudioEncoder {
            conv1,
            gelu1,
            conv2,
            gelu2,
            blocks,
            ln_post,
            positional_embedding,
            n_mels,
            n_audio_ctx,
        }
    }
}

#[derive(Module, Debug)]
pub struct AudioEncoder<B: Backend> {
    conv1: Conv1d<B>,
    gelu1: nn::Gelu,
    conv2: Conv1d<B>,
    gelu2: nn::Gelu,
    blocks: Vec<ResidualEncoderAttentionBlock<B>>,
    ln_post: nn::LayerNorm<B>,
    positional_embedding: Param<BTensor<B, 2>>,
    n_mels: usize,
    n_audio_ctx: usize,
}

impl<B: Backend> AudioEncoder<B> {
    fn forward(
        &mut self,
        x: Rank3Tensor<BB, U80, U3000, B>,
        flush: bool,
    ) -> Result<Rank3Tensor<BB, U1500, U384, B>, Error> {
        let x = self.gelu1.forward(self.conv1.forward(x.into_inner()));
        let x: Rank3Tensor<BB, U384, U1500, B> =
            self.gelu2.forward(self.conv2.forward(x)).try_into()?;

        let x = transpose!(x, U1, U2);
        let positional_embedding: Rank2Tensor<U1500, U384, B> =
            self.positional_embedding.val().try_into()?;
        let positional_embedding = expand!(positional_embedding, &x);
        let mut x = positional_embedding + x;

        for block in &mut self.blocks {
            x = block.forward(x, flush)?;
        }

        self.ln_post
            .forward(x.into_inner())
            .try_into()
            .map_err(Into::into)
    }

    fn ctx_size(&self) -> usize {
        self.n_audio_ctx
    }

    fn reset_kv_cache(&mut self) {
        for block in self.blocks.iter_mut() {
            block.reset_kv_cache();
        }
    }
}

#[derive(Config)]
pub struct ResidualEncoderAttentionBlockConfig {
    n_state: usize,
    n_head: usize,
}

impl ResidualEncoderAttentionBlockConfig {
    pub fn init<B: Backend>(
        &self,
        tensor_device_ref: &B::Device,
    ) -> ResidualEncoderAttentionBlock<B> {
        let attn =
            MultiHeadSelfAttentionConfig::new(self.n_state, self.n_head).init(tensor_device_ref);
        let attn_ln = nn::LayerNormConfig::new(self.n_state).init(tensor_device_ref);
        let mlp = MLPConfig::new(self.n_state).init(tensor_device_ref);
        let mlp_ln = nn::LayerNormConfig::new(self.n_state).init(tensor_device_ref);

        ResidualEncoderAttentionBlock {
            attn,
            attn_ln,
            mlp,
            mlp_ln,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualEncoderAttentionBlock<B: Backend> {
    attn: MultiHeadSelfAttention<B>,
    attn_ln: nn::LayerNorm<B>,
    mlp: MLP<B>,
    mlp_ln: nn::LayerNorm<B>,
}

impl<B: Backend> ResidualEncoderAttentionBlock<B> {
    fn forward(
        &mut self,
        x: Rank3Tensor<BB, U1500, U384, B>,
        flush_cache: bool,
    ) -> Result<Rank3Tensor<BB, U1500, U384, B>, Error> {
        let x = x.clone()
            + self.attn.forward(
                self.attn_ln.forward(x.into_inner()).try_into()?,
                None,
                false,
                flush_cache,
            )?;

        Ok(x.clone()
            + self
                .mlp
                .forward(self.mlp_ln.forward(x.into_inner()).try_into()?)?)
    }

    fn reset_kv_cache(&mut self) {
        self.attn.reset_kv_cache();
    }
}

#[derive(Config)]
pub struct ResidualDecoderAttentionBlockConfig {
    n_state: usize,
    n_head: usize,
}

impl ResidualDecoderAttentionBlockConfig {
    pub fn init<B: Backend>(
        &self,
        tensor_device_ref: &B::Device,
        timestamps: bool,
    ) -> ResidualDecoderAttentionBlock<B> {
        let attn =
            MultiHeadSelfAttentionConfig::new(self.n_state, self.n_head).init(tensor_device_ref);
        let attn_ln = nn::LayerNormConfig::new(self.n_state).init(tensor_device_ref);

        let cross_attn = MultiHeadCrossAttentionConfig::new(self.n_state, self.n_head)
            .init(tensor_device_ref, timestamps);
        let cross_attn_ln = nn::LayerNormConfig::new(self.n_state).init(tensor_device_ref);

        let mlp = MLPConfig::new(self.n_state).init(tensor_device_ref);
        let mlp_ln = nn::LayerNormConfig::new(self.n_state).init(tensor_device_ref);

        ResidualDecoderAttentionBlock {
            attn,
            attn_ln,
            cross_attn,
            cross_attn_ln,
            mlp,
            mlp_ln,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualDecoderAttentionBlock<B: Backend> {
    attn: MultiHeadSelfAttention<B>,
    attn_ln: nn::LayerNorm<B>,
    cross_attn: MultiHeadCrossAttention<B>,
    cross_attn_ln: nn::LayerNorm<B>,
    mlp: MLP<B>,
    mlp_ln: nn::LayerNorm<B>,
}

impl<B: Backend> ResidualDecoderAttentionBlock<B> {
    fn forward(
        &mut self,
        x: Rank3Tensor<BB, L, U384, B>,
        xa: Rank3Tensor<BB, U1500, U384, B>,
        mask: Rank2Tensor<U448, U448, B>,
        enable_self_attn_kv_cache: bool,
        attention_output: Option<&mut TensorCache<B, 4>>,
        flush_cache: bool,
    ) -> Result<Rank3Tensor<BB, L, U384, B>, Error> {
        let x = x.clone()
            + self.attn.forward(
                self.attn_ln.forward(x.into_inner()).try_into()?,
                Some(mask),
                enable_self_attn_kv_cache,
                flush_cache,
            )?;
        let x = x.clone()
            + self.cross_attn.forward(
                self.cross_attn_ln.forward(x.into_inner()).try_into()?,
                xa,
                attention_output,
                flush_cache,
            )?;
        Ok(x.clone()
            + self
                .mlp
                .forward(self.mlp_ln.forward(x.into_inner()).try_into()?)?)
    }

    fn reset_kv_cache(&mut self) {
        self.attn.reset_kv_cache();
        self.cross_attn.reset_kv_cache();
    }
}

#[derive(Config)]
pub struct MLPConfig {
    n_state: usize,
}

impl MLPConfig {
    pub fn init<B: Backend>(&self, tensor_device_ref: &B::Device) -> MLP<B> {
        let lin1 = nn::LinearConfig::new(self.n_state, 4 * self.n_state).init(tensor_device_ref);
        let gelu = nn::Gelu::new();
        let lin2 = nn::LinearConfig::new(4 * self.n_state, self.n_state).init(tensor_device_ref);

        MLP { lin1, gelu, lin2 }
    }
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    lin1: nn::Linear<B>,
    gelu: nn::Gelu,
    lin2: nn::Linear<B>,
}

impl<B: Backend> MLP<B> {
    pub fn forward(
        &self,
        x: Rank3Tensor<BB, L, U384, B>,
    ) -> Result<Rank3Tensor<BB, L, U384, B>, Error> {
        let x = self.lin1.forward(x.into_inner());
        let x = self.gelu.forward(x);
        self.lin2.forward(x).try_into().map_err(Into::into)
    }
}

#[derive(Config)]
pub struct MultiHeadSelfAttentionConfig {
    n_state: usize,
    n_head: usize,
}

impl MultiHeadSelfAttentionConfig {
    fn init<B: Backend>(&self, tensor_device_ref: &B::Device) -> MultiHeadSelfAttention<B> {
        assert!(
            self.n_state % self.n_head == 0,
            "State size {} must be a multiple of head size {}",
            self.n_state,
            self.n_head
        );

        let n_head = self.n_head;
        let query = nn::LinearConfig::new(self.n_state, self.n_state).init(tensor_device_ref);
        let key = nn::LinearConfig::new(self.n_state, self.n_state)
            .with_bias(false)
            .init(tensor_device_ref);
        let value = nn::LinearConfig::new(self.n_state, self.n_state).init(tensor_device_ref);
        let out = nn::LinearConfig::new(self.n_state, self.n_state).init(tensor_device_ref);

        MultiHeadSelfAttention {
            n_head,
            query,
            key,
            value,
            out,
            kv_cache: None,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadSelfAttention<B: Backend> {
    n_head: usize,
    query: nn::Linear<B>,
    key: nn::Linear<B>,
    value: nn::Linear<B>,
    out: nn::Linear<B>,
    kv_cache: Option<(BTensor<B, 3>, BTensor<B, 3>)>,
}

impl<B: Backend> MultiHeadSelfAttention<B> {
    pub fn forward(
        &mut self,
        x: Rank3Tensor<BB, L, U384, B>,
        mask: Option<Rank2Tensor<U448, U448, B>>,
        enable_self_attn_kv_cache: bool,
        flush_cache: bool,
    ) -> Result<Rank3Tensor<BB, L, U384, B>, Error> {
        let q = self.query.forward(x.clone().into_inner());
        if flush_cache {
            self.kv_cache = None;
        }

        let mut k = self.key.forward(x.clone().into_inner());
        let mut v = self.value.forward(x.into_inner());
        if enable_self_attn_kv_cache {
            if let Some((ks, vs)) = self.kv_cache.take() {
                k = BTensor::cat(vec![k, ks], 1);
                v = BTensor::cat(vec![v, vs], 1);
            }
            self.kv_cache = Some((k.clone(), v.clone()));
        }

        let wv = self.qkv_attention(
            q.try_into()?,
            k.try_into()?,
            v.try_into()?,
            mask,
            self.n_head,
        )?;
        self.out
            .forward(wv.into_inner())
            .try_into()
            .map_err(Into::into)
    }

    fn reshape_head(
        &self,
        x: Rank3Tensor<BB, L, U384, B>,
    ) -> Result<Rank4Tensor<BB, U6, L, U64, B>, Error> {
        let [n_batch, n_ctx, n_state] = x.inner().dims();
        let reshaped: Rank4Tensor<BB, L, U6, U64, B> = x
            .into_inner()
            .reshape([n_batch, n_ctx, self.n_head, n_state / self.n_head])
            .try_into()?;

        Ok(transpose!(reshaped, U1, U2))
    }

    pub fn qkv_attention(
        &mut self,
        q: Rank3Tensor<BB, L, U384, B>,
        k: Rank3Tensor<BB, L, U384, B>,
        v: Rank3Tensor<BB, L, U384, B>,
        mask: Option<Rank2Tensor<U448, U448, B>>,
        n_head: usize,
    ) -> Result<Rank3Tensor<BB, L, U384, B>, Error> {
        let [_, n_ctx, n_state] = q.inner().dims();
        let scale = (n_state as f64 / n_head as f64).powf(-0.25);
        let q = self.reshape_head(q)? * scale;
        let k = transpose!(self.reshape_head(k)?, U2, U3) * scale;
        let v = self.reshape_head(v)?;

        let qk = matmul!(q, k);

        let qk = if let Some(mask) = mask {
            let mask = mask.try_slice::<Shape2<L, L>, _, 2>([0..n_ctx, 0..n_ctx])?;
            expand!(mask, &qk) + qk
        } else {
            qk
        };
        let w = softmax!(qk, U3);

        Ok(flatten!(transpose!(matmul!(w, v), U1, U2), U2, U3))
    }

    fn reset_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

#[derive(Config)]
pub struct MultiHeadCrossAttentionConfig {
    n_state: usize,
    n_head: usize,
}

impl MultiHeadCrossAttentionConfig {
    fn init<B: Backend>(
        &self,
        tensor_device_ref: &B::Device,
        timestamps: bool,
    ) -> MultiHeadCrossAttention<B> {
        assert!(
            self.n_state % self.n_head == 0,
            "State size {} must be a multiple of head size {}",
            self.n_state,
            self.n_head
        );

        let n_head = self.n_head;
        let query = nn::LinearConfig::new(self.n_state, self.n_state).init(tensor_device_ref);
        let key = nn::LinearConfig::new(self.n_state, self.n_state)
            .with_bias(false)
            .init(tensor_device_ref);
        let value = nn::LinearConfig::new(self.n_state, self.n_state).init(tensor_device_ref);
        let out = nn::LinearConfig::new(self.n_state, self.n_state).init(tensor_device_ref);

        MultiHeadCrossAttention {
            n_head,
            query,
            key,
            value,
            out,
            kv_cache: None,
            output_attentions: if timestamps { Some(None) } else { None },
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadCrossAttention<B: Backend> {
    n_head: usize,
    query: nn::Linear<B>,
    key: nn::Linear<B>,
    value: nn::Linear<B>,
    out: nn::Linear<B>,
    kv_cache: Option<(BTensor<B, 3>, BTensor<B, 3>)>,
    output_attentions: Option<Option<BTensor<B, 4>>>,
}

impl<B: Backend> MultiHeadCrossAttention<B> {
    pub fn forward(
        &mut self,
        x: Rank3Tensor<BB, L, U384, B>,
        xa: Rank3Tensor<BB, U1500, U384, B>,
        attention_output: Option<&mut TensorCache<B, 4>>,
        flush_cache: bool,
    ) -> Result<Rank3Tensor<BB, L, U384, B>, Error> {
        let q = self.query.forward(x.clone().into_inner());
        let (k, v) = {
            if flush_cache {
                self.kv_cache = None;
            }
            if let Some((k, v)) = &mut self.kv_cache {
                let (a, b) = (k.dims()[0], xa.inner().dims()[0]);
                if a < b {
                    *k = k.clone().repeat(&[b / a, 0, 0]);
                    *v = v.clone().repeat(&[b / a, 0, 0]);
                }

                (k.clone(), v.clone())
            } else {
                let k = self.key.forward(xa.clone().into_inner());
                let v = self.value.forward(xa.into_inner());
                self.kv_cache = Some((k.clone(), v.clone()));
                (k, v)
            }
        };

        let wv = self
            .qkv_attention(
                q.try_into()?,
                k.try_into()?,
                v.try_into()?,
                self.n_head,
                attention_output,
            )?
            .into_inner();

        Ok(self.out.forward(wv).try_into()?)
    }

    fn reshape_head(
        &self,
        x: Rank3Tensor<BB, C, St, B>,
    ) -> Result<Rank4Tensor<BB, U6, C, SH, B>, Error> {
        let [n_batch, n_ctx, n_state] = x.inner().dims();
        let reshaped: Rank4Tensor<BB, C, U6, SH, B> = x
            .into_inner()
            .reshape([n_batch, n_ctx, self.n_head, n_state / self.n_head])
            .try_into()?;
        Ok(transpose!(reshaped, U1, U2))
    }

    pub fn qkv_attention(
        &mut self,
        q: Rank3Tensor<BB, L, U384, B>,
        k: Rank3Tensor<BB, U1500, U384, B>,
        v: Rank3Tensor<BB, U1500, U384, B>,
        n_head: usize,
        attention_output: Option<&mut TensorCache<B, 4>>,
    ) -> Result<Rank3Tensor<U2, L, U384, B>, Error> {
        let [_, _, n_state] = q.inner().dims();
        let scale = (n_state as f64 / n_head as f64).powf(-0.25);

        let q = self.reshape_head(q.transmute())? * scale;
        let k = transpose!(self.reshape_head(k.transmute())?, U2, U3) * scale;
        let v = self.reshape_head(v.transmute())?;
        let qk = matmul!(q, k);

        if let Some(out) = attention_output {
            let _ = out.append(qk.clone().into_inner());
        }
        let w = softmax!(qk, U3);
        Ok(flatten!(transpose!(matmul!(w, v), U1, U2), U2, U3).transmute())
    }

    fn reset_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}
