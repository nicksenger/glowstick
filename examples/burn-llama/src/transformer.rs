use burn::{
    config::Config,
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig, RotaryEncoding,
        SwiGlu, SwiGluConfig,
    },
    tensor::{backend::Backend, Device},
};
use glowstick::num::{U1, U2, U3};
use glowstick_burn::{expand, matmul, reshape, softmax, transpose, tril_mask, unsqueeze};

// Using BS for Batch-Size here, as Burn's `Module` proc-macro expects B for the backend
use crate::cache::AutoregressiveCache;
use crate::shape::{Rank2IntTensor, Rank3Tensor, Rank4Tensor, A, B as BS, C, H, K, KV, N, Q, R, S};

/// Configuration to create a Llama [decoder-only transformer](Transformer).
#[derive(Config)]
pub struct TransformerConfig {
    /// The size of the vocabulary.
    pub vocab_size: usize,
    /// The number of transformer blocks.
    pub n_layers: usize,
    /// The size of the model.
    pub d_model: usize,
    /// The size of the feed-forward hidden inner features.
    pub hidden_size: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// The number of key-value heads.
    pub n_kv_heads: usize,
    /// Maximum token sequence length.
    #[config(default = "512")]
    pub max_seq_len: usize,
    /// RMSNorm epsilon.
    #[config(default = "1e-5")]
    pub norm_eps: f64,
}

impl TransformerConfig {
    /// Initialize a new [decoder-only transformer](Transformer).
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Transformer<B> {
        let tok_embeddings = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        let layers = (0..self.n_layers)
            .map(|_| {
                TransformerBlockConfig::new(
                    self.n_layers,
                    self.d_model,
                    self.hidden_size,
                    self.n_heads,
                    self.n_kv_heads,
                    self.norm_eps,
                )
                .init(device)
            })
            .collect::<Vec<_>>();
        let norm = RmsNormConfig::new(self.d_model)
            .with_epsilon(self.norm_eps)
            .init(device);
        let output = LinearConfig::new(self.d_model, self.vocab_size)
            .with_bias(false)
            .init(device);

        Transformer {
            tok_embeddings,
            layers,
            norm,
            output,
        }
    }
}

/// Llama decoder-only transformer.
#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    tok_embeddings: Embedding<B>,
    layers: Vec<TransformerBlock<B>>,
    norm: RmsNorm<B>,
    // NOTE: Starting with Llama 3.2, the weights of the output layer are tied with the embedding
    output: Linear<B>,
}

impl<Backend: burn::tensor::backend::Backend> Transformer<Backend> {
    pub fn forward(
        &self,
        input: Rank2IntTensor<BS, N, Backend>,
        cache: &mut Vec<KeyValueCache<Backend>>,
        rope: &RotaryEncoding<Backend>,
    ) -> Result<Rank3Tensor<BS, N, C, Backend>, crate::Error> {
        let mut h: Rank3Tensor<BS, N, S, Backend> =
            self.tok_embeddings.forward(input.into_inner()).try_into()?;

        for (layer, cache) in self.layers.iter().zip(cache.into_iter()) {
            h = layer.forward(h, cache, rope).unwrap();
        }

        let h = self.norm.forward(h.into_inner());
        Ok(self.output.forward(h).try_into()?)
    }
}

/// Configuration to create a [decoder-only transformer block](TransformerBlock).
#[derive(Config)]
pub struct TransformerBlockConfig {
    /// The number of transformer blocks.
    pub n_layers: usize,
    /// The size of the model.
    pub d_model: usize,
    /// The size of the feed-forward hidden inner features.
    pub hidden_size: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// The number of key-value heads.
    pub n_kv_heads: usize,
    /// RMSNorm epsilon.
    pub norm_eps: f64,
}

impl TransformerBlockConfig {
    /// Initialize a new [decoder-only transformer block](TransformerBlock).
    pub fn init<B: Backend>(&self, device: &Device<B>) -> TransformerBlock<B> {
        let attention =
            MultiHeadAttentionConfig::new(self.d_model, self.n_heads, self.n_kv_heads).init(device);
        let feed_forward = FeedForwardConfig::new(self.d_model, self.hidden_size).init(device);
        let attention_norm = RmsNormConfig::new(self.d_model)
            .with_epsilon(self.norm_eps)
            .init(device);
        let ffn_norm = RmsNormConfig::new(self.d_model)
            .with_epsilon(self.norm_eps)
            .init(device);

        TransformerBlock {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        }
    }
}

/// Decoder-only transformer block.
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    /// Self-attention.
    attention: MultiHeadAttention<B>,
    /// Feed-forward transformation.
    feed_forward: FeedForward<B>,
    /// Attention pre-normalization.
    attention_norm: RmsNorm<B>,
    /// Feed-forward pre-normalization.
    ffn_norm: RmsNorm<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn forward(
        &self,
        input: Rank3Tensor<BS, N, S, B>,
        cache: &mut KeyValueCache<B>,
        rope: &RotaryEncoding<B>,
    ) -> Result<Rank3Tensor<BS, N, S, B>, crate::Error> {
        let h: Rank3Tensor<BS, N, S, B> = input.clone()
            + self.attention.forward(
                self.attention_norm.forward(input.into_inner()).try_into()?,
                cache,
                rope,
            )?;
        let y: Rank3Tensor<BS, N, S, B> = self
            .feed_forward
            .forward(self.ffn_norm.forward(h.clone().into_inner()).try_into()?)?;
        Ok(h + y)
    }
}

/// Configuration to create a [feed-forward transformation network](FeedForward).
#[derive(Config)]
pub struct FeedForwardConfig {
    /// The size of the model.
    pub d_model: usize,
    /// The size of the hidden inner features.
    pub hidden_size: usize,
}

impl FeedForwardConfig {
    /// Initialize a new [feed-forward transformation network](FeedForward).
    pub fn init<B: Backend>(&self, device: &Device<B>) -> FeedForward<B> {
        let swiglu = SwiGluConfig::new(self.d_model, self.hidden_size)
            .with_bias(false)
            .init(device);
        let w2 = LinearConfig::new(self.hidden_size, self.d_model)
            .with_bias(false)
            .init(device);

        FeedForward { swiglu, w2 }
    }
}

/// Feed-forward transformation network.
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    // Swish gated linear unit with trainable parameters.
    swiglu: SwiGlu<B>,
    /// Outer linear.
    w2: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, seq_length, d_model]`
    /// - output: `[batch_size, seq_length, d_model]`
    pub fn forward(
        &self,
        input: Rank3Tensor<BS, N, S, B>,
    ) -> Result<Rank3Tensor<BS, N, S, B>, crate::Error> {
        Ok(self
            .w2
            .forward(self.swiglu.forward(input.into_inner()))
            .try_into()?)
    }
}

/// Key-value cache for autoregressive models.
pub struct KeyValueCache<B: Backend> {
    key: AutoregressiveCache<B>,
    value: AutoregressiveCache<B>,
}

type KVCacheTensor<B> = Rank4Tensor<BS, K, N, H, B>;
impl<B: Backend> KeyValueCache<B> {
    /// Create a new [key-value cache](KeyValueCache).
    pub fn new(
        max_batch_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        d_model: usize,
        device: &Device<B>,
    ) -> Self {
        Self {
            key: AutoregressiveCache::new(max_batch_size, num_heads, max_seq_len, d_model, device),
            value: AutoregressiveCache::new(
                max_batch_size,
                num_heads,
                max_seq_len,
                d_model,
                device,
            ),
        }
    }

    /// Computes the complete keys and values.
    pub fn forward(
        &mut self,
        key: KVCacheTensor<B>,
        value: KVCacheTensor<B>,
    ) -> Result<(KVCacheTensor<B>, KVCacheTensor<B>), crate::Error> {
        let k = self.key.forward(key);
        let v = self.value.forward(value);
        Ok((k, v))
    }

    /// Returns the cached sequence length.
    pub fn len(&self) -> usize {
        // We can assume key and value have the same length
        self.key.len()
    }

    /// Reset key-value cache.
    /// Use between different contexts (i.e., for each new prompt).
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.key.reset();
        self.value.reset();
    }
}

/// Configuration to create a [multi-head attention](MultiHeadAttention) module.
#[derive(Config)]
pub struct MultiHeadAttentionConfig {
    /// The size of the model.
    pub d_model: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// The number of key-value heads.
    pub n_kv_heads: usize,
}

impl MultiHeadAttentionConfig {
    /// Initialize a new [multi-head attention](MultiHeadAttention) module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> MultiHeadAttention<B> {
        let head_dim = self.d_model / self.n_heads;

        let wq = LinearConfig::new(self.d_model, self.n_heads * head_dim)
            .with_bias(false)
            .init(device);
        let wk = LinearConfig::new(self.d_model, self.n_kv_heads * head_dim)
            .with_bias(false)
            .init(device);
        let wv = LinearConfig::new(self.d_model, self.n_kv_heads * head_dim)
            .with_bias(false)
            .init(device);
        let wo = LinearConfig::new(self.n_heads * head_dim, self.d_model)
            .with_bias(false)
            .init(device);

        MultiHeadAttention {
            wq,
            wk,
            wv,
            wo,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    /// Query projection.
    wq: Linear<B>,
    /// Key projection.
    wk: Linear<B>,
    /// Value projection.
    wv: Linear<B>,
    /// Output projection.
    wo: Linear<B>,

    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl<B: Backend> MultiHeadAttention<B> {
    /// Applies the forward pass on the input tensors.
    ///
    /// # Shapes
    ///
    /// - query: `[batch_size, seq_length_1, d_model]`
    /// - key: `[batch_size, seq_length_2, d_model]`
    /// - value: `[batch_size, seq_length_2, d_model]`
    /// - output: `[batch_size, seq_length_1, d_model]`
    pub fn forward(
        &self,
        input: Rank3Tensor<BS, N, S, B>,
        cache: &mut KeyValueCache<B>,
        rope: &RotaryEncoding<B>,
    ) -> Result<Rank3Tensor<BS, N, S, B>, crate::Error> {
        let device = input.device();
        let [_batch_size, seq_len, _hidden_size] = input.dims();

        let q: Rank3Tensor<BS, N, Q, B> = self.wq.forward(input.clone().into_inner()).try_into()?;
        let k: Rank3Tensor<BS, N, KV, B> =
            self.wk.forward(input.clone().into_inner()).try_into()?;
        let v: Rank3Tensor<BS, N, KV, B> = self.wv.forward(input.into_inner()).try_into()?;

        // [batch_size, num_heads, seq_len, head_dim]
        let q = transpose!(reshape!(q, [BS, { seq_len as i32 } => N, A, H]), U1:U2);
        let k = transpose!(reshape!(k, [BS, { seq_len as i32 } => N, K, H]), U1:U2);
        let v = transpose!(reshape!(v, [BS, { seq_len as i32 } => N, K, H]), U1:U2);

        let cache_seq_len = cache.len();

        let q: Rank4Tensor<BS, A, N, H, B> =
            rope.apply(q.into_inner(), cache_seq_len).try_into()?;
        let k = rope.apply(k.into_inner(), cache_seq_len);

        // Key-value caching
        let (k, v) = cache.forward(k.try_into()?, v)?;

        // Repeat key/value heads if num_kv_heads < num_heads
        let k = self.repeat_kv(k);
        let v = self.repeat_kv(v);

        // Attention scores
        let mut scores = matmul!(q, transpose!(k, U2:U3)) / (self.head_dim as f32).sqrt();

        // Matrix of scores is of size [seqlen, cache_len + seqlen], and the only masked entries are
        // (i, j) for j > cache_len + i, since row i corresponds to token cache_len + i.
        // NOTE: we could possibly improve the mask generation by caching masks for different sequence lengths,
        // though it is probably not necessary at this time.
        if seq_len > 1 {
            let cache_seq_len = cache.len();
            let mask = tril_mask!((cache_seq_len - seq_len) as i64, &device, B, [{ seq_len } => N, { cache_seq_len } => N]);
            let mask = expand!(mask, &scores);
            scores = scores.mask_fill(mask, f32::NEG_INFINITY);
        }

        let scores = softmax!(scores, U3);

        // Output [batch_size, num_heads, seq_len, head_dim]
        let output = matmul!(scores, v);
        let output = transpose!(output, U1:U2);
        let output = reshape!(output, [BS, { seq_len as i32 } => N, S]);
        Ok(self.wo.forward(output.into_inner()).try_into()?)
    }

    /// Repeats a key or value tensor for grouped query attention.
    fn repeat_kv(&self, x: Rank4Tensor<BS, K, N, H, B>) -> Rank4Tensor<BS, A, N, H, B> {
        let n_rep = self.n_heads / self.n_kv_heads;
        if n_rep == 1 {
            // # attn heads == kv heads
            x.into_inner().try_into().unwrap()
        } else {
            let [_batch_size, _num_kv_heads, seq_len, _head_dim] = x.dims();

            let x = unsqueeze!(x, U2);
            let x = expand!(x, [BS, K, R, { seq_len as i32 } => N, H]);
            reshape!(x, [BS, A, { seq_len as i32 } => N, H])
        }
    }
}
