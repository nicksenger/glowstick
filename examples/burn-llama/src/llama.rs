use std::time::Instant;

use burn::{
    config::Config,
    module::Module,
    nn::{RotaryEncoding, RotaryEncodingConfig},
    record::{FileRecorder, HalfPrecisionSettings, RecorderError},
    tensor::{
        backend::Backend as BurnBackend, cast::ToElement, Device, ElementConversion, Shape,
        Tensor as BurnTensor, TensorData,
    },
};
use glowstick::num::{Unsigned, U0, U1};
use glowstick_burn::{cat, narrow, reshape, softmax, squeeze};

use crate::{
    sampling::Sampler,
    shape::*,
    tokenizer::{Tiktoken, Tokenizer},
    transformer::{KeyValueCache, Transformer, TransformerConfig},
    Error,
};

use crate::pretrained::{self, ModelMeta};

#[derive(Config, Debug)]
pub struct LlamaConfig {
    /// The size of the model.
    #[config(default = "4096")]
    pub d_model: usize,
    /// The size of the feed-forward hidden inner features.
    pub hidden_size: usize,
    /// The number of transformer blocks.
    #[config(default = "32")]
    pub num_hidden_layers: usize,
    /// The number of attention heads.
    #[config(default = "32")]
    pub num_attention_heads: usize,
    /// The number of key-value heads.
    pub num_key_value_heads: Option<usize>,
    /// The vocabulary size.
    pub vocab_size: usize,
    /// RMSNorm epsilon
    #[config(default = "1e-5")]
    pub norm_eps: f64,
    /// Rotary positional encoding (RoPE).
    #[config(default = "RopeConfig::new(10000.0)")]
    pub rope: RopeConfig,
    /// Maximum sequence length for input text.
    #[config(default = "128")]
    pub max_seq_len: usize,
    /// Maximum batch size (used for key-value cache).
    #[config(default = "1")]
    pub max_batch_size: usize,
    /// The tokenizer path.
    pub tokenizer: String,
}

/// Rotary positional encoding (RoPE)
#[derive(Config, Debug)]
pub struct RopeConfig {
    pub theta: f32,
    #[config(default = "None")]
    pub scaled: Option<RopeFrequencyScaling>,
}

/// RoPE frequency scaling.
#[derive(Config, Debug)]
pub struct RopeFrequencyScaling {
    #[config(default = "8.")]
    pub scale_factor: f32,
    #[config(default = "1.")]
    pub low_freq_factor: f32,
    #[config(default = "4.")]
    pub high_freq_factor: f32,
    #[config(default = "8192.")]
    pub old_context_len: f32,
}

impl LlamaConfig {
    pub fn with_tokenizer(tokenizer_path: &str) -> Self {
        Self::new(
            <F as Unsigned>::USIZE,
            <C as Unsigned>::USIZE,
            tokenizer_path.to_string(),
        )
        .with_d_model(<S as Unsigned>::USIZE)
        .with_num_attention_heads(<A as Unsigned>::USIZE)
        .with_num_hidden_layers(NUM_HIDDEN_LAYERS)
        .with_num_key_value_heads(Some(<K as Unsigned>::USIZE))
        .with_rope(
            RopeConfig::new(500000.0)
                .with_scaled(Some(RopeFrequencyScaling::new().with_scale_factor(32.))),
        )
    }

    /// Initialize a new [Llama](Llama) module.
    pub fn init<Backend: BurnBackend, T: Tokenizer>(
        &self,
        device: &Device<Backend>,
    ) -> Result<Llama<Backend, T>, String> {
        let tokenizer = T::new(&self.tokenizer)?;
        let num_key_value_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);
        let model = TransformerConfig::new(
            self.vocab_size,
            self.num_hidden_layers,
            self.d_model,
            self.hidden_size,
            self.num_attention_heads,
            num_key_value_heads,
        )
        .with_max_seq_len(self.max_seq_len)
        .with_norm_eps(self.norm_eps)
        .init(device);

        let cache = (0..self.num_hidden_layers)
            .map(|_| {
                KeyValueCache::new(
                    self.max_batch_size,
                    num_key_value_heads,
                    self.max_seq_len,
                    self.d_model / self.num_attention_heads,
                    device,
                )
            })
            .collect::<Vec<_>>();

        let rope = RotaryEncodingConfig::new(
            self.max_seq_len * 2,
            self.d_model / self.num_attention_heads,
        )
        .with_theta(self.rope.theta);

        let rope = if let Some(scaling) = &self.rope.scaled {
            let freq_scaling_fn = move |x| scaling.freq_scaling_by_parts(x);
            rope.init_with_frequency_scaling(freq_scaling_fn, device)
        } else {
            rope.init(device)
        };

        Ok(Llama {
            tokenizer,
            model,
            cache,
            rope,
            device: device.clone(),
        })
    }
    pub fn load_llama<Backend: BurnBackend>(
        checkpoint: &str,
        tokenizer_path: &str,
        max_seq_len: usize,
        device: &Device<Backend>,
    ) -> Result<Llama<Backend, Tiktoken>, String> {
        use burn::record::NamedMpkFileRecorder;

        let llama = Self::with_tokenizer(tokenizer_path)
            .with_max_seq_len(max_seq_len)
            .init::<Backend, Tiktoken>(device)?;

        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
        let llama = llama
            .load(checkpoint, &recorder)
            .map_err(|err| format!("Failed to load pre-trained Llama model.\nError: {err}"))?;

        Ok(llama)
    }

    /// Load pre-trained Llama-3.2 model with [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    ///
    /// # Arguments
    /// - `max_seq_len` - The maximum sequence length for input text.
    /// - `device` - The device to load the model on.
    pub fn llama3_2_pretrained<Backend: BurnBackend>(
        max_seq_len: usize,
        device: &Device<Backend>,
    ) -> Result<Llama<Backend, Tiktoken>, String> {
        // Llama-3.2 models support context length up to 128K tokens.
        check_context_length(max_seq_len, 128 * 1024);

        // Download checkpoint and tokenizer
        #[cfg(not(feature = "3b"))]
        let model = pretrained::Llama::Llama321bInstruct.pretrained();
        #[cfg(feature = "3b")]
        let model = pretrained::Llama::Llama323bInstruct.pretrained();

        let checkpoint = model
            .download_weights()
            .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
        let tokenizer = model
            .download_tokenizer()
            .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

        Self::load_llama(
            checkpoint.to_str().unwrap(),
            tokenizer.to_str().unwrap(),
            max_seq_len,
            device,
        )
    }
}

fn check_context_length(max_seq_len: usize, max_context_len: usize) {
    assert!(
        max_seq_len <= max_context_len,
        "Maximum sequence length must not exceed {max_context_len}"
    );
}

/// Generated text sample output.
pub struct GenerationOutput {
    /// The generated text.
    pub text: String,
    /// The number of generated tokens.
    pub tokens: usize,
    /// The time it took to produce the output tokens (generation + decoding).
    pub time: f64,
}

/// Meta Llama large language model and tokenizer.
pub struct Llama<Backend: BurnBackend, T: Tokenizer> {
    /// The tokenizer.
    pub tokenizer: T,
    /// Llama decoder-only transformer.
    pub model: Transformer<Backend>,
    /// Key-value cache for each transformer block.
    pub cache: Vec<KeyValueCache<Backend>>,
    /// Rotary positional encoding (RoPE).
    pub rope: RotaryEncoding<Backend>,
    pub device: Device<Backend>,
}

impl<Backend: BurnBackend, T: Tokenizer> Llama<Backend, T> {
    pub fn generate(
        &mut self,
        prompt: &str,
        sample_len: usize,
        temperature: f64,
        sampler: &mut Sampler,
    ) -> Result<GenerationOutput, Error> {
        let mut tokens = self.tokenize(prompt);
        let prompt_len = tokens.dims()[0];
        let stop_tokens = BurnTensor::from_ints(self.tokenizer.stop_ids().as_slice(), &self.device);

        let mut num_tokens: usize = 0;
        let mut input_pos =
            Rank1IntTensor::<N, Backend>::arange(0..prompt_len as i64, &self.device);
        let now = Instant::now();
        for i in 0..sample_len {
            let ctx = if i == 0 { prompt_len } else { 1 };
            let x = narrow!(tokens.clone(), U0: [if i == 0 { 0 } else { prompt_len + num_tokens - 1 }, { ctx }] => N);
            let x = reshape!(x, [B, { ctx as i32 } => N]);
            let logits: Rank3Tensor<B, N, C, Backend> =
                self.model.forward(x, &mut self.cache, &self.rope)?;

            let [batch_size, seq_len, _vocab_size] = logits.dims();
            let next_token_logits = narrow!(
                logits,
                U0: [{ 0 }, { batch_size }] => B,
                U1: [{ seq_len - 1 }, U1]
            );
            let mut next_token_logits = squeeze!(next_token_logits, U1);

            if temperature > 0.0 {
                next_token_logits = temperature_scaled_softmax(next_token_logits, temperature);
            };

            let sampled = sampler.sample(next_token_logits);
            let next_token = squeeze!(sampled, U0);

            // Stop when any of the valid stop tokens is encountered
            if stop_tokens
                .clone()
                .equal(next_token.clone().into_inner())
                .any()
                .into_scalar()
                .to_bool()
            {
                break;
            }

            // Update with the new generated token
            tokens = cat!(vec![tokens, next_token.into_inner().try_into()?], U0 => N);
            num_tokens += 1;

            // Advance
            let t = input_pos.dims()[0];
            input_pos = narrow!(input_pos, U0: [{ t - 1 }, { 1 }] => N);
        }

        let tokens = tokens.into_data().as_slice::<Backend::IntElem>().unwrap()
            [prompt_len..prompt_len + num_tokens]
            .iter()
            .map(|t| t.elem::<u32>())
            .collect::<Vec<_>>();

        let generated = self.tokenizer.decode(tokens);
        let elapsed = now.elapsed().as_secs_f64();

        Ok(GenerationOutput {
            text: generated,
            tokens: num_tokens,
            time: elapsed,
        })
    }

    /// Encode a string into a tensor of tokens.
    fn tokenize(&self, text: &str) -> Rank1IntTensor<N, Backend> {
        let tokens = self.tokenizer.encode(text, true, false);

        let shape = Shape::new([tokens.len()]);
        Rank1IntTensor::<N, Backend>::from_ints(TensorData::new(tokens, shape), &self.device)
    }

    /// Save Llama model to file using the specified recorder.
    pub fn save<R: FileRecorder<Backend>>(
        self,
        file_path: &str,
        recorder: &R,
    ) -> Result<(), RecorderError> {
        println!("Saving record...");
        let now = Instant::now();
        self.model.save_file(file_path, recorder)?;
        let elapsed = now.elapsed().as_secs();
        println!("Saved in {}s", elapsed);

        Ok(())
    }

    /// Load Llama model from file using the specified recorder.
    pub fn load<R: FileRecorder<Backend>>(
        mut self,
        file_path: &str,
        recorder: &R,
    ) -> Result<Self, RecorderError> {
        println!("Loading record...");
        let now = Instant::now();
        self.model = self.model.load_file(file_path, recorder, &self.device)?;
        let elapsed = now.elapsed().as_secs();
        println!("Loaded in {}s", elapsed);

        Ok(self)
    }

    /// Reset the model state (used between generations)
    pub fn reset(&mut self) {
        self.cache.iter_mut().for_each(|cache| cache.reset());
    }
}

impl RopeFrequencyScaling {
    /// Applies frequency scaling by parts following Llama 3.1's scheme.
    ///
    /// Adapted from: https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py#L45
    pub fn freq_scaling_by_parts<Backend: BurnBackend>(
        &self,
        freqs: BurnTensor<Backend, 1>,
    ) -> BurnTensor<Backend, 1> {
        let low_freq_wavelen = self.old_context_len / self.low_freq_factor;
        let high_freq_wavelen = self.old_context_len / self.high_freq_factor;

        let wavelen = freqs.clone().recip().mul_scalar(2. * core::f32::consts::PI);

        // if wavelen >= high_freq_wavelen
        let cond = wavelen.clone().greater_equal_elem(high_freq_wavelen);
        let smooth = wavelen
            .clone()
            .recip()
            .mul_scalar(self.old_context_len)
            .sub_scalar(self.low_freq_factor)
            .div_scalar(self.high_freq_factor - self.low_freq_factor);
        // (1 - smooth) * freq / scale_factor + smooth * freq
        let new_freqs = smooth
            .clone()
            .neg()
            .add_scalar(1.)
            .mul(freqs.clone().div_scalar(self.scale_factor))
            .add(smooth.clone().mul(freqs.clone()));
        let new_freqs = freqs.clone().mask_where(cond, new_freqs);

        // if wavelen > low_freq_wavelen
        let cond = wavelen.clone().greater_elem(low_freq_wavelen);
        let new_freqs = new_freqs.mask_where(cond, freqs.clone().div_scalar(self.scale_factor));

        // if wavelen < high_freq_wavelen
        let cond = wavelen.lower_elem(high_freq_wavelen);
        new_freqs.mask_where(cond, freqs)
    }
}

pub(crate) fn temperature_scaled_softmax<Backend: BurnBackend>(
    logits: Rank2Tensor<B, C, Backend>,
    temperature: f64,
) -> Rank2Tensor<B, C, Backend> {
    softmax!(logits / temperature, U1)
}
