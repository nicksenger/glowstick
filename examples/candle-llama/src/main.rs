use anyhow::{Error as E, Result};
use clap::Parser;

use candle::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use glowstick::{
    num::{U0, U1},
    Shape2,
};
use glowstick_candle::tensor::Tensor;
use glowstick_candle::{cat, narrow, squeeze};
use hf_hub::{api::sync::Api, Repo, RepoType};
use llama::{Llama, LlamaConfig, LlamaEosToks};
use tokenizers::Tokenizer;

mod llama;
mod shape;

use shape::*;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Rank mismatch: runtime ({runtime}) vs type-level ({type_level})")]
    RankMismatch { runtime: usize, type_level: usize },

    #[error("Dimension mismatch: expected {type_level} for dim {dim} but received {runtime}")]
    DimensionMismatch {
        dim: usize,
        runtime: usize,
        type_level: usize,
    },

    #[error("Couldn't find the EOS token.")]
    MissingEosToken,

    #[error("No token streams!")]
    NoTokenStreams,

    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),

    #[error("Encode error: {0}")]
    Encode(String),

    #[error("{0}")]
    Candle(#[from] candle::Error),

    #[error("glowstick error: {0}")]
    Glowstick(#[from] glowstick_candle::Error),
}

enum Model {
    Smol(llama::Llama, llama::Cache),
}

impl Model {
    fn forward(
        &mut self,
        xs: &Tensor<Shape2<B, N>>,
        s: usize,
    ) -> Result<Tensor<Shape2<B, C>>, Error> {
        match self {
            Self::Smol(ref mut model, ref mut cache) => Ok(model.forward(xs, s, cache)?),
        }
    }

    fn eos_tokens(&self, t: &TokenOutputStream<'_>) -> Result<Vec<u32>, Error> {
        match self {
            Self::Smol(m, _) => {
                let eos = t.get_token("</s>").ok_or(Error::MissingEosToken);
                match &m.eos_tokens {
                    Some(LlamaEosToks::Multiple(v)) => {
                        Ok(v.iter().copied().chain(eos.ok()).collect())
                    }
                    Some(LlamaEosToks::Single(n)) => {
                        Ok(std::iter::once(*n).chain(eos.ok()).collect())
                    }
                    None => Ok(std::iter::once(eos?).collect()),
                }
            }
        }
    }
}

struct TextGeneration<'a> {
    model: Model,
    device: Device,
    token_streams: Vec<TokenOutputStream<'a>>,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl<'a> TextGeneration<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: &'a Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_k: Option<usize>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        num_return_sequences: usize,
        device: &Device,
    ) -> Self {
        let temperature = temp.and_then(|v| if v < 1e-7 { None } else { Some(v) });
        let sampling = match temperature {
            None => Sampling::ArgMax,
            Some(temperature) => match (top_k, top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            },
        };
        let logits_processor = LogitsProcessor::from_sampling(seed, sampling);
        Self {
            model,
            token_streams: (0..num_return_sequences)
                .map(|_| TokenOutputStream::new(tokenizer))
                .collect::<Vec<_>>(),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<(), Error> {
        use std::io::Write;

        let n_outputs = self.token_streams.len();
        let eos_tokens = self
            .token_streams
            .first()
            .map(|t| self.model.eos_tokens(t))
            .ok_or(Error::NoTokenStreams)??;
        self.token_streams.iter_mut().for_each(|tokenizer| {
            tokenizer.clear();
        });

        enum TokenList {
            Generating(Vec<u32>),
            Terminated(Vec<u32>),
        }
        impl TokenList {
            fn len(&self) -> usize {
                match self {
                    Self::Generating(v) => v.len(),
                    Self::Terminated(v) => v.len(),
                }
            }

            fn iter(&self) -> impl Iterator<Item = &u32> {
                match self {
                    Self::Generating(v) => v.iter(),
                    Self::Terminated(v) => v.iter(),
                }
            }

            fn ctxt(&self, start_pos: usize) -> &[u32] {
                match self {
                    Self::Generating(v) => &v[start_pos..],
                    Self::Terminated(v) => &v[start_pos..],
                }
            }

            fn push(&mut self, t: u32) {
                match self {
                    Self::Generating(v) => {
                        v.push(t);
                    }
                    Self::Terminated(_v) => {}
                }
            }

            fn terminate(&mut self) {
                let v = match self {
                    Self::Generating(v) | Self::Terminated(v) => std::mem::take(v),
                };
                *self = Self::Terminated(v);
            }

            fn is_terminated(&self) -> bool {
                matches!(self, Self::Terminated { .. })
            }
        }

        let mut token_lists = self
            .token_streams
            .iter()
            .map(|t| {
                Ok::<_, Error>(TokenList::Generating(
                    t.tokenizer()
                        .encode(prompt, true)
                        .map_err(|e| Error::Encode(e.to_string()))?
                        .get_ids()
                        .to_vec(),
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut seq_len = 0;
        let mut finished_sequences = vec![vec![]; token_lists.len()];
        for (i, (v, st)) in token_lists.iter().zip(&mut self.token_streams).enumerate() {
            seq_len = v.len();
            for &t in v.iter() {
                if let Some(t) = st.next_token(t)? {
                    if n_outputs == 1 {
                        print!("{t}")
                    }
                    finished_sequences[i].push(t);
                }
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { seq_len };
            let Some(generating) = token_lists
                .iter()
                .find(|v| matches!(v, TokenList::Generating(_)))
            else {
                break;
            };
            let start_pos = generating.len().saturating_sub(context_size);
            let inputs = cat!(token_lists.iter().map(|v| {
                let start_pos = v.len().saturating_sub(context_size);
                let ctxt = v.ctxt(start_pos);
                let input: Tensor<Shape2<U1, N>> = candle::Tensor::new(ctxt, &self.device)?.unsqueeze(0)?.try_into()?;

                Ok::<_, Error>(input)
            }).collect::<Result<Vec<Tensor<_>>, _>>()?.as_slice(), U0 => B)?;

            let logits = self.model.forward(&inputs, start_pos)?;
            for (i, (v, st)) in token_lists
                .iter_mut()
                .zip(&mut self.token_streams)
                .enumerate()
            {
                let logits = narrow!(&logits, U0: [{ i }, U1])?;
                let logits = squeeze![&logits, U0]?.to_dtype(DType::F32)?;
                let logits = if self.repeat_penalty == 1. {
                    logits
                } else {
                    let start_at = v.len().saturating_sub(self.repeat_last_n);
                    candle_transformers::utils::apply_repeat_penalty(
                        logits.inner(),
                        self.repeat_penalty,
                        v.ctxt(start_at),
                    )?
                    .try_into()?
                };

                let next_token = self.logits_processor.sample(logits.inner())?;
                v.push(next_token);
                generated_tokens += 1;
                if eos_tokens.contains(&next_token) && matches!(v, TokenList::Generating(_)) {
                    v.terminate();
                } else if let Some(t) = st.next_token(next_token)? {
                    if n_outputs == 1 {
                        print!("{t}");
                    } else if generated_tokens % 100 == 0 {
                        println!("Generated {} tokens", generated_tokens);
                    }
                    finished_sequences[i].push(t);
                }
            }
            if token_lists.iter().all(|l| l.is_terminated()) {
                break;
            }
        }
        let dt = start_gen.elapsed();

        for (i, (st, finished)) in self
            .token_streams
            .iter()
            .zip(&mut finished_sequences)
            .enumerate()
        {
            if let Some(rest) = st.decode_rest()? {
                finished.push(rest);
            }
            if n_outputs > 1 {
                println!("[OUTPUT SEQUENCE {}]", i + 1);
                for t in finished.iter() {
                    print!("{t}");
                }
            }
            println!("\n");
            std::io::stdout().flush()?;
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum, PartialEq, Eq)]
enum WhichModel {
    #[value(name = "smol-135m")]
    S135m,
    #[value(name = "smol-360m")]
    S360m,
    #[value(name = "smol-1.7b")]
    S1_7b,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    use_flash_attn: bool,

    #[cfg(any(feature = "cuda", feature = "metal"))]
    #[arg(long, default_value = "GPU go brrr")]
    prompt: String,

    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    #[arg(long, default_value = "CPU go brrr")]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 512)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(short, long, default_value_t = 1)]
    num_return_sequences: usize,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;

    #[cfg(feature = "small")]
    let model = WhichModel::S1_7b;
    #[cfg(feature = "smaller")]
    let model = WhichModel::S360m;
    #[cfg(feature = "smallest")]
    let model = WhichModel::S135m;

    let model_id = match args.model_id {
        Some(model_id) => model_id,
        None => {
            let (version, size) = match model {
                WhichModel::S135m => ("2", "135M"),
                WhichModel::S360m => ("2", "360M"),
                WhichModel::S1_7b => ("2", "1.7B"),
            };
            format!("HuggingFaceTB/SmolLM{version}-{size}")
        }
    };
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => match model {
            WhichModel::S135m | WhichModel::S360m | WhichModel::S1_7b => {
                vec![repo.get("model.safetensors")?]
            }
        },
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config_file = repo.get("config.json")?;
    let device = device(args.cpu)?;
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let model = match model {
        WhichModel::S135m | WhichModel::S360m | WhichModel::S1_7b => {
            let config = serde_json::from_slice::<LlamaConfig>(&std::fs::read(&config_file)?)?
                .into_config(args.use_flash_attn);

            let cache = llama::Cache::new(true, dtype, &config, &device)?;
            Model::Smol(Llama::load(vb, &config)?, cache)
        }
    };

    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        &tokenizer,
        args.seed,
        args.temperature,
        args.top_k,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        args.num_return_sequences,
        &device,
    );
    pipeline.run(&args.prompt, args.sample_len)?;
    Ok(())
}

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
pub struct TokenOutputStream<'a> {
    tokenizer: &'a tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl<'a> TokenOutputStream<'a> {
    pub fn new(tokenizer: &'a tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn into_inner(self) -> &'a tokenizers::Tokenizer {
        self.tokenizer
    }

    fn decode(&self, tokens: &[u32]) -> candle::Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => candle::bail!("cannot decode: {err}"),
        }
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>, Error> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> Result<Option<String>, Error> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_all(&self) -> candle::Result<String> {
        self.decode(&self.tokens)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}

/// Loads the safetensors files for a model from the hub based on a json index file.
pub fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> candle::Result<Vec<std::path::PathBuf>> {
    let json_file = repo.get(json_file).map_err(candle::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).map_err(candle::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => candle::bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => candle::bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| repo.get(v).map_err(candle::Error::wrap))
        .collect::<candle::Result<Vec<_>>>()?;
    Ok(safetensors_files)
}
