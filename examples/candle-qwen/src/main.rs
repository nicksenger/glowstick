use anyhow::{Error as E, Result};
use clap::Parser;

use candle::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use glowstick::{
    num::{U0, U1},
    Shape3,
};
use glowstick_candle::squeeze;
use glowstick_candle::tensor::Tensor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

mod qwen3;
mod shape;

use qwen3::{Config as Config3, ModelForCausalLM as Model3};
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

    #[error("{0}")]
    Candle(#[from] candle::Error),

    #[error("glowstick error: {0}")]
    Glowstick(#[from] glowstick_candle::Error),
}

enum Model {
    Base3(Model3),
}

impl Model {
    fn forward(
        &mut self,
        xs: &candle::Tensor,
        s: usize,
    ) -> Result<Tensor<Shape3<U1, U1, U151936>>, Error> {
        match self {
            Self::Base3(ref mut m) => m.forward(xs, s),
        }
    }
}

struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|endoftext|> token"),
        };
        let eos_token2 = match self.tokenizer.get_token("<|im_end|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|im_end|> token"),
        };
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = candle::Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = squeeze![&logits, U1, U0]?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    logits.inner(),
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
                .try_into()?
            };

            let next_token = self.logits_processor.sample(logits.inner())?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token || next_token == eos_token2 {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
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
    #[value(name = "3-0.6b")]
    W3_0_6b,
    #[value(name = "3-1.7b")]
    W3_1_7b,
    #[value(name = "3-4b")]
    W3_4b,
    #[value(name = "3-8b")]
    W3_8b,
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

    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10000)]
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

    #[arg(long, default_value = "0.5b")]
    model: WhichModel,
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
    let model_id = match args.model_id {
        Some(model_id) => model_id,
        None => {
            let (version, size) = match args.model {
                WhichModel::W3_0_6b => ("3", "0.6B"),
                WhichModel::W3_1_7b => ("3", "1.7B"),
                WhichModel::W3_4b => ("3", "4B"),
                WhichModel::W3_8b => ("3", "8B"),
            };
            format!("Qwen/Qwen{version}-{size}")
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
        None => match args.model {
            WhichModel::W3_0_6b => {
                vec![repo.get("model.safetensors")?]
            }
            WhichModel::W3_1_7b | WhichModel::W3_4b | WhichModel::W3_8b => {
                hub_load_safetensors(&repo, "model.safetensors.index.json")?
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
    let model = match args.model {
        WhichModel::W3_0_6b | WhichModel::W3_1_7b | WhichModel::W3_4b | WhichModel::W3_8b => {
            let config: Config3 = serde_json::from_slice(&std::fs::read(config_file)?)?;
            Model::Base3(Model3::new(&config, vb)?)
        }
    };

    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
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
pub struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }

    fn decode(&self, tokens: &[u32]) -> candle::Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => candle::bail!("cannot decode: {err}"),
        }
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
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

    pub fn decode_rest(&self) -> Result<Option<String>> {
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
        &self.tokenizer
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
