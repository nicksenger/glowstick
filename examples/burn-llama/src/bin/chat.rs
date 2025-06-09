#![recursion_limit = "256"]

use std::time::Instant;

use burn::tensor::{backend::Backend, Device};
use burn_llama::{
    llama::{Llama, LlamaConfig},
    sampling::{Sampler, TopP},
    tokenizer::Tokenizer,
};
use clap::Parser;

const DEFAULT_PROMPT: &str = "GPU go brrr";

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Config {
    /// Top-p probability threshold.
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,

    /// Temperature value for controlling randomness in sampling.
    #[arg(long, default_value_t = 0.6)]
    temperature: f64,

    /// Maximum sequence length for input text.
    #[arg(long, default_value_t = 128)]
    max_seq_len: usize,

    /// The number of new tokens to generate (i.e., the number of generation steps to take).
    #[arg(long, short = 'n', default_value_t = 65)]
    sample_len: usize,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// The input prompt.
    #[arg(short, long, default_value_t = String::from(DEFAULT_PROMPT))]
    prompt: String,
}

pub fn generate<B: Backend, T: Tokenizer>(
    llama: &mut Llama<B, T>,
    prompt: &str,
    sample_len: usize,
    temperature: f64,
    sampler: &mut Sampler,
) {
    let now = Instant::now();
    let generated = llama
        .generate(prompt, sample_len, temperature, sampler)
        .unwrap();
    let elapsed = now.elapsed().as_secs();

    println!("> {}\n", generated.text);
    println!(
        "{} tokens generated ({:.4} tokens/s)\n",
        generated.tokens,
        generated.tokens as f64 / generated.time
    );

    println!(
        "Generation completed in {}m{}s",
        (elapsed / 60),
        elapsed % 60
    );
}

pub fn chat<B: Backend>(args: Config, device: Device<B>) {
    let prompt = args.prompt;

    // Sampling strategy
    let mut sampler = if args.temperature > 0.0 {
        Sampler::TopP(TopP::new(args.top_p, args.seed))
    } else {
        Sampler::Argmax
    };

    let mut llama = LlamaConfig::llama3_2_pretrained::<B>(args.max_seq_len, &device).unwrap();
    println!("Processing prompt: {}", prompt);

    generate(
        &mut llama,
        &prompt,
        args.sample_len,
        args.temperature,
        &mut sampler,
    );
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use super::*;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    pub fn run(args: Config) {
        let device = WgpuDevice::default();

        chat::<Wgpu>(args, device);
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use burn::{
        backend::{cuda::CudaDevice, Cuda},
        tensor::f16,
    };

    pub fn run(args: Config) {
        let device = CudaDevice::default();

        chat::<Cuda<f16, i32>>(args, device);
    }
}

pub fn main() {
    // Parse arguments
    let args = Config::parse();

    #[cfg(feature = "wgpu")]
    wgpu::run(args);
    #[cfg(feature = "cuda")]
    cuda::run(args);

    #[cfg(all(not(feature = "wgpu"), not(feature = "cuda")))]
    println!("No backend enabled.");
}
