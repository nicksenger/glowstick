#![recursion_limit = "256"]

use std::path::PathBuf;

use burn::record::{DefaultRecorder, Recorder, RecorderError};
use burn::{module::Module, tensor::backend::Backend};
use clap::{Parser, Subcommand};
use tracing::Level;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::filter::FromEnvError;
use tracing_subscriber::{EnvFilter, FmtSubscriber};
use whisp_rs::model::*;

mod cmd;

cfg_if::cfg_if! {
    if #[cfg(feature = "wgpu")] {
        use burn::backend::wgpu::WgpuDevice;
    } else if #[cfg(feature = "ndarray")] {
        use burn::backend::ndarray::NdArrayDevice;
    }
}

#[derive(Parser, Debug)]
#[command(name = "burn-whisper", about = "example of integrating glowstick with burn", version, long_about = None)]
struct State {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Transcribes the specified audio file(s)
    Transcribe(cmd::transcribe::Params),
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Failed to run transcribe command: {0}")]
    Transcribe(#[from] cmd::transcribe::Error),

    #[error("Failed to setup tracing env: {0}")]
    Tracing(#[from] FromEnvError),
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::DEBUG.into())
        .from_env()?;
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::DEBUG)
        .with_env_filter(filter)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let state = State::parse();
    if let Err(e) = cmd(state).await {
        tracing::error!("{e}");
        std::process::exit(1);
    }

    Ok(())
}

async fn cmd(state: State) -> Result<(), Error> {
    cfg_if::cfg_if! {
        if #[cfg(feature = "wgpu")] {
            type Backend = burn::backend::Wgpu;
            let device = WgpuDevice::default();
        } else if #[cfg(feature = "ndarray")] {
            type Backend = burn::backend::ndarray::NdArray;
            let device = NdArrayDevice::default();
        }
    }

    if let Err(e) = match state.command {
        Command::Transcribe(params) => cmd::transcribe::run::<Backend>(device, params)
            .await
            .map_err(Error::Transcribe),
    } {
        tracing::error!("{}", e);
    }

    Ok(())
}

fn load_whisper_model_file<B: Backend>(
    config: &WhisperConfig,
    weights: impl Into<PathBuf>,
    device: &B::Device,
    timestamps: bool,
) -> Result<Whisper<B>, RecorderError> {
    DefaultRecorder::new()
        .load(weights.into(), device)
        .map(|record| config.init(device, timestamps).load_record(record))
}
