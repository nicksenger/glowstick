#![recursion_limit = "256"]

use std::path::PathBuf;

pub use burn::prelude::Config;
pub use burn::tensor::backend::Backend as BurnBackend;
use burn::{
    module::Module,
    record::{DefaultRecorder, Recorder, RecorderError},
};
use model::{Whisper, WhisperConfig};

pub mod audio;
pub mod beam;
pub mod helper;
pub mod model;
pub mod pcm_decode;
pub mod shape;
pub mod token;
pub mod transcribe;

pub const MEL_LEN: usize = 240_000;
pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const CHUNK_LENGTH: usize = 30;
pub const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE;
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH;
pub const NO_SPEECH_THRESHOLD: f64 = 0.6;
pub const LOGPROB_THRESHOLD: f64 = -1.0;
pub const TEMPERATURES: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
pub const COMPRESSION_RATIO_THRESHOLD: f64 = 2.4;
pub const SOT_TOKEN: &str = "<|startoftranscript|>";
pub const TRANSCRIBE_TOKEN: &str = "<|transcribe|>";
pub const TRANSLATE_TOKEN: &str = "<|translate|>";
pub const NO_TIMESTAMPS_TOKEN: &str = "<|notimestamps|>";
pub const EOT_TOKEN: &str = "<|endoftext|>";
pub const NO_SPEECH_TOKENS: [&str; 2] = ["<|nocaptions|>", "<|nospeech|>"];

cfg_if::cfg_if! {
    if #[cfg(feature = "wgpu")] {
        use burn::backend::wgpu::WgpuDevice;
        type Backend = burn::backend::Wgpu;
    } else if #[cfg(feature = "ndarray")] {
        use burn::backend::ndarray::NdArrayDevice;
        type Backend = burn::backend::ndarray::NdArray;
    }
}

pub fn load_whisper_model_file(
    config: &WhisperConfig,
    path: impl Into<PathBuf>,
    timestamps: bool,
) -> Result<Whisper<Backend>, RecorderError> {
    cfg_if::cfg_if! {
        if #[cfg(feature = "wgpu")] {
            let device = WgpuDevice::default();
        } else if #[cfg(feature = "ndarray")] {
            let device = NdArrayDevice::default();
        }
    }

    DefaultRecorder::new()
        .load(path.into(), &device)
        .map(|record| config.init(&device, timestamps).load_record(record))
}
