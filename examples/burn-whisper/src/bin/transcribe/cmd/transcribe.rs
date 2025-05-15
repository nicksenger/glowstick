use std::{
    io::{BufReader, Read},
    num::NonZeroUsize,
    time::Duration,
};

use burn::{config::Config, prelude::Backend};
use clap::Args;
use futures::TryStreamExt;
use hf_hub::{
    Repo, RepoType,
    api::tokio::{Api, ApiError},
};
use strum::IntoEnumIterator;
use whisp_rs::{
    model::{Whisper, WhisperConfig, timestamps::AlignmentHeads},
    token::{Gpt2Tokenizer, Language},
    transcribe::waveforms_to_text_with_decode,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Failed to load tokenizer: {0}")]
    LoadTokenizer(String),

    #[error("Failed to load config: {0}")]
    LoadConfig(String),

    #[error(
        "Specified batch size of \"{batch_size}\" is not divisible by beam width \"{beam_width}\""
    )]
    BatchSizeBeamWidth {
        batch_size: usize,
        beam_width: usize,
    },

    #[error("Batch size must be greater than zero")]
    BatchSizeZero,

    #[error("Beam width must be greater than zero")]
    BeamWidthZero,

    #[error("Failed to load whisper model file: {0}")]
    LoadModel(String),

    #[error("Unrecognized language \"{0}\"")]
    UnrecognizedLanguage(String),

    #[error("Transcription execution failed: {0}")]
    Transcription(#[from] whisp_rs::transcribe::Error),

    #[error("Join error: {0}")]
    Join(#[from] tokio::task::JoinError),

    #[error("Input file I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Hf hub error: {0}")]
    Hf(#[from] ApiError),
}

#[derive(Debug, Args)]
pub struct Params {
    /// Print verbose output
    #[arg(short = 'v', long)]
    verbose: bool,

    /// Comma delimited input file paths
    #[arg(long, value_delimiter = ',', default_value = "./jfk.wav")]
    inputs: Vec<String>,

    /// Number of mel segments to transcribe in parallel.
    /// This must be non-zero and divisible by the beam_width.
    ///
    /// Defaults to 1
    #[arg(long, default_value = "1")]
    batch_size: usize,

    /// If set to a number greater than 1, enables beam search with
    /// the specified width. This may improve results at the cost of
    /// additional resources.
    ///
    /// Defaults to 1
    #[arg(long, default_value = "1")]
    beam_width: usize,

    // TODO: not really necessary for PoC and results in mostly dynamic dims
    //#[arg(long, default_value = "tiny")]
    //model_name: String,
    /// 2-letter language code to use for transcription
    /// Defaults to `en`
    #[arg(long, default_value = "en")]
    language: String,

    /// Prevent output of word-level timestamps
    #[arg(long, default_value = "false")]
    disable_timestamps: bool,

    /// Prevent use of the self-attention KV cache
    #[arg(long, default_value = "false")]
    disable_kv_cache: bool,

    /// Specify the mel spectrogram buffer size. Defaults to (batch_size / beam_width).
    ///
    /// This is the maximum number of spectrograms what will be processed in parallel while
    /// a transcription is being performed.
    #[arg(long)]
    mel_buffer_size: Option<NonZeroUsize>,

    /// Specify the PCM message buffer size. Defaults to 50
    #[arg(long, default_value = "50")]
    pcm_message_buffer_size: NonZeroUsize,

    /// Specify the # of PCM packets per message. Defaults to 50
    #[arg(long, default_value = "50")]
    pcm_packets_per_message: NonZeroUsize,

    #[arg(long)]
    /// Optionally buffer the file reader before passing to decoder (I think symphonia does some buffering itself)
    file_buffer_size: Option<NonZeroUsize>,
}

pub async fn run<B: Backend>(device: <B as Backend>::Device, params: Params) -> Result<(), Error> {
    let start = std::time::Instant::now();
    match (params.batch_size, params.beam_width) {
        (a, b) if a % b != 0 => {
            return Err(Error::BatchSizeBeamWidth {
                batch_size: params.batch_size,
                beam_width: params.beam_width,
            });
        }

        (0, _) => {
            return Err(Error::BatchSizeZero);
        }

        (_, 0) => {
            return Err(Error::BeamWidthZero);
        }

        _ => {}
    }
    let model = "tiny";

    let waveform_buffer_size = params
        .mel_buffer_size
        .unwrap_or(NonZeroUsize::new(params.batch_size / params.beam_width).unwrap());

    let Some(language) = Language::iter().find(|l| l.as_str() == params.language.as_str()) else {
        return Err(Error::UnrecognizedLanguage(params.language));
    };

    println!("Downloading model assets...");
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        "nicksenger/whisper-mpk".to_string(),
        RepoType::Model,
        "init".to_string(),
    ));
    let vocab = repo.get(&format!("{}.json", model)).await?;
    let cfg = repo.get(&format!("{}.cfg", model)).await?;
    let weights = repo.get(&format!("{}.mpk", model)).await?;
    println!("Done.");

    println!("Loading waveform(s)...");
    let ts_start = std::time::Instant::now();
    let (waveforms, bpe, cfg): (
        Vec<Box<dyn Read + Send + Sync + 'static>>,
        Gpt2Tokenizer,
        WhisperConfig,
    ) = tokio::task::spawn_blocking(move || {
        let waveforms = params
            .inputs
            .into_iter()
            .map(std::fs::File::open)
            .collect::<Result<Vec<_>, std::io::Error>>()?;

        let waveforms = if let Some(capacity) = params.file_buffer_size {
            waveforms
                .into_iter()
                .map(|x| {
                    Box::new(BufReader::with_capacity(capacity.get(), x))
                        as Box<dyn Read + Send + Sync>
                })
                .collect()
        } else {
            waveforms
                .into_iter()
                .map(|x| Box::new(x) as Box<dyn Read + Send + Sync>)
                .collect()
        };

        let bpe = match Gpt2Tokenizer::new(vocab) {
            Ok(bpe) => bpe,
            Err(e) => {
                println!("Failed to load tokenizer: {}", e);
                return Err(Error::LoadTokenizer(e.to_string()));
            }
        };

        let whisper_config = match WhisperConfig::load(cfg) {
            Ok(config) => config,
            Err(e) => {
                println!("Failed to load cfg: {}", e);
                return Err(Error::LoadConfig(e.to_string()));
            }
        };

        Ok((waveforms, bpe, whisper_config))
    })
    .await??;
    println!("Loaded waveforms in {:.2?}", ts_start.elapsed());

    println!("Loading model...");
    let ts_start = std::time::Instant::now();
    let timestamps = !params.disable_timestamps;
    let enable_self_attn_kv_cache = !params.disable_kv_cache;
    let mut whisper = tokio::task::spawn_blocking(move || {
        let whisper: Whisper<B> =
            match crate::load_whisper_model_file(&cfg, weights, &device, timestamps) {
                Ok(whisper_model) => whisper_model,
                Err(e) => {
                    return Err(Error::LoadModel(e.to_string()));
                }
            };

        Ok(whisper)
    })
    .await??;
    println!("Loaded model in {:.2?}", ts_start.elapsed());

    let model_name = model.to_string();
    let timestamps = timestamps.then(|| match model_name.as_str() {
        "tiny" => AlignmentHeads::tiny(),
        "small" => AlignmentHeads::small(),
        "turbo" => AlignmentHeads::large_v3_turbo(),
        "large" => AlignmentHeads::large_v3(),
        _ => AlignmentHeads::default(),
    });
    let mut rx = waveforms_to_text_with_decode(
        &mut whisper,
        &bpe,
        Some(language),
        waveforms,
        params.beam_width,
        params.batch_size,
        enable_self_attn_kv_cache,
        timestamps,
        waveform_buffer_size,
        params.pcm_message_buffer_size,
        params.pcm_packets_per_message,
        true,
    )
    .await?;

    let (mut total, mut audio, mut batch, mut transcript) =
        (0, Duration::ZERO, Duration::ZERO, Duration::ZERO);
    while let Some(output) = rx.try_next().await? {
        total += 1;
        audio += output.avg_audio_processing_time;
        batch += output.batch_processing_time;
        transcript += output.transcription_time;

        for (i, outputs) in output.outputs.iter().enumerate() {
            println!("\nItem {} Outputs:", i);
            for output in outputs {
                println!("{}", output.text);
                for timestamp in &output.timestamps {
                    println!("{}: {}-{}", timestamp.text, timestamp.start, timestamp.end);
                }
            }
        }
    }

    let elapsed = start.elapsed();
    println!("\nTranscription finished in {:.2?}", elapsed);
    println!("Average audio pre-processing time: {:.2?}", audio / total);
    println!("Average batch pre-processing time: {:.2?}", batch / total);
    println!("Average transcription time: {:.2?}", transcript / total);

    Ok(())
}
