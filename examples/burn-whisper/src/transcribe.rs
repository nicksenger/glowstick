use std::iter;
use std::num::NonZeroUsize;
use std::{io::Read, time::Duration};

use burn::{
    module::Module,
    prelude::Int,
    tensor::{Tensor as BTensor, TensorData, backend::Backend},
};
use futures::stream::BoxStream;
use futures::{FutureExt, Stream, channel::mpsc};
use futures::{SinkExt, StreamExt};
use glowstick::{
    Shape1, Shape2, Shape3,
    num::{U0, U1, U2, U80},
};
use tracing::{debug, trace};

#[allow(unused)]
use glowstick::debug_tensor;

use crate::audio::pcm_to_mel;
use crate::beam;
use crate::model;
use crate::model::timestamps::{AlignmentHeads, PostProcessor, Raw};
use crate::model::*;
use crate::pcm_decode;
use crate::pcm_decode::pcm_decode;
use crate::shape::*;
use crate::tensor::Tensor;
use crate::token::*;
use crate::{HOP_LENGTH, MEL_LEN};
use crate::{log_softmax, reshape, squeeze, unsqueeze};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Batch size must be divisible by the beam width")]
    BatchSizeBeamWidth,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Send error: {0}")]
    Send(#[from] mpsc::SendError),

    #[error("Join error: {0}")]
    Join(#[from] tokio::task::JoinError),

    #[error("Tokenization error: {0}")]
    Tokenization(#[from] tokenizers::Error),

    #[error("Error determining timestamps: {0}")]
    Timestamps(#[from] timestamps::Error),

    #[error("Model processing error: {0}")]
    Model(#[from] model::Error),

    #[error("PCM decode error: {0}")]
    PcmDecode(#[from] pcm_decode::Error),

    #[error("Mel conversion error: {0}")]
    MelCast(usize),

    #[error("Missing one or more output attentions")]
    OutputAttentions,

    #[error("Data error: {0}")]
    Data(String),

    #[error("Tensor error: {0}")]
    Tensor(#[from] crate::tensor::Error),
}

#[allow(clippy::too_many_arguments)]
pub async fn waveforms_to_text_with_decode<'a, B: Backend, T: Read + Send + Sync + 'static>(
    whisper: &'a mut Whisper<B>,
    bpe: &'a Gpt2Tokenizer,
    lang: Option<Language>,
    waveforms: impl IntoIterator<Item = T>,
    beam_width: usize,
    batch_size: usize,
    enable_self_attn_kv_cache: bool,
    alignment_heads: Option<AlignmentHeads>,
    waveform_buffer_size: NonZeroUsize,
    pcm_message_buffer_size: NonZeroUsize,
    pcm_packets_per_message: NonZeroUsize,
    verbose: bool,
) -> std::result::Result<BoxStream<'a, std::result::Result<BatchResult, Error>>, Error> {
    let waveforms = waveforms
        .into_iter()
        .map(symphonia::core::io::ReadOnlySource::new)
        .map(|source| {
            pcm_decode(
                source,
                pcm_message_buffer_size.into(),
                pcm_packets_per_message.into(),
            )
            .map_err(Error::PcmDecode)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;

    waveforms_to_text(
        whisper,
        bpe,
        lang,
        waveforms,
        beam_width,
        batch_size,
        enable_self_attn_kv_cache,
        alignment_heads.clone(),
        waveform_buffer_size,
        verbose,
        MEL_LEN,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn waveforms_to_text<
    'a,
    B: Backend,
    T: Stream<Item = Vec<f32>> + Send + Sync + Unpin + 'static,
>(
    whisper: &'a mut Whisper<B>,
    bpe: &'a Gpt2Tokenizer,
    lang: Option<Language>,
    waveforms: impl IntoIterator<Item = T>,
    beam_width: usize,
    batch_size: usize,
    enable_self_attn_kv_cache: bool,
    alignment_heads: Option<AlignmentHeads>,
    waveform_buffer_size: NonZeroUsize,
    verbose: bool,
    mel_len: usize,
) -> std::result::Result<BoxStream<'a, std::result::Result<BatchResult, Error>>, Error> {
    if batch_size % beam_width != 0 {
        return Err(Error::BatchSizeBeamWidth);
    }

    let minibatch_size = batch_size / beam_width;
    let device = whisper.devices()[0].clone();
    let num_mel_bins = whisper.encoder_mel_size();
    let mel_bytes = match num_mel_bins {
        80 => include_bytes!("melfilters.bytes").as_slice(),
        128 => include_bytes!("melfilters128.bytes").as_slice(),
        _ => panic!("unexpected number of mel bins!"),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);
    let buffer = waveform_buffer_size.get();
    let (wav_tx, mut wav_rx) = futures::channel::mpsc::channel(buffer);
    let waveforms = waveforms
        .into_iter()
        .map(move |wf| (wf, wav_tx.clone()))
        .collect::<Vec<_>>();

    let num_waveforms = waveforms.len();
    let sample_fut = futures::future::join_all(waveforms.into_iter().enumerate().map(
        |(i, (mut st, mut tx))| async move {
            let mut timestamp = tokio::time::Instant::now();
            let mut samples = Vec::with_capacity(mel_len);
            let mut leftovers = vec![];

            loop {
                let mut should_exit = false;
                samples.extend(std::mem::take(&mut leftovers));
                loop {
                    match st.next().await {
                        Some(mut floats) => {
                            trace!("Received {} samples for input {}", floats.len(), i);
                            match mel_len.saturating_sub(samples.len()) {
                                n if n < floats.len() => {
                                    leftovers = floats.split_off(n);
                                    debug!(
                                        "Split off {} samples to avoid exceeding maximum mel size",
                                        leftovers.len()
                                    );
                                    samples.extend(floats);
                                    break;
                                }
                                _ => {
                                    samples.extend(floats);
                                }
                            }
                        }
                        None => {
                            debug!("Finished receiving batch samples for input {}", i);
                            should_exit = true;
                            break;
                        }
                    }
                }

                let n_frames = (samples.len() as f64 / crate::HOP_LENGTH as f64).round() as usize;
                let batch_item = (
                    i,
                    (
                        n_frames,
                        std::mem::replace(&mut samples, Vec::with_capacity(mel_len)),
                    ),
                );

                tx.send((timestamp.elapsed(), batch_item)).await?;
                timestamp = tokio::time::Instant::now();
                samples.clear();
                debug!("Sent batch item for input {}", i);
                if should_exit {
                    break;
                }
            }

            Ok::<_, Error>(())
        },
    ))
    .map(|v| v.into_iter().collect::<std::result::Result<(), _>>());

    let (mut batch_tx, mut batch_rx) = mpsc::channel(1);
    let batch_fut = async move {
        let mut timestamp = tokio::time::Instant::now();
        let (mut audios, mut total_audio_processing_time) = (0, Duration::ZERO);

        let mut v = Vec::with_capacity(minibatch_size);
        while let Some((audio_processing_time, (file_idx, (n_frames, mel_floats)))) =
            wav_rx.next().await
        {
            audios += 1;
            total_audio_processing_time += audio_processing_time;

            let (mel, mel_len) = tokio::task::block_in_place(|| {
                let mel_data = pcm_to_mel::<f32>(num_mel_bins, &mel_floats, &mel_filters);
                let mel_len = mel_data.len();
                let mel = Tensor::<BTensor<B, 1>, Shape1<U240000>>::from_floats(
                    mel_data.as_slice(),
                    &device,
                )?;
                let mel = reshape!(mel, Shape3<U1, U80, U3000>)?;

                Ok::<_, Error>((mel, mel_len))
            })?;
            debug!(
                "Processed mel spectrogram of length {} for input {}",
                mel_len, file_idx
            );

            v.push((file_idx, (n_frames, mel)));
            if v.len() == minibatch_size {
                batch_tx
                    .send((
                        timestamp.elapsed(),
                        total_audio_processing_time / audios,
                        std::mem::replace(&mut v, Vec::with_capacity(minibatch_size)),
                    ))
                    .await?;
                debug!("Sent batch data.");
                timestamp = tokio::time::Instant::now();
                audios = 0;
                total_audio_processing_time = Duration::ZERO;
            }
        }

        if !v.is_empty() {
            batch_tx
                .send((timestamp.elapsed(), total_audio_processing_time / audios, v))
                .await?;
        }
        debug!("Finished sending batch data.");

        Ok::<_, Error>(())
    };

    let (out_tx, out_rx) = mpsc::channel(1_000);
    let transcribe_fut = async move {
        let mut time_offsets = vec![0.; num_waveforms];
        let mut batch_num = 0;
        let mut total_batch_processing_time = Duration::ZERO;
        let mut sum_avg_audio_processing_times = Duration::ZERO;
        let mut sum_transcription_times = Duration::ZERO;

        while let Some((batch_processing_time, avg_audio_processing_time, batch)) =
            batch_rx.next().await
        {
            total_batch_processing_time += batch_processing_time;
            sum_avg_audio_processing_times += avg_audio_processing_time;
            let transcription_start = tokio::time::Instant::now();

            if batch.is_empty() {
                debug!("Received empty batch, ignoring..");
                continue;
            }

            batch_num += 1;
            debug!(
                "Received all batch items for transcription batch {}, dims: {}",
                batch_num,
                batch
                    .iter()
                    .map(|t| format!("{:?}", t.1.1.inner().dims()))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            let (item_indices, mels): (Vec<_>, Vec<_>) = batch.into_iter().unzip();
            let (n_frames, mels): (Vec<_>, Vec<_>) = mels.into_iter().unzip();
            let total_frames = n_frames.iter().copied().max().unwrap_or_default();
            let outputs = tokio::task::block_in_place(|| {
                let outputs = mels_to_text(
                    whisper,
                    bpe,
                    lang,
                    mels,
                    total_frames,
                    beam_width,
                    enable_self_attn_kv_cache,
                    alignment_heads.clone(),
                    verbose,
                )?;

                Ok::<_, Error>(outputs)
            })?;

            let mut outs = vec![vec![]; num_waveforms];
            for (item, outputs) in outputs.into_iter().enumerate() {
                outs[item_indices[item]].extend(outputs.into_iter().map(|output| {
                    Output {
                        tokens: output.tokens,
                        text: output.text,
                        timestamps: output
                            .timestamps
                            .into_iter()
                            .map(|word| crate::transcribe::timestamps::Word {
                                start: word.start + time_offsets[item_indices[item]],
                                end: word.end + time_offsets[item_indices[item]],
                                ..word
                            })
                            .collect(),
                    }
                }));

                time_offsets[item_indices[item]] += n_frames[item] as f32 / 100.;
            }

            out_tx
                .clone()
                .send(BatchResult {
                    outputs: outs,
                    batch_processing_time,
                    avg_audio_processing_time,
                    transcription_time: transcription_start.elapsed(),
                })
                .await?;

            sum_transcription_times += transcription_start.elapsed();
        }

        Ok::<(), Error>(())
    };

    tokio::task::spawn(sample_fut);
    tokio::task::spawn(batch_fut);
    let driver = transcribe_fut
        .map(|res| match res {
            Err(e) => Some(Err(e)),
            _ => None,
        })
        .into_stream();
    let receiver = out_rx.map(|msg| Some(Ok(msg)));
    let st = futures::stream::select(driver.boxed(), receiver.boxed())
        .filter_map(futures::future::ready);

    Ok(st.boxed())
}

#[derive(Debug, Clone)]
struct BeamSearchToken {
    token: usize,
    #[allow(unused)]
    log_prob: f64,
    is_end: bool,
}

#[derive(Debug, Clone)]
pub struct Output {
    pub tokens: Vec<usize>,
    pub text: String,
    pub timestamps: Vec<model::timestamps::Word>,
}

#[derive(Debug, Clone)]
pub struct BatchResult {
    pub outputs: Vec<Vec<Output>>,
    pub avg_audio_processing_time: Duration,
    pub batch_processing_time: Duration,
    pub transcription_time: Duration,
}

#[derive(Debug, Clone)]
pub struct TranscriptionOutput {
    pub outputs: Vec<Vec<Output>>,
    pub avg_batch_processing_time: Duration,
    pub avg_audio_processing_time: Duration,
    pub avg_transcription_time: Duration,
}

#[allow(clippy::too_many_arguments)]
fn mels_to_text<B: Backend>(
    whisper: &mut Whisper<B>,
    bpe: &Gpt2Tokenizer,
    lang: Option<Language>,
    mels: Vec<Tensor<BTensor<B, 3>, Shape3<U1, U80, U3000>>>,
    total_frames: usize,
    beam_width: usize,
    enable_self_attn_kv_cache: bool,
    alignment_heads: Option<AlignmentHeads>,
    verbose: bool,
) -> std::result::Result<Vec<Vec<Output>>, Error> {
    let timestamps = alignment_heads.is_some();
    let enable_self_attn_kv_cache = enable_self_attn_kv_cache || timestamps;

    let batch_size = mels.len();
    let item_frames = mels
        .iter()
        .map(|mel| mel.inner().dims()[2])
        .collect::<Vec<_>>();
    let end_timestamp_tokens = item_frames
        .iter()
        .map(|n| {
            bpe.special_token(SpecialToken::Timestamp((n / HOP_LENGTH) as f64))
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();

    let mels: Tensor<BTensor<B, 3>, Shape3<BB, U80, U3000>> = BTensor::cat(
        mels.into_iter()
            .map(|mel| mel.into_inner().repeat(&[beam_width, 1, 1]))
            .collect(),
        0,
    )
    .try_into()?;
    let [_, _, content_frames] = mels.inner().dims();
    let device = mels.inner().device();
    let _n_ctx_max_encoder = whisper.encoder_ctx_size();
    let _n_ctx_max_decoder = whisper.decoder_ctx_size();
    let mut seek = 0;
    let mut time_offset = 0.;
    let mut outputs: Vec<Vec<Output>> = vec![vec![]; batch_size];

    while seek < content_frames {
        whisper.reset_kv_cache();
        let mut tc = vec![
            model::cache::TensorCache::<B, 4>::new(2, usize::MAX);
            whisper.decoder.blocks.len()
        ];

        let segment_size = usize::min(content_frames - seek, crate::N_FRAMES);
        let n_frames = segment_size.min(
            total_frames
                .checked_sub(seek)
                .or_else(|| {
                    seek.checked_sub(crate::N_FRAMES)
                        .and_then(|seek| total_frames.checked_sub(seek))
                })
                .unwrap_or_default(),
        );
        let mel_segment = mels.clone();
        seek += segment_size;
        let _segment_duration =
            (segment_size * crate::HOP_LENGTH) as f64 / crate::SAMPLE_RATE as f64;

        let start_token = bpe.special_token(SpecialToken::StartofTranscript).unwrap();
        let transcription_token = bpe.special_token(SpecialToken::Transcribe).unwrap();
        let lang_token = lang.map(|lang| bpe.special_token(SpecialToken::Language(lang)).unwrap());
        let end_token = bpe.special_token(SpecialToken::EndofText).unwrap();
        let notimestamp = bpe.special_token(SpecialToken::NoTimeStamps).unwrap();

        let encoder_output = whisper.forward_encoder(mel_segment, true)?;
        let mut tokens = vec![start_token];
        if let Some(lang_token) = lang_token {
            tokens.push(lang_token);
        }
        tokens.push(transcription_token);
        let n_start_tokens = tokens.len();

        let neg_infty = -f32::INFINITY;
        let max_depth = 100;

        type BeamNode = beam::BeamNode<BeamSearchToken>;
        let initial_tokens = std::iter::repeat_with(|| {
            std::iter::repeat_with(|| BeamNode {
                seq: tokens
                    .iter()
                    .map(|&token| BeamSearchToken {
                        token,
                        log_prob: 0.0,
                        is_end: false,
                    })
                    .collect(),
                log_prob: 0.0,
                is_finished: false,
            })
            .take(beam_width)
            .collect::<Vec<_>>()
        })
        .take(batch_size)
        .collect::<Vec<_>>();

        let beamsearch_is_finished = |node: &BeamNode| {
            node.is_finished
                || if let Some(btok) = node.seq.last() {
                    btok.token == end_token || btok.is_end
                } else {
                    false
                }
        };

        let vocab_size = bpe.vocab_size();
        let special_tokens_maskout: Vec<f32> = (0..vocab_size)
            .map(|token| {
                if bpe.is_special(token) {
                    neg_infty
                } else {
                    0.0
                }
            })
            .collect();
        let special_tokens_maskout: Tensor<BTensor<B, 1>, Shape1<U51865>> =
            Tensor::from_floats(&special_tokens_maskout[..], &device)?;
        let special_tokens_maskout = unsqueeze!(special_tokens_maskout, U0, U0);

        let beamsearch_next =
            |pass: usize,
             batch_beams: &[Vec<BeamNode>],
             tensor_cache: &mut [model::cache::TensorCache<B, 4>]| {
                // convert tokens into tensor
                let max_seq_len = batch_beams
                    .iter()
                    .flat_map(|beams| beams.iter().map(|beam| beam.seq.len()).max())
                    .max()
                    .unwrap_or(0);
                let flattened_tokens: Vec<_> = batch_beams
                    .iter()
                    .flat_map(|beams| {
                        beams.iter().flat_map(|beam| {
                            let additional_tokens = max_seq_len - beam.seq.len();
                            beam.seq
                                .iter()
                                .map(|btok| btok.token)
                                .chain(iter::once(0).cycle().take(additional_tokens))
                        })
                    })
                    .map(|n| n as u32)
                    .collect();

                let token_tensor = Tensor::<BTensor<B, 2, Int>, Shape2<BB, L>>::from_ints(
                    TensorData::new(flattened_tokens, [beam_width * batch_size, max_seq_len]),
                    &device,
                )?;
                let logits = whisper.forward_decoder(
                    token_tensor,
                    encoder_output.clone(),
                    enable_self_attn_kv_cache,
                    Some(tensor_cache),
                    pass == 0,
                )?;
                let logits = if max_seq_len > 4 {
                    logits
                } else {
                    (logits.into_inner()
                        + special_tokens_maskout.clone().into_inner().repeat(&[
                            beam_width * batch_size,
                            1,
                            1,
                        ]))
                    .try_into()?
                };

                let log_probs = log_softmax!(logits, U2);
                let [_n_batch, seq_len, _n_dict] = log_probs.inner().dims();
                let all_log_probs = (0..(beam_width * batch_size))
                    .map(|i| {
                        let lp = log_probs
                            .clone()
                            .try_slice::<Shape3<U1, U1, U51865>, _, 2>([
                                i..i + 1,
                                (seq_len - 1)..seq_len,
                            ])?;

                        let lp = squeeze![lp, U0, U0];
                        lp.into_inner()
                            .to_data()
                            .to_vec::<f32>()
                            .map(|x| (i, x))
                            .map_err(|e| Error::Data(format!("{:?}", e)))
                    })
                    .filter_map(|res| res.ok());

                let mut continuations = vec![vec![]; batch_size];
                for (i, log_probs) in all_log_probs {
                    let bi = i / beam_width;
                    let batch = &batch_beams[bi];
                    let beam = &batch[i % beam_width];
                    let token_probs = log_probs
                        .into_iter()
                        .enumerate()
                        .map(|(token_id, log_prob)| {
                            (
                                BeamSearchToken {
                                    token: if pass == 0 && enable_self_attn_kv_cache {
                                        notimestamp
                                    } else if beam.is_finished {
                                        end_token
                                    } else if token_id == end_token && !beam.is_finished {
                                        end_timestamp_tokens[bi]
                                    } else {
                                        token_id
                                    },
                                    log_prob: log_prob as f64,
                                    is_end: token_id == end_token || beam.is_finished,
                                },
                                beam.log_prob + log_prob as f64,
                            )
                        })
                        .collect::<Vec<_>>();

                    continuations[bi].push(token_probs);
                }

                Ok(continuations)
            };

        let beam_start = std::time::Instant::now();
        let (batch_indices, batch_tokens): (Vec<_>, Vec<_>) = beam::beam_search(
            initial_tokens,
            beamsearch_next,
            beamsearch_is_finished,
            beam_width,
            max_depth,
            &mut tc,
        )?
        .into_iter()
        .map(|(i, btok)| {
            (
                i,
                btok.into_iter()
                    .take_while(|t| !t.is_end)
                    .map(|t| t.token)
                    .chain(std::iter::once(end_token))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
        if verbose {
            println!("Predicted token sequence in {:.2?}", beam_start.elapsed());
        }

        let attn_outs = tc
            .into_iter()
            .map(|mut c| c.take_all_data())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| Error::OutputAttentions)?;
        let raw_timestamps = if timestamps {
            let ts_start = std::time::Instant::now();
            let raw = whisper
                .dtw_timestamps(
                    attn_outs
                        .into_iter()
                        .filter_map(|x| Tensor::try_from(x).ok())
                        .collect::<Vec<_>>(),
                    alignment_heads.clone().unwrap_or_default(),
                    NonZeroUsize::new(6).unwrap(),
                    n_frames,
                    n_start_tokens,
                    batch_indices,
                )?
                .into_iter()
                .map(|Raw(v)| Raw(v.into_iter().map(|n| n + time_offset as f32).collect()))
                .collect::<Vec<_>>();
            if verbose {
                println!("Generated timestamps in {:.2?}", ts_start.elapsed());
            }
            raw
        } else {
            vec![Raw(vec![]); batch_size / beam_width]
        };

        for (i, (tokens, ts)) in batch_tokens.into_iter().zip(raw_timestamps).enumerate() {
            let text = bpe.decode(&tokens, true)?;
            let timestamps = if timestamps {
                bpe.label(&ts, &tokens)?
            } else {
                vec![]
            };
            outputs[i].push(Output {
                tokens,
                text,
                timestamps,
            })
        }

        time_offset = (seek * crate::HOP_LENGTH) as f64 / crate::SAMPLE_RATE as f64;
    }

    Ok(outputs)
}

impl crate::model::timestamps::PostProcessor for Gpt2Tokenizer {
    type Error = crate::model::timestamps::Error;
    fn decode(
        &self,
        tokens: &[usize],
    ) -> std::result::Result<Vec<crate::model::timestamps::Segment>, crate::model::timestamps::Error>
    {
        let full_decode = Gpt2Tokenizer::decode(self, tokens, true)
            .map_err(|_| crate::model::timestamps::Error::Decode)?;
        let decoded_tokens = tokens
            .iter()
            .filter(|&&n| n < 50_000)
            .copied()
            .map(|n| Gpt2Tokenizer::decode(self, &[n], true).map_err(|e| e.to_string()))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| crate::model::timestamps::Error::Decode)?;

        crate::model::timestamps::unicode_segments(full_decode, decoded_tokens)
    }
}
