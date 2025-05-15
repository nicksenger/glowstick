use std::thread::sleep;
use std::time::Duration;

use futures::channel::mpsc;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::conv::FromSample;
use symphonia::core::io::MediaSource;
use tracing::{debug, trace};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Symphonia error: {0}")]
    Probe(#[from] symphonia::core::errors::Error),

    #[error("Join error: {0}")]
    Join(#[from] tokio::task::JoinError),

    #[error("Error sending PCM data: {0}")]
    SendError(#[from] mpsc::TrySendError<Vec<f32>>),
}

fn conv<T>(samples: &mut Vec<f32>, data: std::borrow::Cow<symphonia::core::audio::AudioBuffer<T>>)
where
    T: symphonia::core::sample::Sample,
    f32: symphonia::core::conv::FromSample<T>,
{
    samples.extend(data.chan(0).iter().map(|v| f32::from_sample(*v)))
}

/// Decodes PCM asynchronously, buffer size is in # of packets
pub fn pcm_decode(
    source: impl MediaSource + 'static,
    message_buffer_size: usize,
    packets_per_message: usize,
) -> Result<mpsc::Receiver<Vec<f32>>, Error> {
    let (mut tx, rx) = mpsc::channel(message_buffer_size);

    let mss = symphonia::core::io::MediaSourceStream::new(Box::new(source), Default::default());
    let hint = symphonia::core::probe::Hint::new();
    let meta_opts: symphonia::core::meta::MetadataOptions = Default::default();
    let fmt_opts: symphonia::core::formats::FormatOptions = Default::default();

    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .expect("no supported audio tracks");

    let dec_opts: DecoderOptions = Default::default();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .expect("unsupported codec");
    let track_id = track.id;
    let _sample_rate = track.codec_params.sample_rate.unwrap_or(0);

    tokio::task::spawn(async move {
        tokio::task::spawn_blocking(move || {
            let mut packets = 0;
            let mut pcm_data = vec![];

            while let Ok(packet) = format.next_packet() {
                while !format.metadata().is_latest() {
                    format.metadata().pop();
                }

                if packet.track_id() != track_id {
                    continue;
                }

                match decoder.decode(&packet)? {
                    AudioBufferRef::F32(buf) => pcm_data.extend(buf.chan(0)),
                    AudioBufferRef::U8(data) => conv(&mut pcm_data, data),
                    AudioBufferRef::U16(data) => conv(&mut pcm_data, data),
                    AudioBufferRef::U24(data) => conv(&mut pcm_data, data),
                    AudioBufferRef::U32(data) => conv(&mut pcm_data, data),
                    AudioBufferRef::S8(data) => conv(&mut pcm_data, data),
                    AudioBufferRef::S16(data) => conv(&mut pcm_data, data),
                    AudioBufferRef::S24(data) => conv(&mut pcm_data, data),
                    AudioBufferRef::S32(data) => conv(&mut pcm_data, data),
                    AudioBufferRef::F64(data) => conv(&mut pcm_data, data),
                }
                packets += 1;

                if packets == packets_per_message {
                    packets = 0;
                    let mut pcm_data = std::mem::take(&mut pcm_data);
                    while let Err(e) = tx.try_send(pcm_data) {
                        match e {
                            e if e.is_full() => {
                                pcm_data = e.into_inner();
                                trace!("Packet buffer is full, waiting 100ms.");
                                sleep(Duration::from_millis(100));
                            }
                            e => {
                                return Err(Error::SendError(e));
                            }
                        }
                    }
                }
            }
            while let Err(e) = tx.try_send(std::mem::take(&mut pcm_data)) {
                match e {
                    e if e.is_full() => {
                        pcm_data = e.into_inner();
                        trace!("Packet buffer is full, waiting 100ms.");
                        sleep(Duration::from_millis(100));
                    }
                    e => {
                        return Err(Error::SendError(e));
                    }
                }
            }

            Ok::<_, Error>(())
        })
        .await??;

        Ok::<_, Error>(())
    });

    debug!("PCM decode finished.");
    Ok(rx)
}
