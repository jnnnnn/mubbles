//! File transcription module
//!
//! Handles transcribing audio files using the Whisper model.
//! Isolated from the microphone pipeline with its own channel and progress reporting.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::Sender,
    Arc,
};

use candle_core::Tensor;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::conv::FromSample;

use crate::{
    audio::{convert_sample_rate, TARGET_SAMPLE_RATE},
    mel,
    whisper::{load_whisper_model, WhichModel, WhisperContext},
};

/// Maximum chunk length in samples (30s at Whisper's sample rate).
const MAX_CHUNK_SAMPLES: usize = TARGET_SAMPLE_RATE * 30;

/// Window in samples to search for a quiet point around each 30s boundary (±2s).
const SEARCH_WINDOW: usize = TARGET_SAMPLE_RATE * 2;

/// Sliding RMS window in samples (50ms) for measuring local energy.
const RMS_WINDOW: usize = TARGET_SAMPLE_RATE / 20;

/// Split PCM audio into chunks of at most 30s, preferring to break at the
/// quietest point within ±2s of each nominal 30s boundary.
fn split_at_silence(pcm: &[f32]) -> Vec<&[f32]> {
    if pcm.len() <= MAX_CHUNK_SAMPLES {
        return vec![pcm];
    }

    let mut chunks = Vec::new();
    let mut offset = 0;

    while offset < pcm.len() {
        let remaining = pcm.len() - offset;
        if remaining <= MAX_CHUNK_SAMPLES {
            chunks.push(&pcm[offset..]);
            break;
        }

        // Nominal split at 30s
        let nominal = offset + MAX_CHUNK_SAMPLES;
        let search_start = nominal.saturating_sub(SEARCH_WINDOW).max(offset);
        let search_end = (nominal + SEARCH_WINDOW).min(pcm.len());

        // Find the quietest RMS_WINDOW-sized window in the search range
        let best = (search_start..search_end.saturating_sub(RMS_WINDOW))
            .map(|i| {
                let window = &pcm[i..i + RMS_WINDOW];
                let energy: f32 = window.iter().map(|s| s * s).sum::<f32>() / RMS_WINDOW as f32;
                (i, energy)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(nominal);

        // Split at the midpoint of the quietest window
        let split = (best + RMS_WINDOW / 2).min(pcm.len());
        chunks.push(&pcm[offset..split]);
        offset = split;
    }

    chunks
}

/// Updates from the file transcription thread (separate from WhisperUpdate)
#[derive(Debug, Clone)]
pub enum FileUpdate {
    Status(String),
    Progress { chunk: usize, total: usize },
    Transcription(String),
    Error(String),
    Complete,
}

/// Helper to convert any symphonia sample type to f32
fn conv<T>(samples: &mut Vec<f32>, data: std::borrow::Cow<'_, symphonia::core::audio::AudioBuffer<T>>)
where
    T: symphonia::core::sample::Sample,
    f32: FromSample<T>,
{
    samples.extend(data.chan(0).iter().map(|v| f32::from_sample(*v)));
}

/// Load any supported audio file (wav, mp3, flac, ogg, m4a, etc.),
/// decode to mono f32 PCM, and resample to Whisper's target sample rate.
pub fn load_audio_file(path: &std::path::Path) -> Result<Vec<f32>, anyhow::Error> {
    let src = std::fs::File::open(path)?;
    let mss =
        symphonia::core::io::MediaSourceStream::new(Box::new(src), Default::default());

    let hint = symphonia::core::probe::Hint::new();
    let meta_opts: symphonia::core::meta::MetadataOptions = Default::default();
    let fmt_opts: symphonia::core::formats::FormatOptions = Default::default();

    let probed =
        symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow::anyhow!("No supported audio track found"))?;

    let dec_opts: DecoderOptions = Default::default();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)?;
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| anyhow::anyhow!("Unknown sample rate"))?;

    let mut pcm_data = Vec::new();

    while let Ok(packet) = format.next_packet() {
        // skip metadata updates
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
    }

    if pcm_data.is_empty() {
        anyhow::bail!("No audio data decoded from file");
    }

    // resample to target rate if needed
    if sample_rate as usize != TARGET_SAMPLE_RATE {
        convert_sample_rate(&pcm_data, sample_rate as usize)
            .map_err(|e| anyhow::anyhow!("Resampling failed: {}", e))
    } else {
        Ok(pcm_data)
    }
}

/// Transcribe an audio file, sending progress and results to the dedicated channel.
pub fn transcribe_file(
    path: std::path::PathBuf,
    model: WhichModel,
    tx: Sender<FileUpdate>,
    cancel: Arc<AtomicBool>,
) -> Result<(), anyhow::Error> {
    tx.send(FileUpdate::Status(format!(
        "Loading audio: {}",
        path.display()
    )))?;

    let resampled = load_audio_file(&path)?;
    let duration_secs = resampled.len() as f32 / TARGET_SAMPLE_RATE as f32;
    tx.send(FileUpdate::Status(format!(
        "Loaded {:.1}s of audio",
        duration_secs
    )))?;

    // load whisper model, forwarding download progress to our channel
    tx.send(FileUpdate::Status("Loading Whisper model...".to_string()))?;
    let tx_status = tx.clone();
    let mut ctx: WhisperContext = load_whisper_model(model, move |s| {
        tx_status.send(FileUpdate::Status(s)).ok();
    })?;
    tx.send(FileUpdate::Status("Model loaded".to_string()))?;

    // Split audio into ≤30s chunks at silence boundaries
    let chunks = split_at_silence(&resampled);
    let total_chunks = chunks.len();

    for (i, chunk) in chunks.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            tx.send(FileUpdate::Status("Cancelled".to_string()))?;
            return Ok(());
        }

        tx.send(FileUpdate::Progress {
            chunk: i + 1,
            total: total_chunks,
        })?;

        // generate mel spectrogram
        let mel_raw =
            mel::pcm_to_mel(ctx.config.num_mel_bins, chunk, &ctx.mel_filters);
        let num_bins = ctx.config.num_mel_bins;
        let num_mel_frames = mel_raw.len() / num_bins;
        let mel_tensor =
            Tensor::from_slice(&mel_raw, (1, num_bins, num_mel_frames), &ctx.device)?;

        // run decoder
        let tx_token = tx.clone();
        let token_cb = Some(move |text: String| {
            tx_token
                .send(FileUpdate::Status(text))
                .unwrap_or_default();
        });
        let (segments, last_tokens) = ctx.decoder.run(&mel_tensor, None, None, &token_cb)?;
        ctx.previous_content_tokens = last_tokens;

        for segment in &segments {
            for phrase in &segment.dr.text {
                tx.send(FileUpdate::Transcription(phrase.clone()))?;
            }
        }
    }

    tx.send(FileUpdate::Complete)?;
    Ok(())
}
