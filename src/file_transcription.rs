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

use crate::{
    audio::{convert_sample_rate, TARGET_SAMPLE_RATE},
    mel,
    whisper::{load_whisper_model, WhichModel, WhisperContext},
};

/// Updates from the file transcription thread (separate from WhisperUpdate)
#[derive(Debug, Clone)]
pub enum FileUpdate {
    Status(String),
    Progress { chunk: usize, total: usize },
    Transcription(String),
    Error(String),
    Complete,
}

/// Load a WAV audio file, convert to mono, and resample to Whisper's target sample rate.
pub fn load_audio_file(path: &std::path::Path) -> Result<Vec<f32>, anyhow::Error> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    // decode samples to f32
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let max_value = 2_i32.pow(spec.bits_per_sample as u32 - 1) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_value))
                .collect::<Result<Vec<_>, _>>()?
        }
    };

    // convert to mono
    let mono = if spec.channels == 1 {
        samples
    } else {
        samples
            .chunks(spec.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
            .collect()
    };

    // resample to target rate
    if spec.sample_rate as usize != TARGET_SAMPLE_RATE {
        convert_sample_rate(&mono, spec.sample_rate as usize)
            .map_err(|e| anyhow::anyhow!("Resampling failed: {}", e))
    } else {
        Ok(mono)
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

    // whisper processes audio in 30s chunks
    const CHUNK_SIZE: usize = TARGET_SAMPLE_RATE * 30;
    let total_chunks = (resampled.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (i, chunk) in resampled.chunks(CHUNK_SIZE).enumerate() {
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
