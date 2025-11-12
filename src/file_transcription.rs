//! File transcription module
//! 
//! Handles transcribing audio files using the Whisper model.

use std::sync::mpsc::Sender;

use crate::{
    app::WhisperUpdate,
    audio::PcmAudio,
    whisper::{load_whisper_model, whisperize, WhichModel, WhisperContext},
};

/// Transcribe an audio file
pub fn transcribe_file(
    path: std::path::PathBuf,
    model: WhichModel,
    _accuracy: usize,
    app: Sender<WhisperUpdate>,
) -> Result<(), anyhow::Error> {
    app.send(WhisperUpdate::Status(format!(
        "Loading audio file: {}",
        path.display()
    )))?;

    // we need to get the file from disk
    let mut reader = hound::WavReader::open(&path)?;
    let spec = reader.spec();
    
    app.send(WhisperUpdate::Status(format!(
        "Audio format: {} Hz, {} channels, {} bits",
        spec.sample_rate, spec.channels, spec.bits_per_sample
    )))?;

    // whisper requires samples as f32
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.samples::<f32>()
                .collect::<Result<Vec<_>, _>>()?
        }
        hound::SampleFormat::Int => {
            let max_value = 2_i32.pow(spec.bits_per_sample as u32 - 1) as f32;
            reader.samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_value))
                .collect::<Result<Vec<_>, _>>()?
        }
    };

    // whisper model expects mono audio, so convert if necessary
    let mono_samples: Vec<f32> = if spec.channels == 1 {
        samples
    } else {
        samples.chunks(spec.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
            .collect()
    };

    app.send(WhisperUpdate::Status(format!(
        "Loaded {} samples ({:.2} seconds)",
        mono_samples.len(),
        mono_samples.len() as f32 / spec.sample_rate as f32
    )))?;

    // whisper requires 16kHz sample rate, so resample if necessary
    let resampled = if spec.sample_rate != 16000 {
        app.send(WhisperUpdate::Status(format!(
            "Resampling from {} Hz to 16000 Hz",
            spec.sample_rate
        )))?;
        crate::audio::convert_sample_rate(&mono_samples, spec.sample_rate as usize)
            .map_err(|e| anyhow::anyhow!("Resampling failed: {}", e))?
    } else {
        mono_samples
    };

    // We need a whisper model in memory to run transcriptions
    app.send(WhisperUpdate::Status("Loading Whisper model...".to_string()))?;
    let mut ctx: WhisperContext = load_whisper_model(model, app.clone())?;
    
    // whisper processes audio in 30s chunks, so we will do that here
    const CHUNK_SIZE: usize = 16000 * 30; // 30 seconds at 16kHz
    let total_chunks = (resampled.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    for (i, chunk) in resampled.chunks(CHUNK_SIZE).enumerate() {
        app.send(WhisperUpdate::Status(format!(
            "Transcribing chunk {}/{}...",
            i + 1,
            total_chunks
        )))?;
        
        // Create PcmAudio for compatibility with existing whisper code
        let _pcm = PcmAudio {
            data: chunk.to_vec(),
            sample_rate: 16000,
        };
        
        // Use the existing whisperize function
        whisperize(&mut ctx, chunk, &app)?;
    }

    app.send(WhisperUpdate::FileTranscriptionComplete)?;
    Ok(())
}
