use std::collections::VecDeque;

use crate::{
    app::WhisperUpdate,
    audio::{PcmAudio, TARGET_SAMPLE_RATE},
    mel::pcm_to_mel_frame,
    whisper::{load_whisper_model, WhichModel, WhisperContext},
};

pub(crate) fn start_partial_thread(
    app: std::sync::mpsc::Sender<crate::app::WhisperUpdate>,
    partial_rx: std::sync::mpsc::Receiver<PcmAudio>,
) {
    let result = std::thread::Builder::new()
        .name("partials".to_string())
        .spawn(move || match partial_loop(app, partial_rx) {
            Ok(_) => tracing::info!("Partials thread finished successfully"),
            Err(e) => tracing::error!("Partials thread failed: {:?}", e),
        });
    match result {
        Ok(_) => tracing::info!("Partials thread started"),
        Err(e) => tracing::error!("Failed to start partials thread: {:?}", e),
    }
}

struct PartialAudio {
    resampled: VecDeque<f32>,
    unused: usize, // the number of samples that have not yet been used to generate mel frames
}

pub const PARTIAL_MEL_BINS: usize = 80; // smaller models store MEL_BINS frequency bins per frame
const MEL_FRAME_CAPACITY: usize = 500; // Mel 10ms per frame for 5s

fn partial_loop(
    app: std::sync::mpsc::Sender<crate::app::WhisperUpdate>,
    partial_rx: std::sync::mpsc::Receiver<PcmAudio>,
) -> Result<(), anyhow::Error> {
    let mut recent_samples = VecDeque::<f32>::new();
    let mut offset: usize = 0;

    let mut whisper_context =
        load_whisper_model(WhichModel::TinyEn).expect("Failed to load whisper model");

    let mut last_5s_mel = VecDeque::<[f32; PARTIAL_MEL_BINS]>::new();
    loop {
        accumulate_audio(&mut recent_samples, &mut offset, &partial_rx)?;
        let filters = whisper_context.mel_filters.as_slice();
        generate_new_mel_frames(
            &mut recent_samples,
            &mut offset,
            &mut last_5s_mel,
            filters,
            &app,
        )?;

        if last_5s_mel.len() == 0 {
            continue;
        }
        perform_partial_transcription(&mut last_5s_mel, &mut whisper_context, &app)?;
    }
}

const PAD_MEL: usize = 30; // an extra second of silence keeps the model stable when the last frame would otherwise be part way through a word
const MEL_SILENT: f32 = -10.0; // the mel value for silence, used to pad the mel frames

// Fades a mel value to "silence" over PAD_MEL frames.
// This is used to avoid abrupt changes in the mel spectrogram at the start and end of the audio which can confuse the model.
fn fade_factor(frame: usize, n_frames: usize) -> f32 {
    if frame < PAD_MEL {
        (PAD_MEL - frame) as f32 / PAD_MEL as f32
    } else if frame >= n_frames - PAD_MEL {
        (n_frames - frame) as f32 / PAD_MEL as f32
    } else {
        1.0
    }
}

fn perform_partial_transcription(
    last_5s_mel: &mut VecDeque<[f32; 80]>,
    whisper_context: &mut WhisperContext,
    app: &std::sync::mpsc::Sender<WhisperUpdate>,
) -> Result<(), anyhow::Error> {
    let n_mel_frames = last_5s_mel.len();
    // whisper takes mel as a flat vector, ordered as [mel_bin, mel_frame]
    let mut melvec = vec![-10.0; n_mel_frames * PARTIAL_MEL_BINS];
    for (f, &frame) in last_5s_mel.iter().enumerate() {
        let fade = fade_factor(f, n_mel_frames);
        for (b, &bin) in frame.iter().enumerate() {
            melvec[b * n_mel_frames + (PAD_MEL + f)] = (bin - MEL_SILENT) * fade + MEL_SILENT;
        }
    }
    crate::mel::normalize(&mut melvec);
    app.send(WhisperUpdate::Mel(melvec.clone()))?;

    let c = &whisper_context.config;
    let mel_tensor = candle_core::Tensor::from_slice(
        melvec.as_slice(),
        (1, c.num_mel_bins, n_mel_frames),
        &whisper_context.device,
    )?;
    let start_time = std::time::Instant::now();
    let (segments_results, _last_segment_content_tokens) =
        whisper_context
            .decoder
            .run(&mel_tensor, None, None, n_mel_frames as f32 / 100.0)?;
    let elapsed = start_time.elapsed();
    tracing::debug!(
        "Partial transcription took {:.2?} for {} mel frames",
        elapsed,
        n_mel_frames
    );

    match segments_results.last() {
        Some(segment) => {
            app.send(WhisperUpdate::Alignment(segment.dr.alignment.clone()))?;
        }
        None => {
            tracing::warn!("No segments found in results");
        }
    }
    Ok(())
}

fn generate_new_mel_frames(
    recent_samples: &mut VecDeque<f32>,
    offset: &mut usize,
    last_5s_mel: &mut VecDeque<[f32; 80]>,
    mel_filters: &[f32],
    app: &std::sync::mpsc::Sender<WhisperUpdate>,
) -> Result<(), anyhow::Error> {
    // mel frames are generated with a bit of lookahead -- 160 samples plus lookahead of 240 goes into the fft.
    if *offset + 400 < recent_samples.len() {
        let frames = (recent_samples.len() - *offset - 240) / 160;
        let _pcm = recent_samples
            .range(*offset..*offset + frames * 160 + 240)
            .cloned()
            .collect::<Vec<f32>>();
        tracing::debug!(
            "Generating {} mel frames from {} samples",
            frames,
            _pcm.len()
        );
        *offset += frames * 160;
        let mels = pcm_to_mel_frame(PARTIAL_MEL_BINS, &_pcm, &mel_filters);
        let mut frame_count = 0;

        for frame in mels {
            frame_count += 1;
            //app.send(WhisperUpdate::MelFrame(frame.into()))?;
            if last_5s_mel.len() >= MEL_FRAME_CAPACITY {
                last_5s_mel.pop_front();
            }
            last_5s_mel.push_back(frame);
        }
        tracing::debug!("Generated {} mel frames", frame_count);
    }
    Ok(())
}

fn accumulate_audio(
    recent_samples: &mut VecDeque<f32>,
    offset: &mut usize,
    partial_rx: &std::sync::mpsc::Receiver<PcmAudio>,
) -> Result<(), anyhow::Error> {
    let PcmAudio {
        mut data,
        sample_rate,
    } = match partial_rx.recv() {
        Ok(pcm) => pcm,
        Err(_) => {
            tracing::info!("Partial stream closed");
            return Ok(());
        }
    };
    // consume any other messages in the queue
    while let Ok(pcm) = partial_rx.try_recv() {
        let PcmAudio {
            data: other,
            sample_rate: _,
        } = pcm;
        data.extend(other);
    }
    let max_size = (5 * TARGET_SAMPLE_RATE) as usize;
    let mut extra = crate::audio::convert_sample_rate(&data, sample_rate).unwrap();
    extra.truncate(max_size);
    let extra_len = extra.len();
    recent_samples.extend(extra);
    // we only want to preview the most recent samples
    let stale = recent_samples.len().saturating_sub(max_size);
    recent_samples.rotate_left(stale);
    *offset -= stale;
    recent_samples.truncate(max_size);
    tracing::debug!(
        "Received {} samples, recent_samples {}, offset {}",
        extra_len,
        recent_samples.len(),
        offset
    );
    Ok(())
}
