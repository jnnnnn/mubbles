use std::collections::VecDeque;

use crate::{
    app::WhisperUpdate,
    audio::{PcmAudio, TARGET_SAMPLE_RATE},
    mel::pcm_to_mel_frame,
    whisper::{load_whisper_model, WhichModel},
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

fn partial_loop(
    app: std::sync::mpsc::Sender<crate::app::WhisperUpdate>,
    partial_rx: std::sync::mpsc::Receiver<PcmAudio>,
) -> Result<(), anyhow::Error> {
    let mut recent_samples = VecDeque::<f32>::new();
    let mut offset: usize = 0;

    let whisper_context =
        load_whisper_model(WhichModel::TinyEn).expect("Failed to load whisper model");

    loop {
        let PcmAudio { data, sample_rate } = match partial_rx.recv() {
            Ok(pcm) => pcm,
            Err(_) => {
                tracing::info!("Partial stream closed");
                return Ok(());
            }
        };

        let extra = crate::audio::convert_sample_rate(&data, sample_rate).unwrap();
        let extra_len = extra.len();
        recent_samples.extend(extra);
        // we only want to preview the most recent samples
        let max_size = (5 * TARGET_SAMPLE_RATE) as usize;
        let stale = recent_samples.len().saturating_sub(max_size);
        recent_samples.rotate_left(stale);
        offset -= stale;
        recent_samples.truncate(max_size);
        tracing::info!(
            "Received {} samples, recent_samples {}, offset {}",
            extra_len,
            recent_samples.len(),
            offset
        );

        // mel frames are generated with a bit of lookahead -- 160 samples plus lookahead of 240 goes into the fft.
        if offset + 400 < recent_samples.len() {
            let frames = (recent_samples.len() - offset - 240) / 160;
            let _pcm = recent_samples
                .range(offset..offset + frames * 160 + 240)
                .cloned()
                .collect::<Vec<f32>>();
            tracing::info!(
                "Generating {} mel frames from {} samples",
                frames,
                _pcm.len()
            );
            offset += frames * 160;
            let mels = pcm_to_mel_frame(80, &_pcm, &whisper_context.mel_filters);
            let mut frame_count = 0;

            for frame in mels {
                frame_count += 1;
                app.send(WhisperUpdate::MelFrame(frame.into()))
                    .expect("Failed to send mel frame");
            }
            tracing::debug!("Generated {} mel frames", frame_count);
        }
    }
}
