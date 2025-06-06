use std::collections::VecDeque;

use crate::audio::{PcmAudio, TARGET_SAMPLE_RATE};

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

    let recent_samples = VecDeque::<f32>::new();

    loop {
        let PcmAudio{data, sample_rate} = match partial_rx.recv() {
            Ok(pcm) => pcm,
            Err(_) => {
                tracing::info!("Partial stream closed");
                return Ok(());
            }
        };

        // push data into the resampled buffer
        let extra = crate::audio::convert_sample_rate(&data, sample_rate).unwrap();
        let extra_len = extra.len();
        recent_samples.extend(extra);
        // keep the resampled buffer under 5 seconds, discarding the oldest samples
        let max_size = (5 * TARGET_SAMPLE_RATE) as usize;
        let stale = recent_samples.len().saturating_sub(max_size);
        recent_samples.rotate_left(stale);
        // keep the offset in sync with the resampled buffer
        partial.unused += extra_len - stale;
        recent_samples.truncate(max_size);
        // if we have enough for at least one more mel frame, generate
        if partial.unused >= 400 {
            // there are enough `unused` samples that we can generate at least one more frame.
            // each frame is 160 samples, but the mel needs to look ahead 240 samples to get the next frame.
            // so we can generate `floor((unused - 240) / 160)` frames.
            let frames = (partial.unused - 240) / 160;
            let _pcm = partial
                .resampled
                .range(partial.unused..partial.unused + frames * 160 + 240)
                .cloned()
                .collect::<Vec<f32>>();
            partial.unused -= frames * 160;
            let mels = pcm_to_mel_frame(80, pcm, mel_filters);
            for frame in mels.chunks(80) {
                app.send(WhisperUpdate::MelFrame(frame.to_vec()))
                    .expect("Failed to send mel update");
            }
        }
    }

    Ok(())
}
