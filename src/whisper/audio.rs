// This file does all the glue required to move and reformat audio from the operating system into Whisper.
// Major tasks include
// 1. enumerating audio devices,
// 2. starting a stream from a selected audio device,
// 3. converting audio from whatever bit depth and rate to Whisper's required 16bit mono 16Khz,
// 4. filtering out silence, and
// 5. sending chunks of filtered audio to a channel that Whisper can read it from.

use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;

use cpal;
use cpal::Device;
use cpal::SupportedStreamConfig;

pub struct AppDevice {
    pub name: String,
    pub(crate) device: Device,
    pub(crate) config: cpal::SupportedStreamConfig,
}

pub fn get_devices() -> Vec<AppDevice> {
    let host = cpal::default_host();
    let mut all = Vec::new();
    let input_devices = match host.input_devices() {
        Ok(devices) => devices.collect(),
        Err(whatever) => {
            tracing::warn!("Failed to get input devices: {}", whatever);
            Vec::new()
        }
    };
    for device in input_devices {
        let config = match device.default_input_config() {
            Ok(config) => config,
            Err(whatever) => {
                tracing::info!("Failed to get config for {:?}: {}", device.name(), whatever);
                continue;
            }
        };
        let name = device.name().unwrap_or("Unknown".to_string());
        all.push(AppDevice {
            name,
            device,
            config,
        });
    }
    let output_devices = match host.output_devices() {
        Ok(devices) => devices.collect(),
        Err(whatever) => {
            tracing::warn!("Failed to get output devices: {}", whatever);
            Vec::new()
        }
    };
    for device in output_devices {
        let config = match device.default_output_config() {
            Ok(config) => config,
            Err(whatever) => {
                tracing::info!("Failed to get config for {:?}: {}", device.name(), whatever);
                continue;
            }
        };
        let name = device.name().unwrap_or("Unknown".to_string());
        all.push(AppDevice {
            name,
            device,
            config,
        });
    }

    all
}

// convert an audio stream into a stream of text chunks using Whisper
pub fn filter_audio_loop(
    app: Sender<WhisperUpdate>,
    audio_rx: Receiver<Vec<f32>>,
    filtered_tx: Sender<Vec<f32>>,
    device_config: SupportedStreamConfig,
) {
    // here's the basic idea: receive 480 samples at a time (48000 / 100 = 480). If the max value
    // of the samples is above a threshold, then we know that there is a sound. If there is a sound,
    // then we can start recording the audio. Once we stop recording, we can send the recorded audio to Whisper.
    let mut under_threshold_count = 101;
    let mut recording_buffer: Vec<f32> = Vec::new();

    // a dynamic threshold (or something like silero-vad) would be better
    let threshold = 0.02f32;

    // accumulate data until we've been under the threshold for 100 samples
    loop {
        let mut data = match audio_rx.recv() {
            Ok(data) => data,
            Err(_) => {
                tracing::info!("Audio stream closed");
                return;
            }
        };

        // Convert audio to 16KHz mono f32 samples, as required by the model.
        // These utilities are provided for convenience, but can be replaced with custom conversion logic.
        // SIMD variants of these functions are also available on nightly Rust (see the docs).
        if device_config.channels() == 2 {
            data = convert_stereo_to_mono_audio(&data).expect("monoize");
        } else if device_config.channels() != 1 {
            panic!(">2 channels unsupported");
        }

        let mut max = 0.0;
        for sample in data.iter() {
            if *sample > max {
                max = *sample;
            }
        }
        app.send(WhisperUpdate::Level(max))
            .expect("Failed to send level update");

        let sample_rate = device_config.sample_rate().0 as usize;
        let full_whisper_buffer = 30/*seconds*/ * sample_rate /*samples per second*/;

        if max > threshold {
            if under_threshold_count > 100 {
                // we've been listening to silence for a while, so we stopped recording. Indicate that we're listening again.
                app.send(WhisperUpdate::Recording(true))
                    .expect("Failed to send recording update");
            }
            recording_buffer.extend_from_slice(&data);
            under_threshold_count = 0;
        } else {
            // the incoming audio is below the threshold. Check how long it's been silent for.
            under_threshold_count += 1;
            if under_threshold_count < 100 {
                // not long enough, keep listening
                recording_buffer.extend_from_slice(&data);
            } else {
                if recording_buffer.len() > 0 {
                    app.send(WhisperUpdate::Recording(false))
                        .expect("Failed to send recording update");
                    let resampled = convert_sample_rate(&recording_buffer, sample_rate).unwrap();
                    filtered_tx.send(resampled).expect("Send filtered");
                    recording_buffer.clear();
                }
            }
        }

        // Whisper is trained to process 30 seconds at a time. So if we've got that much, send it now.
        if recording_buffer.len() >= full_whisper_buffer {
            let resampled = convert_sample_rate(&recording_buffer, sample_rate).unwrap();
            filtered_tx.send(resampled).expect("Send filtered");
            recording_buffer.clear();
        }
    }
}

fn convert_stereo_to_mono_audio(samples: &[f32]) -> Result<Vec<f32>, &'static str> {
    if samples.len() & 1 != 0 {
        return Err("The stereo audio vector has an odd number of samples. \
            This means a half-sample is missing somewhere");
    }

    Ok(samples
        .chunks_exact(2)
        .map(|x| (x[0] + x[1]) / 2.0)
        .collect())
}

use cpal::traits::DeviceTrait;
use cpal::traits::HostTrait;
use rubato::Resampler;

use super::WhisperUpdate;
fn convert_sample_rate(
    audio: &[f32],
    original_sample_rate: usize,
) -> Result<Vec<f32>, &'static str> {
    let params = rubato::SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: rubato::SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: rubato::WindowFunction::BlackmanHarris2,
    };
    const TARGET_SAMPLE_RATE: f64 = 16000.;
    let ratio = TARGET_SAMPLE_RATE / original_sample_rate as f64;
    let mut resampler =
        rubato::SincFixedIn::<f32>::new(ratio, 2.0, params, audio.len(), 1).unwrap();

    let waves_in = vec![audio; 1];
    let waves_out = resampler.process(&waves_in, None).unwrap();
    Ok(waves_out[0].to_vec())
}
