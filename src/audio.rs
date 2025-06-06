use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use rubato::Resampler;
use std::{
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

pub struct AudioChunk {
    data: Vec<f32>,
}

use crate::app::WhisperUpdate;
// whisper is trained on 16kHz audio
pub(crate) const TARGET_SAMPLE_RATE: usize = 16000; 

pub(crate) fn convert_sample_rate(
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
    let ratio = TARGET_SAMPLE_RATE as f64 / original_sample_rate as f64;
    let mut resampler =
        rubato::SincFixedIn::<f32>::new(ratio, 2.0, params, audio.len(), 1).unwrap();

    let waves_in = vec![audio; 1];
    let waves_out = resampler.process(&waves_in, None).unwrap();
    Ok(waves_out[0].to_vec())
}

// a thread that collects non-silent audio samples and sends them on
fn filter_audio_loop(
    app: Sender<WhisperUpdate>,
    audio_rx: Receiver<PcmAudio>,
    filtered_tx: Sender<PcmAudio>,
) {
    // here's the basic idea: receive 480 samples at a time (48000 / 100 = 480). If the max value
    // of the samples is above a threshold, then we know that there is a sound. If there is a sound,
    // then we can start recording the audio. Once we stop recording, we can send the recorded audio to Whisper.
    let mut under_threshold_count = 101;
    let mut recording_buffer: Vec<f32> = Vec::new();

    // a dynamic threshold (or something like silero-vad) would be better
    let threshold = 0.05f32;

    // accumulate data until we've been under the threshold for 100 samples
    loop {
        let PcmAudio{data, sample_rate} = match audio_rx.recv() {
            Ok(pcmaudio) => pcmaudio,
            Err(_) => {
                tracing::info!("Audio stream closed");
                return;
            }
        };

        let mut max = 0.0;
        for sample in data.iter() {
            if *sample > max {
                max = *sample;
            }
        }
        app.send(WhisperUpdate::Level(max))
            .expect("Failed to send level update");

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
            if under_threshold_count < 50 {
                // not long enough, keep listening
                recording_buffer.extend_from_slice(&data);
            } else {
                if recording_buffer.len() > 0 {
                    app.send(WhisperUpdate::Recording(false))
                        .expect("Failed to send recording update");
                    let resampled = convert_sample_rate(&recording_buffer, sample_rate).unwrap();
                    filtered_tx.send(PcmAudio{data: resampled, sample_rate: TARGET_SAMPLE_RATE}).expect("Send filtered");
                    recording_buffer.clear();
                }
            }
        }

        // Whisper is trained to process 30 seconds at a time. So if we've got that much, send it now.
        if recording_buffer.len() >= full_whisper_buffer {
            let resampled = convert_sample_rate(&recording_buffer, sample_rate).unwrap();
            filtered_tx.send(PcmAudio{data: resampled, sample_rate: TARGET_SAMPLE_RATE}).expect("Send filtered");
            recording_buffer.clear();
        }
    }
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

pub struct StreamState {
    // the app holds this handle to keep the stream open (and the thread alive)
    #[allow(dead_code)]
    stream: cpal::Stream,
}

pub struct AppDevice {
    pub name: String,
    pub device: cpal::Device,
    pub config: cpal::SupportedStreamConfig,
}

pub struct PcmAudio {
    pub data: Vec<f32>,
    pub sample_rate: usize,
}

// once the return value is dropped, listening stops
// and the sender is closed
pub fn start_audio_thread(
    app: Sender<WhisperUpdate>,
    app_device: &AppDevice,
) -> anyhow::Result<(StreamState, Receiver<PcmAudio>, Receiver<PcmAudio>)> {
    tracing::info!("Listening on device: {}", app_device.device.name()?);

    let (audio_tx, audio_rx) = mpsc::channel::<PcmAudio>();
    let (partial_tx, partial_rx) = mpsc::channel::<PcmAudio>();

    let err_fn = move |err| tracing::error!("an error occurred on stream: {}", err);

    let audio_config = &app_device.config;
    let channel_count = audio_config.channels() as usize;
    let sample_rate = audio_config.sample_rate().0 as usize;
    let data_callback = move |raw: &[f32], _: &_| {
        let data = raw
            .iter()
            .step_by(channel_count)
            .copied()
            .collect::<Vec<f32>>();
        audio_tx
            .send(PcmAudio {
                data: data.clone(),
                sample_rate,
            })
            .expect("Send audio data");
        partial_tx
            .send(PcmAudio {
                data,
                sample_rate,
            })
            .expect("Send partial audio data");
    };
    let config2 = app_device.config.clone();
    let stream = app_device.device.build_input_stream(
        &config2.into(),
        data_callback,
        err_fn,
        None,
    )?;

    stream.play()?;

    let app2 = app.clone();
    let (filtered_tx, filtered_rx): (Sender<PcmAudio>, Receiver<PcmAudio>) = mpsc::channel();
    thread::spawn(move || {
        filter_audio_loop(app2, audio_rx, filtered_tx);
    });

    Ok((StreamState { stream }, filtered_rx, partial_rx))
}
