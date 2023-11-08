use std::{
    path::Path,
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, SupportedStreamConfig,
};
use rubato::Resampler;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperState};

pub enum WhisperUpdate {
    Recording(bool),
    Transcribing(bool),
    Transcript(String),
    Level(f32),
}

pub struct StreamState {
    // the app holds this handle to keep the stream open (and the whisper context / thread alive)
    #[allow(dead_code)]
    stream: cpal::Stream,
}

pub struct AppDevice {
    pub name: String,
    device: Device,
    config: cpal::SupportedStreamConfig,
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

pub struct WhisperParams {
    pub accuracy: usize, // 1 for greedy, more for beam search
}

pub fn load_whisper_model() -> WhisperContext {
    // load the model from either the local directory or from
    // ~/.cache/whisper/base.bin it must be in ggml format -- use
    // https://raw.githubusercontent.com/ggerganov/whisper.cpp/master/models/convert-pt-to-ggml.py
    // if you need to convert a pytorch model to ggml
    let model_file = "medium.en.bin";
    // check for a local model first
    let ctx = if Path::new(model_file).exists() {
        tracing::info!("Loading local model");
        WhisperContext::new(model_file).expect("failed to load model from local folder")
    } else {
        let model = dirs::home_dir()
            .expect("No home")
            .join(".cache")
            .join("whisper")
            .join(model_file) // tiny is faster (we're running on CPU) and quality is still pretty good
            .into_os_string()
            .into_string()
            .expect("No path conversion?");
        WhisperContext::new(model.as_str())
            .expect("failed to load model from local directory and ~/.cache/whisper/")
    };
    ctx
}


// once the return value is dropped, listening stops
// and the sender is closed
pub fn start_listening(
    app: &Sender<WhisperUpdate>,
    app_device: &AppDevice,
    params: WhisperParams,
) -> Option<StreamState> {
    tracing::info!(
        "Listening on device: {}",
        app_device.device.name().expect("device name")
    );

    let (audio_tx, audio_rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = mpsc::channel();

    let err_fn = move |err| tracing::error!("an error occurred on stream: {}", err);
    let data_callback =
        move |data: &[f32], _: &_| audio_tx.send(data.to_vec()).expect("Failed to send data");
    let stream = app_device
        .device
        .build_input_stream(
            &app_device.config.clone().into(),
            data_callback,
            err_fn,
            None,
        )
        .expect("Failed to build input stream");

    stream.play().expect("Failed to play stream");

    let app2 = app.clone();
    let (filtered_tx, filtered_rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = mpsc::channel();
    let config2 = app_device.config.clone();
    thread::spawn(move || {
        filter_audio_loop(app2, audio_rx, filtered_tx, config2);
    });

    let app2 = app.clone();
    thread::spawn(move || {
        whisper_loop(app2, filtered_rx, params);
    });

    Some(StreamState { stream })
}

fn whisper_loop(
    app: Sender<WhisperUpdate>,
    filtered_rx: Receiver<Vec<f32>>,
    params: WhisperParams,
) {
    let ctx = load_whisper_model();
    let mut state = ctx.create_state().expect("failed to create state");
    loop {
        // first recv needs to be blocking to prevent the thread from spinning
        let mut aggregated_data = match filtered_rx.recv() {
            Ok(data) => data,
            Err(_) => {
                tracing::info!("Filtered stream closed");
                return;
            }
        };
        // because whisper processes audio in chunks of 30 seconds (and takes a while to do so), it's
        // worth aggregating several chunks of audio before sending it to whisper (if they are available)
        while aggregated_data.len() < 48000 * 30 {
            match filtered_rx.try_recv() {
                Ok(data) => aggregated_data.extend(data),
                Err(_) => break,
            }
        }
        whisperize(&mut state, &aggregated_data, &app, &params);
    }
}

// convert an audio stream into a stream of text chunks using Whisper
fn filter_audio_loop(
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
            data = whisper_rs::convert_stereo_to_mono_audio(&data).expect("monoize");
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

fn whisperize(
    state: &mut WhisperState<'_>,
    resampled: &[f32],
    app: &Sender<WhisperUpdate>,
    params: &WhisperParams,
) {
    // Consider making the beam size configurable for user performance tuning.
    // a beam_size of 6 is recommended; 1 is fast but use Greedy instead of BeamSearch; 16 crashes.

    let mut params = FullParams::new(match params.accuracy {
        1 => SamplingStrategy::Greedy { best_of: 1 },
        _ => SamplingStrategy::BeamSearch {
            beam_size: params.accuracy as i32,
            patience: 1.0,
        },
    });

    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    // record start time
    let start = std::time::Instant::now();

    //write_raw_floats_to_file(resampled);

    // Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text
    app.send(WhisperUpdate::Transcribing(true))
        .expect("Failed to send transcribing update");
    state.full(params, resampled).expect("failed to run model");

    // Iterate through the segments of the transcript.
    let num_segments = state
        .full_n_segments()
        .expect("failed to get number of segments");
    for i in 0..num_segments {
        // Get the transcribed text and timestamps for the current segment.
        let segment = state
            .full_get_segment_text(i)
            .expect("failed to get segment");
        // if trimmed segment starts with punctuation, it's probably something
        // like [BLANK_AUDIO] or (crickets chirping). Better to ignore this sort
        // of noise.
        if segment
            .trim_start()
            .starts_with(|c: char| c.is_ascii_punctuation())
        {
            continue;
        }
        
        tracing::info!("scribe: {}", segment.trim());
        app.send(WhisperUpdate::Transcript(segment.clone()))
            .expect("Failed to send transcript update");
    }
    app.send(WhisperUpdate::Transcribing(false))
        .expect("Failed to send transcribing update");

    // trace how long it took and how long the input was
    let duration = start.elapsed().as_secs() as f64;
    let input_duration = resampled.len() as f64 / 16000.;
    tracing::info!(
        "Transcribed {} seconds of audio in {} seconds",
        input_duration,
        duration
    );

}

#[allow(dead_code)]
fn write_raw_floats_to_file(resampled: &[f32]) {
    // write resampled to a binary file
    let contents = unsafe {
        std::slice::from_raw_parts(
            resampled.as_ptr() as *const u8,
            resampled.len() * std::mem::size_of::<f32>(),
        )
    };
    std::fs::write("resampled.bin", contents).expect("Unable to write file");
}

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
