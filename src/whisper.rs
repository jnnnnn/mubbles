use std::{
    path::Path,
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

use cpal::{
    traits::{DeviceTrait, StreamTrait},
    Device,
};
use rubato::Resampler;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

pub enum WhisperUpdate {
    Recording(bool),
    Transcribing(bool),
    Transcript(String),
    Level(f32),
}

#[allow(dead_code)] // we need to keep a handle of stream; state id will be used when transcribing audio out is implemented as there will be two streams
pub struct StreamState {
    stream: cpal::Stream,
    whisper_state_id: usize,
}

// once the return value is dropped, listening stops
pub fn start_listening(app: &Sender<WhisperUpdate>, device: &Device) -> StreamState {
    let config = device
        .default_input_config()
        .expect("Failed to get default input config");
    println!("Default input config: {:?}", config);
    println!(
        "Listening on device: {}",
        device.name().expect("device name")
    );

    let channels = config.channels();

    let (audio_tx, audio_rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = mpsc::channel();

    let err_fn = move |err| eprintln!("an error occurred on stream: {}", err);
    let data_callback =
        move |data: &[f32], _: &_| audio_tx.send(data.to_vec()).expect("Failed to send data");
    let stream = device
        .build_input_stream(&config.into(), data_callback, err_fn, None)
        .expect("Failed to build input stream");

    stream.play().expect("Failed to play stream");

    // load the model from either the local directory or from
    // ~/.cache/whisper/base.bin it must be in ggml format -- use
    // https://raw.githubusercontent.com/ggerganov/whisper.cpp/master/models/convert-pt-to-ggml.py
    // if you need to convert a pytorch model to ggml

    // check for a local model first
    let ctx = if Path::new("whisper.bin").exists() {
        println!("Loading local model");
        WhisperContext::new("whisper.bin").expect("failed to load model from base.bin")
    } else {
        let model = dirs::home_dir()
            .expect("No home")
            .join(".cache")
            .join("whisper")
            .join("tiny.bin") // tiny is faster (we're running on CPU) and quality is still pretty good
            .into_os_string()
            .into_string()
            .expect("No path conversion?");
        WhisperContext::new(model.as_str())
            .expect("failed to load model from ~/.cache/whisper/tiny.bin")
    };
    let state_id = 1usize;
    ctx.create_key(state_id).expect("failed to create key");

    let app2 = app.clone();
    let (filtered_tx, filtered_rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = mpsc::channel();
    thread::spawn(move || {
        filter_audio_loop(app2, audio_rx, filtered_tx, channels);
    });

    let app2 = app.clone();
    thread::spawn(move || {
        whisper_loop(app2, ctx, filtered_rx);
    });

    StreamState {
        stream,
        whisper_state_id: state_id,
    }
}

fn whisper_loop(
    app: Sender<WhisperUpdate>,
    ctx: WhisperContext<usize>,
    filtered_rx: Receiver<Vec<f32>>,
) {
    loop {
        // first recv needs to be blocking to prevent the thread from spinning
        let mut aggregated_data = match filtered_rx.recv() {
            Ok(data) => data,
            Err(_) => {
                println!("Filtered stream closed");
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
        whisperize(&ctx, &aggregated_data, &app);
    }
}

// convert an audio stream into a stream of text chunks using Whisper
fn filter_audio_loop(
    app: Sender<WhisperUpdate>,
    audio_rx: Receiver<Vec<f32>>,
    filtered_tx: Sender<Vec<f32>>,
    channels: u16,
) {
    // here's the basic idea: receive 480 samples at a time (48000 / 100 = 480). If the max value
    // of the samples is above a threshold, then we know that there is a sound. If there is a sound,
    // then we can start recording the audio. Once we stop recording, we can send the recorded audio to Whisper.
    let mut under_threshold_count = 101;
    let mut recording_buffer: Vec<f32> = Vec::new();

    // get a very simple threshold by recording one second of audio and finding the max value
    // a dynamic threshold (or something like silero-vad) would be better
    let mut threshold = 0.02f32;
    for _ in 0..100 {
        let data = match audio_rx.recv() {
            Ok(data) => data,
            Err(_) => {
                println!("Audio stream closed");
                return;
            }
        };
        let mut max = 0.0;
        for sample in data.iter() {
            if *sample > max {
                max = *sample;
            }
        }
        if max > threshold {
            threshold = max;
        }
    }

    println!("Threshold: {}", threshold);

    // accumulate data until we've been under the threshold for 100 samples
    loop {
        let mut data = match audio_rx.recv() {
            Ok(data) => data,
            Err(_) => {
                println!("Audio stream closed");
                return;
            }
        };

        // Convert audio to 16KHz mono f32 samples, as required by the model.
        // These utilities are provided for convenience, but can be replaced with custom conversion logic.
        // SIMD variants of these functions are also available on nightly Rust (see the docs).
        if channels == 2 {
            data = whisper_rs::convert_stereo_to_mono_audio(&data).expect("monoize");
        } else if channels != 1 {
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
        if max > threshold {
            if under_threshold_count > 100 {
                app.send(WhisperUpdate::Recording(true))
                    .expect("Failed to send recording update");
            }
            recording_buffer.extend_from_slice(&data);
            under_threshold_count = 0;
        } else {
            under_threshold_count += 1;
            if under_threshold_count < 100 {
                recording_buffer.extend_from_slice(&data);
            } else {
                if recording_buffer.len() > 0 {
                    app.send(WhisperUpdate::Recording(false))
                        .expect("Failed to send recording update");
                    let resampled = convert_sample_rate(&recording_buffer, 48000, 16000).unwrap();
                    filtered_tx
                        .send(resampled)
                        .expect("Send filtered audio to whisper thread");
                    recording_buffer.clear();
                }
            }
        }
    }
}

fn whisperize(ctx: &WhisperContext<usize>, resampled: &[f32], app: &Sender<WhisperUpdate>) {
    // Consider making the beam size configurable for user performance tuning.
    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: 3,
        patience: 1f32,
    });
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    // Run the model.
    let state_id = 1usize;
    app.send(WhisperUpdate::Transcribing(true))
        .expect("Failed to send transcribing update");
    ctx.full(&state_id, params, resampled)
        .expect("failed to run model");

    // Iterate through the segments of the transcript.
    let num_segments = ctx
        .full_n_segments(&state_id)
        .expect("failed to get number of segments");
    for i in 0..num_segments {
        // Get the transcribed text and timestamps for the current segment.
        let segment = ctx
            .full_get_segment_text(&state_id, i)
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
        app.send(WhisperUpdate::Transcript(segment.clone()))
            .expect("Failed to send transcript update");
    }
    app.send(WhisperUpdate::Transcribing(false))
        .expect("Failed to send transcribing update");
}

fn convert_sample_rate(
    audio: &[f32],
    original_sample_rate: u32,
    target_sample_rate: i32,
) -> Result<Vec<f32>, &'static str> {
    let params = rubato::InterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: rubato::InterpolationType::Linear,
        oversampling_factor: 256,
        window: rubato::WindowFunction::BlackmanHarris2,
    };
    let ratio = target_sample_rate as f64 / original_sample_rate as f64;
    let mut resampler =
        rubato::SincFixedIn::<f32>::new(ratio, 2.0, params, audio.len(), 1).unwrap();

    let waves_in = vec![audio; 1];
    let waves_out = resampler.process(&waves_in, None).unwrap();
    Ok(waves_out[0].to_vec())
}
