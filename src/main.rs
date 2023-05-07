#![warn(clippy::all, rust_2018_idioms)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::sync::mpsc::{self, Receiver, Sender};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::Resampler;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    // Log to stdout (if you run with `RUST_LOG=debug`).
    tracing_subscriber::fmt::init();

    dosoundtest();

    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "subbles",
        native_options,
        Box::new(|cc| Box::new(subbles::TemplateApp::new(cc))),
    )
}

fn dosoundtest() {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("no default input device");
    println!("Input device: {}", device.name().expect("Name not found"));
    let config = device
        .default_input_config()
        .expect("Failed to get default input config");
    println!("Default input config: {:?}", config);

    let (tx, rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = mpsc::channel();

    let ctx =
        WhisperContext::new("C:/Users/J/.cache/whisper/base.bin").expect("failed to load model");
    let state_id = "1";
    // Create a state
    ctx.create_key(state_id).expect("failed to create key");

    // A flag to indicate that recording is in progress.
    println!("Begin recording...");

    let err_fn = move |err| {
        eprintln!("an error occurred on stream: {}", err);
    };
    let sender = tx.clone();
    let data_callback = move |data: &[f32], _: &_| write_input_data(data, &sender);
    let stream = device
        .build_input_stream(&config.into(), data_callback, err_fn, None)
        .expect("Failed to build input stream");

    stream.play().expect("Failed to play stream");

    // here's the basic idea: receive 480 samples at a time (48000 / 100 = 480). If the max value
    // of the samples is above a threshold, then we know that there is a sound. If there is a sound,
    // then we can start recording the audio. Once we start recording, we can record for a 5 seconds
    // and then stop recording. Once we stop recording, we can send the recorded audio to Whisper.
    let mut under_threshold_count = 101;
    let mut recording_buffer: Vec<f32> = Vec::new();

    // get a threshold by recording some audio and finding the max value
    let mut threshold = 0.0;
    for _ in 0..100 {
        let data = rx.recv().expect("Failed to receive data");
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
    loop {
        let data = rx.recv().expect("Failed to receive data");
        let mut max = 0.0;
        for sample in data.iter() {
            if *sample > max {
                max = *sample;
            }
        }
        if max > threshold {
            if under_threshold_count > 100 {
                println!("Recording started");
            }
            under_threshold_count = 0;
            recording_buffer.extend_from_slice(&data);
        } else {
            under_threshold_count += 1;
            if under_threshold_count < 100 {
                recording_buffer.extend_from_slice(&data);
            } else {
                if recording_buffer.len() > 0 {
                    println!("Recording stopped");
                    let resampled = convert_sample_rate(&recording_buffer, 48000, 16000).unwrap();
                    println!("Resampled length: {}", resampled.len());
                    whisperize(&ctx, &resampled);
                    recording_buffer.clear();
                }
            }
        }
    }
}

fn whisperize(ctx: &WhisperContext<&str>, resampled: &[f32]) {
    // Create a params object for running the model.
    // The number of past samples to consider defaults to 0.
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    // Run the model.
    let state_id = "1";
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
        let start_timestamp = ctx
            .full_get_segment_t0(&state_id, i)
            .expect("failed to get start timestamp");
        let end_timestamp = ctx
            .full_get_segment_t1(&state_id, i)
            .expect("failed to get end timestamp");

        // Print the segment to stdout.
        println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
    }
}

fn write_input_data(data: &[f32], sender: &Sender<Vec<f32>>) {
    sender.send(data.to_vec()).expect("Failed to send data");
}

//  windowed sinc function resampling
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

// when compiling to web using trunk.
#[cfg(target_arch = "wasm32")]
fn main() {
    // Make sure panics are logged using `console.error`.
    console_error_panic_hook::set_once();

    // Redirect tracing to console.log and friends:
    tracing_wasm::set_as_global_default();

    let web_options = eframe::WebOptions::default();

    wasm_bindgen_futures::spawn_local(async {
        eframe::start_web(
            "the_canvas_id", // hardcode it
            web_options,
            Box::new(|cc| Box::new(subbles::TemplateApp::new(cc))),
        )
        .await
        .expect("failed to start eframe");
    });
}
