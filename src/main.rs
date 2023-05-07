#![warn(clippy::all, rust_2018_idioms)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::sync::mpsc::{self, Receiver, Sender};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::Resampler;

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

    loop {
        let data = rx.recv().expect("Failed to receive data");

        println!("Data, len: {}, max: {}", data.len(), data.iter().fold(0.0f32, |a, &b| a.max(b)));
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
