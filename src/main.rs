#![warn(clippy::all, rust_2018_idioms)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    let _trace_state = set_up_tracing();

    let mut native_options = eframe::NativeOptions::default();
    native_options.icon_data = Some(load_icon());
    eframe::run_native(
        "mubbles",
        native_options,
        Box::new(|cc| Box::new(mubbles::MubblesApp::new(cc))),
    )
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
            Box::new(|cc| Box::new(mubbles::MubblesApp::new(cc))),
        )
        .await
        .expect("failed to start eframe");
    });
}

pub(crate) fn load_icon() -> eframe::IconData {
    let (icon_rgba, icon_width, icon_height) = {
        let icon = include_bytes!("../assets/icon-256.png");
        let image = image::load_from_memory(icon)
            .expect("Failed to open icon path")
            .into_rgba8();
        let (width, height) = image.dimensions();
        let rgba = image.into_raw();
        (rgba, width, height)
    };

    eframe::IconData {
        rgba: icon_rgba,
        width: icon_width,
        height: icon_height,
    }
}

use tracing_subscriber::prelude::*;

fn set_up_tracing() -> Box<dyn std::any::Any> {
    // keep ten days of logs in daily files up to 1MB
    let file_appender = rolling_file::BasicRollingFileAppender::new(
        "./mubbles.log",
        rolling_file::RollingConditionBasic::new()
            .daily()
            .max_size(1024 * 1024),
        10,
    )
    .expect("Couldn't open log file");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    let console_layer = tracing_subscriber::fmt::Layer::new()
        .with_writer(std::io::stdout.with_max_level(tracing::Level::INFO))
        .pretty();
    let file_layer = tracing_subscriber::fmt::Layer::new()
        .with_writer(non_blocking.with_max_level(tracing::Level::INFO))
        .with_ansi(false)
        .without_time();

    tracing::subscriber::set_global_default(
        tracing_subscriber::registry()
            .with(console_layer)
            .with(file_layer),
    )
    .expect("Couldn't set up tracing");

    Box::new(_guard)
}
