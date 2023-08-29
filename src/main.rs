#![warn(clippy::all, rust_2018_idioms)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    set_up_tracing();

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

use tracing_subscriber::{prelude::*, Registry};

fn set_up_tracing() {
    let stdout_log = tracing_subscriber::fmt::layer().pretty();
    let subscriber = Registry::default().with(stdout_log);

    // keep one week of logs in daily files
    let file_appender = rolling_file::BasicRollingFileAppender::new(
        "./log.log",
        rolling_file::RollingConditionBasic::new()
            .daily()
            .max_size(1024 * 1024),
        10,
    )
    .expect("Couldn't open log file");

    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
    let file_layer = tracing_subscriber::fmt::layer().with_writer(non_blocking);

    let subscriber = subscriber.with(file_layer);

    tracing::subscriber::set_global_default(subscriber).expect("Unable to set global subscriber");
}
