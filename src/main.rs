#![warn(clippy::all, rust_2018_idioms)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    // Log to stdout (if you run with `RUST_LOG=debug`).
    tracing_subscriber::fmt::init();

    let result = dosoundiotest();

    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "subbles",
        native_options,
        Box::new(|cc| Box::new(subbles::TemplateApp::new(cc))),
    )
}

fn dosoundiotest() -> soundio::Result {
    let ctx = soundio::Context::new();
    ctx.set_app_name("Subbles or Mubbles");
    ctx.connect()?;
    ctx.flush_events();

    let input_devices = ctx.input_devices()?;
    for dev in input_devices.iter() {
        println!("Input device: {}", dev.name());
    }

    let dev = ctx.default_input_device().expect("No input device");
    let mut input_stream = dev.open_instream(
        44100,
        soundio::Format::S16LE,
        soundio::ChannelLayout::get_builtin(soundio::ChannelLayoutId::Stereo),
        1.0,
        read_callback,
        None::<fn()>,
        None::<fn(soundio::Error)>,
    )?;

    input_stream.start()?;
}

fn read_callback(stream: &mut soundio::InStreamReader) {
    let frame_count_max = stream.frame_count_max();
    if let Err(e) = stream.begin_read(frame_count_max) {
        println!("Error reading from stream: {}", e);
        return;
    }

    let mut maxsample = 0;
    for f in 0..stream.frame_count() {
        for c in 0..stream.channel_count() {
            let sample = stream.sample::<i16>(c, f);
            if sample > maxsample {
                maxsample = sample;
            }
        }
    }
    println(
        "Callback, frame count: {}, max: {}",
        stream.frame_count(),
        maxsample,
    );
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
