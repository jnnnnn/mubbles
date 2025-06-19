#![warn(clippy::all, rust_2018_idioms)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

fn main() -> eframe::Result<()> {
    let _trace_state = set_up_tracing();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([320.0, 240.0])
            .with_min_inner_size([300.0, 220.0])
            .with_drag_and_drop(true)
            .with_icon(
                eframe::icon_data::from_png_bytes(&include_bytes!("../assets/icon-256.png")[..])
                    .expect("Failed to load icon"),
            ),
        ..Default::default()
    };
    eframe::run_native(
        "mubbles",
        native_options,
        Box::new(|cc| Ok(Box::new(mubbles::MubblesApp::new(cc)))),
    )
}

use std::time::Instant;
use tracing::span::{Attributes, Id};
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::{prelude::*, EnvFilter};

fn set_up_tracing() -> Box<dyn std::any::Any> {
    // keep ten days of logs in daily files up to 1MB
    let file_appender = rolling_file::BasicRollingFileAppender::new(
        "./mubbles.log",
        rolling_file::RollingConditionBasic::new()
            .daily()
            .max_size(1024 * 1024 * 10),
        3,
    )
    .expect("Couldn't open log file");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    let console_layer = tracing_subscriber::fmt::Layer::new()
        .pretty()
        .with_writer(std::io::stdout.with_max_level(tracing::Level::WARN))
        .with_filter(EnvFilter::from_default_env());
    let file_layer = tracing_subscriber::fmt::Layer::new()
        .with_writer(non_blocking)
        .with_ansi(false)
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
        //.without_time()
        .with_filter(EnvFilter::from_default_env())
        ;

    // use RUST_LOG="warn,mubbles=trace" to see app tracing
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry()
            .with(SpanTimingLayer)
            .with(console_layer)
            .with(file_layer)
    )
    .expect("Couldn't set up tracing");

    Box::new(_guard)
}



struct SpanTiming {
    started_at: Instant,
}

pub struct SpanTimingLayer;

impl<S> Layer<S> for SpanTimingLayer
where
    S: Subscriber,
    S: for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, _attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let span = ctx.span(id).unwrap();

        span.extensions_mut().insert(SpanTiming {
            started_at: Instant::now(),
        });
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let span = ctx.span(&id).unwrap();
        let started_at = span.extensions().get::<SpanTiming>().unwrap().started_at;
        let elapsed = (Instant::now() - started_at).as_millis();
    }
}