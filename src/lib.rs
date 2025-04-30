// disable dead code warning for the whole file as I'm still working on summary
#![allow(dead_code)]
#![warn(clippy::all, rust_2018_idioms)]

mod app;
pub use app::MubblesApp;

mod whisper;

mod summary;

mod multilingual;