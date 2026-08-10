// disable dead code warning for the whole file as I'm still working on summary
#![allow(dead_code)]
#![warn(clippy::all, rust_2018_idioms)]

mod app;
pub use app::MubblesApp;
mod workers;

mod mel;
mod multilingual;
mod whisper;
mod whisper_model;
mod whisper_word_align;

mod audio;
#[cfg(target_os = "linux")]
mod audio_pipewire;
mod autotype;
mod file_transcription;
pub mod log_capture;
mod partial;
mod summary;
pub mod vad;
