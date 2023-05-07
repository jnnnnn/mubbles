use std::sync::mpsc::{self, TryRecvError};

use crate::whisper::WhisperUpdate;

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)] // if we add new fields, give them default values when deserializing old state
pub struct MubblesApp {
    // Example stuff:
    text: String,

    // this how you opt-out of serialization of a member
    #[serde(skip)]
    recording: bool,

    #[serde(skip)]
    from_whisper: mpsc::Receiver<WhisperUpdate>,
}

impl Default for MubblesApp {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        crate::whisper::start_listening(tx);
        Self {
            text: "".to_owned(),
            recording: false,
            from_whisper: rx,
        }
    }
}

impl MubblesApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        if let Some(storage) = cc.storage {
            return eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
        }

        Default::default()
    }
}

impl eframe::App for MubblesApp {
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let Self {
            text,
            recording,
            from_whisper,
        } = self;
        let whisper_update_result = from_whisper.try_recv();
        match whisper_update_result {
            Ok(WhisperUpdate::Transcript(t)) => {
                text.push_str(&t);
            }
            Ok(WhisperUpdate::Recording(r)) => {
                *recording = r;
            }
            Err(TryRecvError::Empty) => {
                ();
            }
            Err(TryRecvError::Disconnected) => {
                panic!("Whisper channel disconnected");
            }
        }
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.checkbox(recording, "Recording");
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.text_edit_multiline(text);
        });
    }
}
