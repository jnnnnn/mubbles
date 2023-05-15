use std::{
    sync::mpsc::{self, TryRecvError},
    time::Duration,
};

use cpal::{
    traits::{DeviceTrait, HostTrait},
    Device,
};

use crate::whisper::{StreamState, WhisperUpdate};

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)] // if we add new fields, give them default values when deserializing old state
pub struct MubblesApp {
    text: String,

    #[serde(skip)]
    recording: bool,

    #[serde(skip)]
    transcribing: bool,

    #[serde(skip)]
    from_whisper: mpsc::Receiver<WhisperUpdate>,

    #[serde(skip)]
    stream: Option<StreamState>,

    #[serde(skip)]
    devices: Vec<Device>,

    #[serde(skip)]
    selected_device: usize,

    #[serde(skip)]
    whisper_tx: mpsc::Sender<WhisperUpdate>,

    #[serde(skip)]
    level: f32,

    autotype: bool,

    always_on_top: bool,
}

#[derive(Debug, PartialEq)]
struct DeviceOption {
    name: String,
}

impl Default for MubblesApp {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        let host = cpal::default_host();
        let default_device = host
            .default_output_device()
            .expect("no default input device");
        let devices: Vec<Device> = host
            .output_devices()
            .expect("No input devices on default host")
            .collect();
        let selected_device = devices
            .iter()
            .position(|d| {
                d.name().expect("device name") == default_device.name().expect("device name")
            })
            .expect("default device index error");

        Self {
            text: "".to_owned(),
            recording: false,
            transcribing: false,
            from_whisper: rx,
            stream: crate::whisper::start_listening(&tx, &default_device),
            devices: devices,
            selected_device: selected_device,
            whisper_tx: tx,
            level: 0f32,
            autotype: false,
            always_on_top: false,
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
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let Self {
            text,
            recording,
            transcribing,
            from_whisper,
            devices,
            selected_device,
            stream,
            whisper_tx,
            level,
            autotype,
            ..
        } = self;
        // drain from_whisper channel
        loop {
            let whisper_update_result = from_whisper.try_recv();
            match whisper_update_result {
                Ok(WhisperUpdate::Transcript(t)) => {
                    text.push_str(&t.trim());
                    text.push_str("\n");
                    // if autotype enabled and this window is in the background, send the text
                    let _focused = !frame.info().window_info.minimized;
                    if *autotype {
                        winput::send_str(&t);
                    }
                }
                Ok(WhisperUpdate::Recording(r)) => *recording = r,
                Ok(WhisperUpdate::Transcribing(t)) => *transcribing = t,
                Ok(WhisperUpdate::Level(l)) => *level = l,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => panic!("Whisper channel disconnected"),
            }
        }

        // eframe will go to sleep when data is waiting.. this is a hack to keep it awake.
        // it would be better for the channel to call this when it has posted data.
        ctx.request_repaint_after(Duration::from_millis(100));

        // Draw the UI
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("Level: {:.2}", level));
                let source = egui::ComboBox::from_label("Sound device").show_index(
                    ui,
                    selected_device,
                    devices.len(),
                    |i| devices[i].name().expect("Device name"),
                );
                if source.changed() {
                    let device = &devices[*selected_device];
                    *stream = crate::whisper::start_listening(whisper_tx, device);
                }
                ui.add_enabled_ui(false, |ui| {
                    ui.checkbox(recording, "Recording");
                    ui.checkbox(transcribing, "Transcribing");
                });
                ui.checkbox(autotype, "Autotype").on_hover_text(
                    "Type whatever is said into other applications on this computer",
                );
                // remove this for now because it's annoying
                // if ui.checkbox(always_on_top, "Always on top").changed() {
                //     frame.set_always_on_top(*always_on_top);
                // }
                if ui.button("Clear").clicked() {
                    text.clear()
                }
            });
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.add_sized(ui.available_size(), egui::TextEdit::multiline(text));
            });
        });
    }
}
