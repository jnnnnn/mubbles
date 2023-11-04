use std::{
    collections::VecDeque,
    sync::mpsc::{self, TryRecvError},
    time::Duration,
};

use cpal::traits::{DeviceTrait, HostTrait};

use crate::whisper::{
    candle_example::WhichModel, get_devices, AppDevice, StreamState, WhisperParams, WhisperUpdate,
};

use crate::summary;

use egui_plot::{Line, Plot, PlotPoints};

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
    devices: Vec<AppDevice>,

    #[serde(skip)]
    selected_device: usize,

    #[serde(skip)]
    whisper_tx: mpsc::Sender<WhisperUpdate>,

    #[serde(skip)]
    level: VecDeque<f32>,

    autotype: bool,

    #[serde(skip)]
    always_on_top: bool,

    #[serde(skip)]
    changed: bool,

    #[serde(skip)]
    show_summary: bool,

    summary: summary::SummaryState,

    params: WhisperParams,
}

#[derive(Debug, PartialEq)]
struct DeviceOption {
    name: String,
}

impl Default for MubblesApp {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        let devices = get_devices();
        let host = cpal::default_host();
        let default_device_name = match host.default_output_device() {
            Some(d) => d.name().unwrap_or(String::from("Unknown")).to_owned(),
            None => "Unkown".to_owned(),
        };
        let selected_device = devices
            .iter()
            .position(|d| d.name == default_device_name)
            .expect("default device index error");

        Self {
            text: "".to_owned(),
            summary: summary::SummaryState::default(),
            recording: false,
            transcribing: false,
            from_whisper: rx,
            stream: crate::whisper::start_listening(
                &tx,
                &devices[selected_device],
                WhisperParams {
                    accuracy: 1,
                    model: WhichModel::TinyEn,
                    quantized: true,
                },
            ),
            devices: devices,
            selected_device: selected_device,
            whisper_tx: tx,
            level: VecDeque::with_capacity(100),
            autotype: false,
            always_on_top: false,
            changed: false,
            show_summary: false,
            params: WhisperParams {
                accuracy: 1,
                model: WhichModel::TinyEn,
                quantized: true,
            },
        }
    }
}

impl MubblesApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        tracing::info!("Startup at {}", chrono::Local::now());
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

static MODELS: [WhichModel; 12] = [
    WhichModel::Tiny,
    WhichModel::TinyEn,
    WhichModel::Base,
    WhichModel::BaseEn,
    WhichModel::Small,
    WhichModel::SmallEn,
    WhichModel::Medium,
    WhichModel::MediumEn,
    WhichModel::Large,
    WhichModel::LargeV2,
    WhichModel::DistilMediumEn,
    WhichModel::DistilLargeV2,
];

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
            params,
            always_on_top,
            changed,
            ..
        } = self;
        // drain from_whisper channel
        loop {
            let whisper_update_result = from_whisper.try_recv();
            match whisper_update_result {
                Ok(WhisperUpdate::Transcript(t)) => {
                    text.push_str(t.trim());
                    text.push_str("\n");
                    *changed = true;

                    // if autotype enabled and this window is in the background, send the text
                    let _focused = !frame.info().window_info.minimized;
                    if *autotype {
                        winput::send_str(&t);
                    }
                }
                Ok(WhisperUpdate::Recording(r)) => *recording = r,
                Ok(WhisperUpdate::Transcribing(t)) => *transcribing = t,
                Ok(WhisperUpdate::Level(l)) => {
                    if level.len() > 99 {
                        level.pop_front();
                    }
                    level.push_back(l);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => panic!("Whisper channel disconnected"),
            }
        }

        // eframe will go to sleep when data is waiting.. this is a hack to keep it awake.
        // it would be better for the channel to call this when it has posted data.
        ctx.request_repaint_after(Duration::from_millis(100));

        // Draw the UI
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.with_layout(
                egui::Layout::left_to_right(egui::Align::LEFT)
                    .with_main_wrap(true)
                    .with_cross_align(egui::Align::TOP),
                |ui| {
                    plot_level(level, ui);

                    let source = egui::ComboBox::from_label("Sound device").show_index(
                        ui,
                        selected_device,
                        devices.len(),
                        |i| devices[i].name.clone(),
                    );
                    if source.changed() {
                        let device = &devices[*selected_device];
                        *stream = crate::whisper::start_listening(whisper_tx, device, *params);
                    }
                },
            );
            ui.with_layout(
                egui::Layout::left_to_right(egui::Align::LEFT)
                    .with_main_wrap(true)
                    .with_cross_align(egui::Align::TOP),
                |ui| {
                    ui.add_enabled_ui(false, |ui| {
                        ui.checkbox(recording, "Recording");
                        ui.checkbox(transcribing, "Transcribing");
                    });

                    // choose a model, from WhichModel
                    let cb = egui::ComboBox::from_label("Model")
                        .selected_text(format!("{:?}", params.model))
                        .show_ui(ui, |ui| {
                            for model in MODELS.iter() {
                                ui.selectable_value(
                                    &mut params.model,
                                    *model,
                                    format!("{:?}", model),
                                );
                            }
                        });

                    if cb.response.changed {
                        *stream = crate::whisper::start_listening(
                            whisper_tx,
                            &devices[*selected_device],
                            *params,
                        );
                    }
                },
            );
            ui.with_layout(
                egui::Layout::left_to_right(egui::Align::LEFT)
                    .with_main_wrap(true)
                    .with_cross_align(egui::Align::TOP),
                |ui| {
                    ui.checkbox(autotype, "Autotype").on_hover_text(
                        "Type whatever is said into other applications on this computer",
                    );
                    // remove this for now because it's annoying
                    if ui.checkbox(always_on_top, "Always on top").changed() {
                        frame.set_always_on_top(*always_on_top);
                    }
                    if ui.button("Clear").clicked() {
                        text.clear();
                        // log the time as well as a message
                        tracing::info!("Cleared text, time: {}", chrono::Local::now());
                    }
                    if ui.button("Open Log").clicked() {
                        let logpath = // current exe directory + "mubbles.log":
                            std::env::current_exe()
                                .expect("Error getting current exe directory")
                                .parent()
                                .expect("Error getting parent directory")
                                .to_path_buf();
                        // don't add this because it breaks -- user can open the file themselves
                        //.join("mubbles.log");
                        if let Err(err) = open::that(logpath) {
                            tracing::error!("Error opening log: {}", err);
                        }
                    }
                },
            );
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            // tabs for either raw transcript or summary:
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.show_summary, false, "Transcript");
                ui.selectable_value(&mut self.show_summary, true, "Summary");
            });

            if self.show_summary {
                summary::summary_ui(&mut self.summary, ui, text);
            }

            let scroll_area = egui::ScrollArea::vertical();
            let scroll_area = if *changed {
                *changed = false;
                scroll_area.vertical_scroll_offset(10000000f32)
            } else {
                scroll_area
            };
            scroll_area.show(ui, |ui| {
                ui.add_sized(
                    ui.available_size(),
                    egui::TextEdit::multiline(if self.show_summary {
                        &mut self.summary.text
                    } else {
                        text
                    }),
                );
            });
        });
    }
}

fn plot_level(level: &VecDeque<f32>, ui: &mut egui::Ui) {
    let pairs: PlotPoints = level
        .iter()
        .enumerate()
        .map(|(i, v)| [i as f64, *v as f64])
        .collect();
    let line = Line::new(pairs);
    ui.add_enabled_ui(false, |ui| {
        Plot::new("my_plot")
            .width(100f32)
            .height(30f32)
            .include_y(0f32)
            .include_y(1f32)
            .view_aspect(2.0)
            .show(ui, |plot_ui| plot_ui.line(line));
    });
}
