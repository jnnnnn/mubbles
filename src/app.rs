use std::{
    collections::VecDeque,
    sync::mpsc::{self, TryRecvError},
    time::Duration,
};

use cpal::traits::{DeviceTrait, HostTrait};

use crate::whisper::{
    get_devices, AppDevice, DisplayMel, StreamState, WhichModel, WhisperParams, WhisperUpdate,
};

use crate::summary;

use egui_plot::{Line, Plot, PlotPoints};

#[derive(Debug, PartialEq)]
enum AppTab {
    Transcript,
    StatisticalSummary,
    AISummary,
    AIUserPrompt,
    AISystemPrompt,
}

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
    selected_model: usize,

    #[serde(skip)]
    whisper_tx: mpsc::Sender<WhisperUpdate>,

    #[serde(skip)]
    level: VecDeque<f32>,

    #[serde(skip)]
    mel_texture: Option<egui::TextureHandle>,

    autotype: bool,
    partials: bool,

    #[serde(skip)]
    always_on_top: bool,

    #[serde(skip)]
    changed: bool,

    #[serde(skip)]
    tab: AppTab,

    statistical_summary: summary::SummaryState,
    ai_summary: summary::SummaryState,

    accuracy: usize,
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
            statistical_summary: summary::SummaryState::default(),
            ai_summary: summary::SummaryState::default(),
            recording: false,
            transcribing: false,
            from_whisper: rx,
            stream: crate::whisper::start_listening(
                &tx,
                &devices[selected_device],
                WhisperParams {
                    accuracy: 1,
                    model: WhichModel::from(1),
                },
            ),
            devices: devices,
            selected_device: selected_device,
            selected_model: 1,
            whisper_tx: tx,
            level: VecDeque::with_capacity(100),
            mel_texture: None,
            autotype: false,
            partials: false,
            always_on_top: false,
            changed: false,
            tab: AppTab::Transcript,
            accuracy: 1,
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

impl eframe::App for MubblesApp {
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let Self {
            text,
            recording,
            transcribing,
            from_whisper,
            devices,
            selected_device,
            selected_model,
            stream,
            whisper_tx,
            level,
            autotype,
            partials,
            accuracy,
            changed,
            mel_texture,
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
                }
                Ok(WhisperUpdate::Recording(r)) => *recording = r,
                Ok(WhisperUpdate::Transcribing(t)) => *transcribing = t,
                Ok(WhisperUpdate::Level(l)) => {
                    if level.len() > 99 {
                        level.pop_front();
                    }
                    level.push_back(l);
                }
                Ok(WhisperUpdate::Mel(m)) => {
                    let color_image = mel_float_to_image(m);

                    *mel_texture = Some(ctx.load_texture(
                        "mel_spectrogram",
                        color_image,
                        egui::TextureOptions::default(),
                    ));
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
                    
                    plot_mel(mel_texture, ui);

                    let source = egui::ComboBox::from_label("Sound device").show_index(
                        ui,
                        selected_device,
                        devices.len(),
                        |i| devices[i].name.clone(),
                    );
                    if source.changed() {
                        let device = &devices[*selected_device];
                        *stream = crate::whisper::start_listening(
                            whisper_tx,
                            device,
                            WhisperParams {
                                accuracy: *accuracy,
                                model: WhichModel::from(*selected_model),
                            },
                        );
                    }
                    let model = egui::ComboBox::from_label("Model")
                        .selected_text(WhichModel::from(*selected_model).to_string())
                        .show_index(ui, selected_model, WhichModel::len(), |i| {
                            WhichModel::from(i).to_string()
                        });
                    if model.changed() {
                        *stream = crate::whisper::start_listening(
                            whisper_tx,
                            &devices[*selected_device],
                            WhisperParams {
                                accuracy: *accuracy,
                                model: WhichModel::from(*selected_model),
                            },
                        );
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

                    if ui
                        .add(egui::Slider::new(accuracy, 1..=8).text("Accuracy"))
                        .changed()
                    {
                        *stream = crate::whisper::start_listening(
                            whisper_tx,
                            &devices[*selected_device],
                            WhisperParams {
                                accuracy: *accuracy,
                                model: WhichModel::from(*selected_model),
                            },
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
                    ui.checkbox(partials, "Partials").on_hover_text(
                        "Show partials as a block is dictated, erasing with the full model once it is done",
                    );
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
                ui.selectable_value(&mut self.tab, AppTab::Transcript, "Transcript");
                ui.selectable_value(
                    &mut self.tab,
                    AppTab::StatisticalSummary,
                    "Statistical Summary",
                );
                ui.selectable_value(&mut self.tab, AppTab::AISummary, "AI Summary");
                ui.selectable_value(&mut self.tab, AppTab::AIUserPrompt, "AI User Prompt");
                ui.selectable_value(&mut self.tab, AppTab::AISystemPrompt, "AI System Prompt");
            });

            // add extra UI depending on which tab we're on
            match self.tab {
                AppTab::StatisticalSummary => {
                    summary::statistical_ui(&mut self.statistical_summary, ui, text)
                }
                AppTab::AISummary => summary::ai_ui(&mut self.ai_summary, ui, text),
                AppTab::AISystemPrompt => {}
                AppTab::AIUserPrompt => {}
                AppTab::Transcript => {}
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
                    egui::TextEdit::multiline(match self.tab {
                        AppTab::Transcript => text,
                        AppTab::StatisticalSummary => &mut self.statistical_summary.text,
                        AppTab::AISummary => &mut self.ai_summary.text,
                        AppTab::AIUserPrompt => &mut self.ai_summary.user_prompt,
                        AppTab::AISystemPrompt => &mut self.ai_summary.system_prompt,
                    }),
                );
            });
        });
    }
}

fn mel_float_to_image(m: DisplayMel) -> egui::ColorImage {
    // map -1..1 m.mel float vec to 0..255 u8 vec
    let min = -1.0;
    let max = 1.0;
    let bytes = m
        .mel
        .iter()
        .map(|&x| {
            let x = (x -min) * (255.0 / (max - min));
            if x < 0.0 {
                0
            } else if x > 255.0 {
                255
            } else {
                x as u8
            }
        })
        .collect::<Vec<u8>>();
    // Convert Mel data to ColorImage
    let color_image = egui::ColorImage::from_gray(
        [m.num_frames, m.num_bins], // Dimensions of the Mel spectrogram
        &bytes,                     // Raw RGB data
    );
    color_image
}

fn plot_level(level: &VecDeque<f32>, ui: &mut egui::Ui) {
    let pairs: PlotPoints<'_> = level
        .iter()
        .enumerate()
        .map(|(i, v)| [i as f64, *v as f64])
        .collect();
    let line = Line::new("line_name", pairs);
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

fn plot_mel(mel_texture: &Option<egui::TextureHandle>, ui: &mut egui::Ui) {
    if let Some(texture) = mel_texture {
        ui.add(
            egui::Image::new(texture)
                .corner_radius(10.0),
        );
    }
}
