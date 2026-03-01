use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Sender, TryRecvError},
        Arc,
    },
    thread::JoinHandle,
    time::Duration,
};

use candle_core::Tensor;
use egui_plot::{Line, Plot, PlotPoints};
use egui_extras::{Column, TableBuilder};

use crate::{
    audio::{check_device_enumeration, get_devices, AppDevice, DeviceEnumerationReceiver, StreamState},
    file_transcription::{self, FileUpdate},
    partial::PARTIAL_MEL_BINS,
    summary,
    whisper::{WhichModel, WhisperParams},
    whisper_word_align::AlignedWord,
};

// UI constants
const LEVEL_PLOT_WIDTH: f32 = 100.0;
const LEVEL_PLOT_HEIGHT: f32 = 30.0;
const LEVEL_BUFFER_SIZE: usize = 100;
const MEL_UPDATE_HZ: f32 = 100.0;  // Mel frames per second (10ms per frame)
const REPAINT_INTERVAL_MS: u64 = 100;
const ALIGNED_WORD_ROWS: usize = 6;
const ALIGNED_WORD_ROW_HEIGHT: f32 = 12.0;
const WORD_CHAR_WIDTH: f32 = 7.0;

/// Application tab selection
#[derive(Debug, PartialEq, serde::Deserialize, serde::Serialize)]
enum AppTab {
    Transcript,
    TranscriptionHistory,
    StatisticalSummary,
    AISummary,
    FileTranscription,
    Settings,
    Log,
}

// Maximum mel frames to display (5 seconds at 100 Hz)
const MAX_MEL_FRAMES: usize = 500;

/// Mel spectrogram display state with pre-allocated texture
struct DisplayMel {
    texture: Option<egui::TextureHandle>,
    frame_count: usize, // number of frames populated in texture
    min: f32,
    max: f32,
}

impl DisplayMel {
    fn new() -> Self {
        Self {
            texture: None,
            frame_count: 0,
            min: -10.0,
            max: 0.0,
        }
    }

    fn update_range(&mut self, frame: &[f32]) {
        self.min = self
            .min
            .min(frame.iter().copied().fold(f32::INFINITY, f32::min));
        self.max = self
            .max
            .max(frame.iter().copied().fold(f32::NEG_INFINITY, f32::max))
            + 0.01;
    }

    /// Convert mel frame values to grayscale bytes
    fn frame_to_bytes(&self, frame: &[f32]) -> Vec<u8> {
        frame
            .iter()
            .map(|&x| {
                let normalized = (x - self.min) * (255.0 / (self.max - self.min));
                normalized.clamp(0.0, 255.0) as u8
            })
            .collect()
    }

    /// Push a new mel frame to the texture using incremental update
    fn push_frame(&mut self, frame: &[f32], ctx: &egui::Context) {
        if self.frame_count >= MAX_MEL_FRAMES {
            return; // Don't overflow pre-allocated texture
        }

        self.update_range(frame);
        let bytes = self.frame_to_bytes(frame);

        // Create column image for just this frame (1 pixel wide, PARTIAL_MEL_BINS tall)
        let column = egui::ColorImage::from_gray([1, PARTIAL_MEL_BINS], &bytes);

        if let Some(tex) = &mut self.texture {
            // Update column in pre-allocated texture
            tex.set_partial([self.frame_count, 0], column, egui::TextureOptions::default());
            self.frame_count += 1;
        } else {
            // Create pre-allocated texture (black/silent initially)
            let empty = egui::ColorImage::new(
                [MAX_MEL_FRAMES, PARTIAL_MEL_BINS],
                vec![egui::Color32::BLACK; MAX_MEL_FRAMES * PARTIAL_MEL_BINS],
            );
            let mut tex = ctx.load_texture("mel_partial", empty, egui::TextureOptions::default());
            // Set the first column
            tex.set_partial([0, 0], column, egui::TextureOptions::default());
            self.texture = Some(tex);
            self.frame_count = 1;
        }
    }

    /// Reset mel display (clear texture and frame count)
    fn reset(&mut self) {
        // Clear texture to black, keep the allocation
        if let Some(tex) = &mut self.texture {
            let empty = egui::ColorImage::new(
                [MAX_MEL_FRAMES, PARTIAL_MEL_BINS],
                vec![egui::Color32::BLACK; MAX_MEL_FRAMES * PARTIAL_MEL_BINS],
            );
            tex.set(empty, egui::TextureOptions::default());
        }
        self.frame_count = 0;
        self.min = -10.0;
        self.max = 0.0;
    }
}

/// Main application state
///
/// Deserialize/Serialize enables persistence on shutdown
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct MubblesApp {
    // Transcript state
    text: String,
    #[serde(skip)]
    aligned_words: Vec<AlignedWord>,
    #[serde(skip)]
    word_history: Vec<Vec<AlignedWord>>,

    // Recording state
    #[serde(skip)]
    recording: bool,
    #[serde(skip)]
    transcribing: bool,
    #[serde(skip)]
    status: String,

    // Audio devices
    #[serde(skip)]
    devices: Vec<AppDevice>,
    #[serde(skip)]
    device_enumeration_rx: DeviceEnumerationReceiver,
    selected_device1: usize,
    selected_device2: usize,

    // Model configuration
    selected_model: usize,
    accuracy: usize,

    // Feature flags
    autotype: bool,
    partials: bool,
    #[serde(skip)]
    always_on_top: bool,

    // Worker threads
    #[serde(skip)]
    worker: Option<Worker>,
    #[serde(skip)]
    from_whisper: mpsc::Receiver<WhisperUpdate>,
    #[serde(skip)]
    whisper_tx: mpsc::Sender<WhisperUpdate>,

    // Visualization state
    #[serde(skip)]
    level: VecDeque<f32>,
    #[serde(skip)]
    mel: DisplayMel,

    // UI state
    #[serde(skip)]
    changed: bool,
    #[serde(skip)]
    tab: AppTab,

    // Summary state
    statistical_summary: summary::SummaryState,
    ai_summary: summary::SummaryState,

    // File transcription state
    #[serde(skip)]
    file_transcription_running: bool,
    #[serde(skip)]
    file_transcription_thread: Option<JoinHandle<()>>,
    #[serde(skip)]
    file_tx: mpsc::Sender<FileUpdate>,
    #[serde(skip)]
    file_rx: mpsc::Receiver<FileUpdate>,
    #[serde(skip)]
    file_transcription_text: String,
    #[serde(skip)]
    file_transcription_progress: Option<(usize, usize)>,
    #[serde(skip)]
    file_transcription_status: String,
    #[serde(skip)]
    file_cancel: Arc<AtomicBool>,

    // Logging state
    #[serde(skip)]
    log_buffer: crate::log_capture::LogBuffer,
    log_level: usize, // 0=TRACE, 1=DEBUG, 2=INFO, 3=WARN, 4=ERROR

    transcription_folder: String,
    #[serde(skip)]
    transcription_logger: crate::log_capture::TranscriptionLogger,
}

impl Default for MubblesApp {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        let (devices, device_enumeration_rx) = get_devices();
        let (file_tx, file_rx) = mpsc::channel();

        let selected_device = Self::find_default_device(&devices);

        Self {
            // Transcript state
            text: String::new(),
            aligned_words: vec![],
            word_history: Vec::new(),

            // Recording state
            recording: false,
            transcribing: false,
            status: "Init".to_owned(),

            // Audio devices
            devices,
            device_enumeration_rx,
            selected_device1: selected_device,
            selected_device2: selected_device,

            // Model configuration
            selected_model: 1,
            accuracy: 1,

            // Feature flags
            autotype: false,
            partials: false,
            always_on_top: false,

            // Worker threads
            worker: None,
            from_whisper: rx,
            whisper_tx: tx,

            // Visualization state
            level: VecDeque::with_capacity(LEVEL_BUFFER_SIZE),
            mel: DisplayMel::new(),

            // UI state
            changed: false,
            tab: AppTab::Transcript,

            // Summary state
            statistical_summary: summary::SummaryState::default(),
            ai_summary: summary::SummaryState::default(),

            // File transcription state
            file_transcription_running: false,
            file_transcription_thread: None,
            file_tx,
            file_rx,
            file_transcription_text: String::new(),
            file_transcription_progress: None,
            file_transcription_status: String::new(),
            file_cancel: Arc::new(AtomicBool::new(false)),

            // Logging state
            log_buffer: crate::log_capture::LogBuffer::new(),
            log_level: 2, // Default to INFO

            // Monthly transcription file settings
            transcription_folder: String::new(),
            transcription_logger: crate::log_capture::TranscriptionLogger::new(),
        }
    }
}

impl MubblesApp {
    /// Find the default device index in the device list.
    /// Returns the first device if no preferred default is found.
    fn find_default_device(devices: &[AppDevice]) -> usize {
        // On Linux with PipeWire, prefer monitor sources for desktop audio capture
        // On other platforms, try to find the default output device
        #[cfg(target_os = "linux")]
        {
            // Prefer first monitor source, or first device if none
            devices.iter().position(|d| d.name.contains("Monitor")).unwrap_or(0)
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            use cpal::traits::{DeviceTrait, HostTrait};
            let host = cpal::default_host();
            let default_device_name = host
                .default_output_device()
                .and_then(|d| d.name().ok())
                .unwrap_or_else(|| "Unknown".to_owned());

            devices
                .iter()
                .position(|d| d.name == default_device_name)
                .unwrap_or(0)
        }
    }

    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        tracing::info!("Startup at {}", chrono::Local::now());

        // Load previous app state (if any).
        cc.storage
            .and_then(|storage| eframe::get_value(storage, eframe::APP_KEY))
            .unwrap_or_default()
    }

    /// Get a clone of the log buffer for passing to the tracing setup
    pub fn get_log_buffer(&self) -> crate::log_capture::LogBuffer {
        self.log_buffer.clone()
    }

    /// Process updates from the Whisper thread
    fn process_whisper_updates(&mut self, ctx: &egui::Context) {
        loop {
            let update = match self.from_whisper.try_recv() {
                Ok(update) => update,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    panic!("Whisper channel disconnected")
                }
            };

            let span = tracing::span!(tracing::Level::TRACE, "whisper_update", ?update);
            let _enter = span.enter();

            match update {
                WhisperUpdate::Transcription(t) => {
                    let trimmed = t.trim().to_string();
                    self.text.push_str(&trimmed);
                    self.text.push('\n');
                    self.changed = true;
                    self.transcription_logger.append(&self.transcription_folder, &trimmed);
                    if self.autotype {
                        crate::autotype::type_text(&trimmed);
                    }
                    // Reset mel display when final transcription arrives
                    self.mel.reset();
                    self.aligned_words.clear();
                }
                WhisperUpdate::Recording(r) => self.recording = r,
                WhisperUpdate::Transcribing(t) => self.transcribing = t,
                WhisperUpdate::Level(l) => {
                    if self.level.len() >= LEVEL_BUFFER_SIZE {
                        self.level.pop_front();
                    }
                    self.level.push_back(l);
                }
                WhisperUpdate::Alignment(a) => {
                    self.word_history.push(a.clone());
                    self.aligned_words = a; // Complete replacement
                }
                WhisperUpdate::MelFrame(frame) => {
                    if self.partials {
                        self.mel.push_frame(&frame, ctx);
                    }
                }
                WhisperUpdate::Mel(_m) => {
                    // Ignored: we use incremental MelFrame updates instead
                }
                WhisperUpdate::Status(s) => {
                    self.status = s;
                }
            }
        }
    }

    /// Process updates from the file transcription thread
    fn process_file_updates(&mut self) {
        while let Ok(update) = self.file_rx.try_recv() {
            match update {
                FileUpdate::Status(s) => {
                    self.file_transcription_status = s;
                }
                FileUpdate::Progress { chunk, total } => {
                    self.file_transcription_progress = Some((chunk, total));
                    self.file_transcription_status =
                        format!("Transcribing chunk {}/{}...", chunk, total);
                }
                FileUpdate::Transcription(t) => {
                    let trimmed = t.trim();
                    if !trimmed.is_empty() {
                        if !self.file_transcription_text.is_empty() {
                            self.file_transcription_text.push('\n');
                        }
                        self.file_transcription_text.push_str(trimmed);
                    }
                }
                FileUpdate::Error(msg) => {
                    self.file_transcription_running = false;
                    self.file_transcription_status = format!("Error: {}", msg);
                    self.file_transcription_progress = None;
                }
                FileUpdate::Complete => {
                    self.file_transcription_running = false;
                    self.file_transcription_status = "File transcription complete".to_string();
                    self.file_transcription_progress = None;
                }
            }
        }
    }

    /// Check if async device enumeration has completed and update device list
    fn check_device_enumeration(&mut self) {
        if let Some(new_devices) = check_device_enumeration(&mut self.device_enumeration_rx) {
            tracing::info!("Device enumeration complete, found {} devices", new_devices.len());
            
            // Preserve selection if possible, otherwise find default
            let old_device1_name = self.devices.get(self.selected_device1).map(|d| d.name.clone());
            let old_device2_name = self.devices.get(self.selected_device2).map(|d| d.name.clone());
            
            self.devices = new_devices;
            
            // Try to restore previous selection by name
            self.selected_device1 = old_device1_name
                .and_then(|name| self.devices.iter().position(|d| d.name == name))
                .unwrap_or_else(|| Self::find_default_device(&self.devices));
            
            self.selected_device2 = old_device2_name
                .and_then(|name| self.devices.iter().position(|d| d.name == name))
                .unwrap_or_else(|| Self::find_default_device(&self.devices));
            
            self.status = format!("Found {} audio devices", self.devices.len());
        }
    }

    /// Render the top panel with controls
    fn render_top_panel(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            self.render_control_row(ui);
            self.render_status_row(ui);
            let mel_response = self.render_mel_row(ui);
            self.render_aligned_words(ctx, ui, mel_response);
            self.render_options_row(ui);
        });
    }

    /// Render the main control buttons and device selectors
    fn render_control_row(&mut self, ui: &mut egui::Ui) {
        ui.with_layout(
            egui::Layout::left_to_right(egui::Align::LEFT)
                .with_main_wrap(true)
                .with_cross_align(egui::Align::TOP),
            |ui| {
                plot_level(&self.level, ui);

                // Start/Stop button
                let started = self.worker.is_some();
                let button_text = if started { "Stop" } else { "Start" };
                if ui.button(button_text).clicked() {
                    self.toggle_recording();
                }

                // Device selectors (clamp to valid range in case persisted value is stale)
                if !self.devices.is_empty() {
                    self.selected_device1 = self.selected_device1.min(self.devices.len() - 1);
                    self.selected_device2 = self.selected_device2.min(self.devices.len() - 1);
                }

                ui.vertical(|ui| {
                    let source1 = egui::ComboBox::from_label("Sound device").show_index(
                        ui,
                        &mut self.selected_device1,
                        self.devices.len(),
                        |i| self.devices[i].name.clone(),
                    );
                    if source1.changed() {
                        self.worker = None;
                    }

                    let source2 = egui::ComboBox::from_label("Sound device 2").show_index(
                        ui,
                        &mut self.selected_device2,
                        self.devices.len(),
                        |i| self.devices[i].name.clone(),
                    );
                    if source2.changed() {
                        self.worker = None;
                    }
                });
            },
        );

        ui.with_layout(
            egui::Layout::left_to_right(egui::Align::LEFT).with_cross_align(egui::Align::TOP),
            |ui| {
                // Model selector
                let model = egui::ComboBox::from_label("Model")
                    .selected_text(WhichModel::from(self.selected_model).to_string())
                    .show_index(ui, &mut self.selected_model, WhichModel::len(), |i| {
                        WhichModel::from(i).to_string()
                    });
                if model.changed() {
                    self.worker = None;
                }
            },
        );
    }

    /// Toggle recording on/off
    fn toggle_recording(&mut self) {
        if self.worker.is_some() {
            self.worker = None;
        } else {
            match start_listening(
                &self.whisper_tx,
                &self.devices[self.selected_device1],
                &self.devices[self.selected_device2],
                WhisperParams {
                    accuracy: self.accuracy,
                    model: WhichModel::from(self.selected_model),
                    partials: self.partials,
                },
            ) {
                Ok(new_worker) => {
                    self.worker = Some(new_worker);
                }
                Err(e) => {
                    tracing::error!("Failed to start listening: {}", e);
                    self.worker = None;
                }
            }
        }
    }

    /// Render the status line
    fn render_status_row(&self, ui: &mut egui::Ui) {
        ui.label(format!("Status: {}", self.status));
    }

    /// Render the mel spectrogram (only when partials enabled)
    fn render_mel_row(&mut self, ui: &mut egui::Ui) -> Option<egui::InnerResponse<()>> {
        if !self.partials {
            return None; // Don't render mel when partials disabled
        }

        Some(ui.with_layout(
            egui::Layout::left_to_right(egui::Align::LEFT)
                .with_main_wrap(true)
                .with_cross_align(egui::Align::TOP),
            |ui| {
                draw_mel(&mut self.mel, ui);
            },
        ))
    }

    /// Render aligned words overlay on mel spectrogram
    fn render_aligned_words(
        &mut self,
        ctx: &egui::Context,
        ui: &mut egui::Ui,
        mel_response: Option<egui::InnerResponse<()>>,
    ) {
        if let Some(response) = mel_response {
            draw_aligned_words(ctx, &mut self.aligned_words, ui, response, &self.mel);
        }
    }

    /// Render options checkboxes and buttons
    fn render_options_row(&mut self, ui: &mut egui::Ui) {
        ui.with_layout(
            egui::Layout::left_to_right(egui::Align::LEFT)
                .with_main_wrap(true)
                .with_cross_align(egui::Align::TOP),
            |ui| {
                ui.add_enabled_ui(false, |ui| {
                    ui.checkbox(&mut self.recording, "Recording");
                    ui.checkbox(&mut self.transcribing, "Transcribing");
                });

                if ui
                    .add(egui::Slider::new(&mut self.accuracy, 1..=8).text("Accuracy"))
                    .changed()
                {
                    self.worker = None;
                }
            },
        );

        ui.with_layout(
            egui::Layout::left_to_right(egui::Align::LEFT)
                .with_main_wrap(true)
                .with_cross_align(egui::Align::TOP),
            |ui| {
                ui.checkbox(&mut self.autotype, "Autotype")
                    .on_hover_text("Type whatever is said into other applications on this computer");

                let partials_changed = ui
                    .checkbox(&mut self.partials, "Partials")
                    .on_hover_text(
                        "Show partials as a block is dictated, erasing with the full model once it is done",
                    )
                    .changed();

                if partials_changed {
                    self.worker = None;
                }

                if ui.button("Clear").clicked() {
                    self.clear_transcript();
                }

                if ui.button("Open Log").clicked() {
                    self.open_log_directory();
                }
            },
        );
    }

    /// Clear the transcript and reset mel spectrogram
    fn clear_transcript(&mut self) {
        self.text.clear();
        tracing::info!("Cleared text at: {}", chrono::Local::now());
        self.mel.reset();
        self.aligned_words.clear();
    }

    /// Open the log directory
    fn open_log_directory(&self) {
        let log_path = std::env::current_exe()
            .expect("Error getting current exe directory")
            .parent()
            .expect("Error getting parent directory")
            .to_path_buf();

        if let Err(err) = open::that(log_path) {
            tracing::error!("Error opening log: {}", err);
        }
    }

    /// Start file transcription
    fn start_file_transcription(&mut self) {
        let file_dialog = rfd::FileDialog::new()
            .add_filter("Audio Files", &["wav", "mp3", "flac", "ogg", "m4a", "aac", "wma"])
            .set_title("Select Audio File to Transcribe");

        if let Some(path) = file_dialog.pick_file() {
            let tx = self.file_tx.clone();
            let model = WhichModel::from(self.selected_model);
            let cancel = self.file_cancel.clone();

            self.file_transcription_running = true;
            self.file_transcription_text.clear();
            self.file_transcription_progress = None;
            self.file_transcription_status = "Starting...".to_string();
            self.file_cancel.store(false, Ordering::Relaxed);
            self.tab = AppTab::FileTranscription;

            let thread = std::thread::spawn(move || {
                let tx2 = tx.clone();
                if let Err(e) = file_transcription::transcribe_file(path, model, tx, cancel) {
                    tracing::error!("File transcription failed: {}", e);
                    tx2.send(FileUpdate::Error(e.to_string())).ok();
                }
            });

            self.file_transcription_thread = Some(thread);
        }
    }

    /// Render the central panel with tabs and text editor
    fn render_central_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_tabs(ui);
            self.render_text_editor(ui);
        });
    }

    /// Render the tab bar
    fn render_tabs(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.tab, AppTab::Transcript, "Transcript");
            ui.selectable_value(&mut self.tab, AppTab::TranscriptionHistory, "History");
            ui.selectable_value(
                &mut self.tab,
                AppTab::StatisticalSummary,
                "Statistical Summary",
            );
            ui.selectable_value(&mut self.tab, AppTab::AISummary, "AI Summary");
            ui.selectable_value(&mut self.tab, AppTab::FileTranscription, "File");
            ui.selectable_value(&mut self.tab, AppTab::Settings, "Settings");
            ui.selectable_value(&mut self.tab, AppTab::Log, "Log");
        });

        // Tab-specific UI
        match self.tab {
            AppTab::StatisticalSummary => {
                summary::statistical_ui(&mut self.statistical_summary, ui, &mut self.text)
            }
            AppTab::AISummary => summary::ai_ui(&mut self.ai_summary, ui, &mut self.text),
            AppTab::Settings => {
                // Settings UI will be rendered in render_text_editor
            }
            _ => {}
        }
    }

    /// Render the text editor for the selected tab
    fn render_text_editor(&mut self, ui: &mut egui::Ui) {
        let scroll_area = egui::ScrollArea::vertical();
        let scroll_area = if self.changed {
            self.changed = false;
            scroll_area.vertical_scroll_offset(10_000_000.0)
        } else {
            scroll_area
        };

        scroll_area.show(ui, |ui| match self.tab {
            AppTab::Transcript => {
                ui.add_sized(
                    ui.available_size(),
                    egui::TextEdit::multiline(&mut self.text),
                );
            }
            AppTab::TranscriptionHistory => {
                self.render_transcription_history(ui);
            }
            AppTab::StatisticalSummary => {
                ui.add_sized(
                    ui.available_size(),
                    egui::TextEdit::multiline(&mut self.statistical_summary.text),
                );
            }
            AppTab::AISummary => {
                ui.add_sized(
                    ui.available_size(),
                    egui::TextEdit::multiline(&mut self.ai_summary.text),
                );
            }
            AppTab::Settings => {
                ui.heading("AI Provider");
                ui.add_space(10.0);

                ui.horizontal(|ui| {
                    ui.label("Provider:");
                    let prev_provider = self.ai_summary.provider.clone();
                    egui::ComboBox::from_id_salt("ai_provider")
                        .selected_text(format!("{}", self.ai_summary.provider))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.ai_summary.provider,
                                summary::ApiProvider::OpenAI,
                                "OpenAI",
                            );
                            ui.selectable_value(
                                &mut self.ai_summary.provider,
                                summary::ApiProvider::Ollama,
                                "Ollama (local)",
                            );
                            ui.selectable_value(
                                &mut self.ai_summary.provider,
                                summary::ApiProvider::Custom,
                                "Custom (any OpenAI-compatible API)",
                            );
                        });
                    // When provider changes, update URL and model to defaults
                    if self.ai_summary.provider != prev_provider {
                        self.ai_summary.api_url =
                            self.ai_summary.provider.default_url().to_string();
                        self.ai_summary.model =
                            self.ai_summary.provider.default_model().to_string();
                    }
                });

                ui.add_space(5.0);

                ui.horizontal(|ui| {
                    ui.label("API URL:");
                    ui.add_sized(
                        egui::vec2(ui.available_width(), 20.0),
                        egui::TextEdit::singleline(&mut self.ai_summary.api_url)
                            .hint_text("https://api.openai.com/v1/chat/completions"),
                    );
                });

                ui.horizontal(|ui| {
                    ui.label("Model:");
                    ui.add_sized(
                        egui::vec2(ui.available_width(), 20.0),
                        egui::TextEdit::singleline(&mut self.ai_summary.model)
                            .hint_text("gpt-4o-mini"),
                    );
                });

                if self.ai_summary.provider.needs_key() {
                    ui.horizontal(|ui| {
                        ui.label("API Key:");
                        ui.add_sized(
                            egui::vec2(ui.available_width(), 20.0),
                            egui::TextEdit::singleline(&mut self.ai_summary.api_key)
                                .password(true)
                                .hint_text("sk-..."),
                        );
                    });
                }

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(10.0);

                ui.heading("AI Prompts");
                ui.add_space(10.0);

                ui.label("System Prompt:");
                ui.add_sized(
                    egui::vec2(ui.available_width(), 100.0),
                    egui::TextEdit::multiline(&mut self.ai_summary.system_prompt),
                );

                ui.add_space(10.0);

                ui.label("User Prompt (use %SOFAR% and %ADDITIONAL% as placeholders):");
                ui.add_sized(
                    egui::vec2(ui.available_width(), 100.0),
                    egui::TextEdit::multiline(&mut self.ai_summary.user_prompt),
                );

                ui.add_space(20.0);
                ui.separator();
                ui.add_space(10.0);

                ui.heading("Monthly Transcription Log");
                ui.add_space(10.0);

                ui.horizontal(|ui| {
                    ui.label("Folder:");
                    ui.add_sized(
                        egui::vec2(ui.available_width() - 80.0, 20.0),
                        egui::TextEdit::singleline(&mut self.transcription_folder)
                            .hint_text("e.g., ~/transcriptions"),
                    );
                    if ui.button("Browse...").clicked() {
                        if let Some(path) = rfd::FileDialog::new().pick_folder() {
                            self.transcription_folder = path.display().to_string();
                        }
                    }
                });

                ui.label("Transcriptions will be appended to monthly files (e.g., 2025-12.txt) in this folder.");
                if !self.transcription_folder.is_empty() {
                    let now = chrono::Local::now();
                    let filename = format!("{}.txt", now.format("%Y-%m"));
                    ui.label(format!("Current file: {}/{}", self.transcription_folder, filename));
                }
            }
            AppTab::FileTranscription => {
                ui.heading("File Transcription");
                ui.add_space(10.0);

                ui.horizontal(|ui| {
                    if self.file_transcription_running {
                        if ui.button("Cancel").clicked() {
                            self.file_cancel.store(true, Ordering::Relaxed);
                        }
                        ui.spinner();
                    } else if ui.button("Select audio File...").clicked() {
                        self.start_file_transcription();
                    }
                });

                // Status line
                if !self.file_transcription_status.is_empty() {
                    ui.add_space(5.0);
                    ui.label(&self.file_transcription_status);
                }

                // Progress bar
                if let Some((chunk, total)) = self.file_transcription_progress {
                    ui.add_space(5.0);
                    let fraction = chunk as f32 / total as f32;
                    ui.add(
                        egui::ProgressBar::new(fraction)
                            .text(format!("{}/{}", chunk, total))
                            .animate(self.file_transcription_running),
                    );
                }

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(5.0);

                // Output text area
                if !self.file_transcription_text.is_empty() {
                    ui.horizontal(|ui| {
                        if ui.button("Copy to Clipboard").clicked() {
                            ui.ctx().copy_text(self.file_transcription_text.clone());
                        }
                        if ui.button("Clear").clicked() {
                            self.file_transcription_text.clear();
                        }
                    });
                    ui.add_space(5.0);
                }

                ui.add_sized(
                    ui.available_size(),
                    egui::TextEdit::multiline(&mut self.file_transcription_text.as_str())
                        .desired_width(f32::INFINITY),
                );
            }
            AppTab::Log => {
                ui.heading("Application Logs");
                ui.add_space(10.0);

                ui.horizontal(|ui| {
                    ui.label("Log Level:");
                    let level_names = ["TRACE", "DEBUG", "INFO", "WARN", "ERROR"];
                    let old_level = self.log_level;
                    egui::ComboBox::from_id_salt("log_level")
                        .selected_text(level_names[self.log_level])
                        .show_index(ui, &mut self.log_level, 5, |i| level_names[i]);
                    
                    // Update the level control if the level changed
                    if old_level != self.log_level {
                        let tracing_level = crate::log_capture::index_to_tracing_level(self.log_level);
                        self.log_buffer.level_control().set_level(tracing_level);
                    }

                    if ui.button("Clear Logs").clicked() {
                        self.log_buffer.clear();
                    }
                });

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(5.0);

                let logs = self.log_buffer.get_all();
                let log_text = logs
                    .iter()
                    .map(|entry| entry.format())
                    .collect::<Vec<_>>()
                    .join("\n");

                ui.add_sized(
                    ui.available_size(),
                    egui::TextEdit::multiline(&mut log_text.as_str())
                        .font(egui::TextStyle::Monospace)
                        .desired_width(f32::INFINITY),
                );
            }
        });
    }

    /// Render the transcription history tab with a virtualized table of words
    fn render_transcription_history(&self, ui: &mut egui::Ui) {
        if self.word_history.is_empty() {
            ui.label("No transcriptions yet. Start recording to see transcriptions here.");
            return;
        }

        ui.horizontal(|ui| {
            ui.label(format!("{} rows", self.word_history.len()));
        });

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(5.0);

        let row_height = 18.0;
        let num_rows = self.word_history.len();

        TableBuilder::new(ui)
            .striped(true)
            .resizable(true)
            .column(Column::auto().at_least(70.0)) // Timestamp
            .column(Column::auto().at_least(50.0)) // Probability
            .column(Column::remainder().at_least(100.0)) // Word
            .cell_layout(egui::Layout::left_to_right(egui::Align::LEFT))
            .header(20.0, |mut header| {
                header.col(|ui| { ui.strong("Time"); });
                header.col(|ui| { ui.strong("Words"); });
            })
            .body(|body| {
                body.rows(row_height, num_rows, |mut row| {
                    let idx = row.index();
                    let words = &self.word_history[idx];
                    
                    row.col(|ui| {
                        ui.label("00:00");
                    });
                    row.col(|ui| {
                        for word in words {
                            // Skip special tokens like <|startoftranscript|>, <|en|>, etc.
                            if word.word.starts_with("<|") && word.word.ends_with("|>") {
                                continue;
                            }
                            let color = probability_to_color(word.probability);
                            ui.label(egui::RichText::new(&word.word).color(color));
                        }
                    });
                });
            });
    }
}

/// we want low-confidence words to be red
fn probability_to_color(probability: f32) -> egui::Color32 {
    let t = ((probability - 0.5) * 2.0).clamp(0.0, 1.0);
    let green = (t * 255.0) as u8;
    let blue = (t * 255.0) as u8;
    egui::Color32::from_rgb(255, green, blue)
}

/// Updates from the Whisper processing thread
#[derive(Debug, Clone)]
pub enum WhisperUpdate {
    Recording(bool),
    Transcribing(bool),
    Transcription(String),
    Alignment(Vec<AlignedWord>),
    Level(f32),
    Mel(Tensor),
    Status(String),
    MelFrame(Vec<f32>),
}

impl eframe::App for MubblesApp {
    /// Called by the framework to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Keep the UI responsive with periodic repaints
        ctx.request_repaint_after(Duration::from_millis(REPAINT_INTERVAL_MS));

        // Check for thread panics
        if let Some(worker) = &mut self.worker {
            check_thread_error(&mut worker.whisper_thread);
        }
        check_thread_error(&mut self.file_transcription_thread);

        // Check if device enumeration completed
        self.check_device_enumeration();

        // Process updates from Whisper thread
        self.process_whisper_updates(ctx);

        // Process updates from file transcription thread
        self.process_file_updates();

        // Render UI
        self.render_top_panel(ctx);
        self.render_central_panel(ctx);
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Check if a thread has panicked and log the error
fn check_thread_error(join: &mut Option<JoinHandle<()>>) {
    if let Some(thread) = join.take() {
        if thread.is_finished() {
            if let Err(e) = thread.join() {
                let error_msg = if let Some(es) = (&e).downcast_ref::<&'static str>() {
                    es.to_string()
                } else if let Some(es) = (&e).downcast_ref::<String>() {
                    es.clone()
                } else {
                    format!("{:?}", e)
                };
                tracing::error!("Thread panicked: {}", error_msg);
            }
        } else {
            *join = Some(thread);
        }
    }
}

/// Draw aligned words overlay on the mel spectrogram
fn draw_aligned_words(
    ctx: &egui::Context,
    aligned_words: &[AlignedWord],
    ui: &mut egui::Ui,
    mel_response: egui::InnerResponse<()>,
    display: &DisplayMel,
) {
    // Use frame_count for actual content, not texture size
    if display.frame_count == 0 {
        return;
    }

    // Each frame is 10ms (100 Hz)
    let mel_seconds = display.frame_count as f32 / MEL_UPDATE_HZ;

    if mel_seconds < 0.1 {
        return;
    }

    let rx = mel_response.response.rect;
    let seconds_to_pixels = rx.width() / mel_seconds;

    for (i, word) in aligned_words.iter().enumerate() {
        let row = (i % ALIGNED_WORD_ROWS) as f32 * ALIGNED_WORD_ROW_HEIGHT;
        let word_pixels = WORD_CHAR_WIDTH * word.word.len() as f32;

        let rect = egui::Rect::from_min_max(
            egui::pos2(
                rx.left() + seconds_to_pixels * word.start as f32,
                rx.top() + row,
            ),
            egui::pos2(
                rx.left() + seconds_to_pixels * word.end as f32 + word_pixels,
                rx.top() + ALIGNED_WORD_ROW_HEIGHT + row,
            ),
        );

        // Draw background
        ui.painter()
            .rect_filled(rect, 0.0, egui::Color32::from_rgb(0, 0, 0));

        // Draw text with color based on probability
        let color = egui::Color32::from_rgb(
            255, // Always red component to ensure visibility
            (word.probability * 255.0) as u8,
            (word.probability * 255.0) as u8,
        );

        ui.painter().text(
            rect.left_center(),
            egui::Align2::LEFT_CENTER,
            &word.word,
            egui::TextStyle::Body.resolve(&ctx.style()),
            color,
        );
    }
}

/// Plot audio level over time
fn plot_level(level: &VecDeque<f32>, ui: &mut egui::Ui) {
    let points: PlotPoints<'_> = level
        .iter()
        .enumerate()
        .map(|(i, v)| [i as f64, *v as f64])
        .collect();

    let line = Line::new("level", points);

    ui.add_enabled_ui(false, |ui| {
        Plot::new("level_plot")
            .width(LEVEL_PLOT_WIDTH)
            .height(LEVEL_PLOT_HEIGHT)
            .include_y(0.0)
            .include_y(1.0)
            .view_aspect(2.0)
            .show(ui, |plot_ui| plot_ui.line(line));
    });
}

/// Draw mel spectrogram from pre-allocated texture with scrolling
fn draw_mel(mel: &mut DisplayMel, ui: &mut egui::Ui) {
    if let Some(tex) = &mel.texture {
        // Fixed display width, spectrogram scrolls within
        const DISPLAY_WIDTH: f32 = 400.0;
        const DISPLAY_HEIGHT: f32 = 80.0;

        // Calculate visible portion (scroll to show most recent frames)
        let total_frames = MAX_MEL_FRAMES as f32;
        let visible_frames = DISPLAY_WIDTH; // 1 pixel per frame
        
        // UV coordinates for scrolling: show the rightmost portion
        let start_frame = if mel.frame_count as f32 > visible_frames {
            (mel.frame_count as f32 - visible_frames) / total_frames
        } else {
            0.0
        };
        let end_frame = mel.frame_count as f32 / total_frames;

        // Create UV rect to show scrolled portion
        let uv = egui::Rect::from_min_max(
            egui::pos2(start_frame, 0.0),
            egui::pos2(end_frame.max(0.01), 1.0), // Ensure non-zero width
        );

        ui.add(
            egui::Image::from_texture(egui::load::SizedTexture::from_handle(tex))
                .uv(uv)
                .corner_radius(5.0)
                .maintain_aspect_ratio(false)
                .fit_to_exact_size(egui::vec2(DISPLAY_WIDTH, DISPLAY_HEIGHT)),
        );
    }
}

// =============================================================================
// Worker Thread Management
// =============================================================================

/// Holds handles to keep audio streams and worker threads alive
struct Worker {
    audio: StreamState,
    audio2: Option<StreamState>,
    partial_thread: Option<JoinHandle<()>>,
    whisper_thread: Option<JoinHandle<()>>,
}

/// Start audio capture and transcription
fn start_listening(
    app: &Sender<WhisperUpdate>,
    app_device: &AppDevice,
    app_device2: &AppDevice,
    params: WhisperParams,
) -> Result<Worker, anyhow::Error> {
    // Start partial transcription thread if enabled
    let (partial_thread, partial_tx) = if params.partials {
        let (partial_tx, partial_rx) = mpsc::channel();
        let thread = crate::partial::start_partial_thread(app.clone(), partial_rx)?;
        (Some(thread), Some(partial_tx))
    } else {
        (None, None)
    };

    // Start filtered audio channel
    let (filtered_tx, filtered_rx) = mpsc::channel();

    // Start primary audio stream
    let stream =
        crate::audio::start_audio_thread(app.clone(), app_device, filtered_tx.clone(), partial_tx)?;

    // Start secondary audio stream if different device selected
    let audio2 = if app_device2.name != app_device.name {
        Some(crate::audio::start_audio_thread(
            app.clone(),
            app_device2,
            filtered_tx,
            None,
        )?)
    } else {
        None
    };

    // Start whisper transcription thread
    let whisper_thread = crate::whisper::start_whisper_thread(app.clone(), filtered_rx, params)?;

    Ok(Worker {
        audio: stream,
        audio2,
        partial_thread,
        whisper_thread: Some(whisper_thread),
    })
}
