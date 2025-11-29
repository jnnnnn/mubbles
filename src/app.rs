use std::{
    collections::VecDeque,
    sync::mpsc::{self, Sender, TryRecvError},
    thread::JoinHandle,
    time::Duration,
};

use candle_core::Tensor;
use egui_plot::{Line, Plot, PlotPoints};

use crate::{
    audio::{check_device_enumeration, get_devices, AppDevice, DeviceEnumerationReceiver, StreamState},
    file_transcription::transcribe_file,
    partial::PARTIAL_MEL_BINS,
    summary,
    whisper::{WhichModel, WhisperParams},
    whisper_word_align::AlignedWord,
};

// UI constants
const LEVEL_PLOT_WIDTH: f32 = 100.0;
const LEVEL_PLOT_HEIGHT: f32 = 30.0;
const LEVEL_BUFFER_SIZE: usize = 100;
const MEL_BUFFER_SIZE: usize = 500; // 5 seconds at 100Hz
const MEL_UPDATE_HZ: f32 = 100.0;
const REPAINT_INTERVAL_MS: u64 = 100;
const ALIGNED_WORD_ROWS: usize = 6;
const ALIGNED_WORD_ROW_HEIGHT: f32 = 12.0;
const WORD_CHAR_WIDTH: f32 = 7.0;

/// Application tab selection
#[derive(Debug, PartialEq, serde::Deserialize, serde::Serialize)]
enum AppTab {
    Transcript,
    StatisticalSummary,
    AISummary,
    Settings,
    Log,
}

/// Mel spectrogram display state
struct DisplayMel {
    buffer: VecDeque<[u8; PARTIAL_MEL_BINS]>,
    texture: Option<egui::TextureHandle>,
    image: Option<egui::ColorImage>,
    min: f32,
    max: f32,
}

impl DisplayMel {
    fn new() -> Self {
        Self {
            buffer: VecDeque::with_capacity(MEL_BUFFER_SIZE),
            texture: None,
            image: None,
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

    fn push_frame(&mut self, frame: &[f32]) {
        self.update_range(frame);

        let bytes: Vec<u8> = frame
            .iter()
            .map(|&x| {
                let normalized = (x - self.min) * (255.0 / (self.max - self.min));
                normalized.clamp(0.0, 255.0) as u8
            })
            .collect();

        let mut arr = [0u8; PARTIAL_MEL_BINS];
        let len = bytes.len().min(PARTIAL_MEL_BINS);
        arr[..len].copy_from_slice(&bytes[..len]);

        if self.buffer.len() >= MEL_BUFFER_SIZE {
            self.buffer.pop_front();
        }
        self.buffer.push_back(arr);
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
    #[serde(skip)]
    selected_device1: usize,
    #[serde(skip)]
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
    mel1: DisplayMel,
    #[serde(skip)]
    mel2: Tensor,

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

    // Logging state
    #[serde(skip)]
    log_buffer: crate::log_capture::LogBuffer,
    log_level: usize, // 0=TRACE, 1=DEBUG, 2=INFO, 3=WARN, 4=ERROR
}

impl Default for MubblesApp {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        let (devices, device_enumeration_rx) = get_devices();

        let selected_device = Self::find_default_device(&devices);

        Self {
            // Transcript state
            text: String::new(),
            aligned_words: vec![],

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
            mel1: DisplayMel::new(),
            mel2: Self::create_empty_mel_tensor(),

            // UI state
            changed: false,
            tab: AppTab::Transcript,

            // Summary state
            statistical_summary: summary::SummaryState::default(),
            ai_summary: summary::SummaryState::default(),

            // File transcription state
            file_transcription_running: false,
            file_transcription_thread: None,

            // Logging state
            log_buffer: crate::log_capture::LogBuffer::new(),
            log_level: 2, // Default to INFO
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

    /// Create an empty mel spectrogram tensor
    fn create_empty_mel_tensor() -> Tensor {
        Tensor::zeros((2, 3), candle_core::DType::F32, &candle_core::Device::Cpu)
            .expect("Failed to create mel tensor")
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
    fn process_whisper_updates(&mut self) {
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
                    self.text.push_str(t.trim());
                    self.text.push('\n');
                    self.changed = true;
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
                    self.aligned_words = a;
                }
                WhisperUpdate::MelFrame(_frame) => {
                    // Disabled: update_mel_buffer(&frame, &mut self.mel1);
                }
                WhisperUpdate::Mel(_m) => {
                    // Disabled: slow (up to 1s)
                    tracing::debug!(
                        "App received mel spectrogram with shape: {:?}",
                        self.mel2.shape()
                    );
                }
                WhisperUpdate::Status(s) => {
                    self.status = s;
                }
                WhisperUpdate::FileTranscriptionComplete => {
                    self.file_transcription_running = false;
                    self.status = "File transcription complete".to_string();
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

                // Device selectors
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

    /// Render the mel spectrogram
    fn render_mel_row(&mut self, ui: &mut egui::Ui) -> egui::InnerResponse<()> {
        ui.with_layout(
            egui::Layout::left_to_right(egui::Align::LEFT)
                .with_main_wrap(true)
                .with_cross_align(egui::Align::TOP),
            |ui| {
                if let Err(e) = draw_mel2(&mut self.mel2, &mut self.mel1, ui) {
                    tracing::error!("Error drawing mel spectrogram: {}", e);
                }
            },
        )
    }

    /// Render aligned words overlay on mel spectrogram
    fn render_aligned_words(
        &mut self,
        ctx: &egui::Context,
        ui: &mut egui::Ui,
        mel_response: egui::InnerResponse<()>,
    ) {
        draw_aligned_words(ctx, &mut self.aligned_words, ui, mel_response, &self.mel1);
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
        self.mel2 = Self::create_empty_mel_tensor();
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
        // Open file dialog
        let file_dialog = rfd::FileDialog::new()
            .add_filter("Audio Files", &["wav", "mp3", "flac", "ogg", "m4a"])
            .set_title("Select Audio File to Transcribe");

        if let Some(path) = file_dialog.pick_file() {
            let tx = self.whisper_tx.clone();
            let model = WhichModel::from(self.selected_model);
            let accuracy = self.accuracy;

            self.file_transcription_running = true;

            let thread = std::thread::spawn(move || {
                if let Err(e) = transcribe_file(path, model, accuracy, tx) {
                    tracing::error!("File transcription failed: {}", e);
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
            ui.selectable_value(
                &mut self.tab,
                AppTab::StatisticalSummary,
                "Statistical Summary",
            );
            ui.selectable_value(&mut self.tab, AppTab::AISummary, "AI Summary");
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
                ui.heading("AI Prompts");
                ui.add_space(10.0);

                ui.label("User Prompt:");
                ui.add_sized(
                    egui::vec2(ui.available_width(), 150.0),
                    egui::TextEdit::multiline(&mut self.ai_summary.user_prompt),
                );

                ui.add_space(10.0);

                ui.label("System Prompt:");
                ui.add_sized(
                    egui::vec2(ui.available_width(), 150.0),
                    egui::TextEdit::multiline(&mut self.ai_summary.system_prompt),
                );

                ui.add_space(20.0);
                ui.separator();
                ui.add_space(10.0);

                ui.heading("File Transcription");
                ui.add_space(10.0);

                ui.horizontal(|ui| {
                    let button_text = if self.file_transcription_running {
                        "Transcribing..."
                    } else {
                        "Transcribe Audio File..."
                    };

                    if ui
                        .add_enabled(
                            !self.file_transcription_running,
                            egui::Button::new(button_text),
                        )
                        .clicked()
                    {
                        self.start_file_transcription();
                    }

                    if self.file_transcription_running {
                        ui.spinner();
                    }
                });

                ui.label("Select a WAV audio file to transcribe using the current model settings.");
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
    FileTranscriptionComplete,
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
        self.process_whisper_updates();

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
    let mel_seconds = match &display.texture {
        Some(tex) => tex.size()[0] as f32 / MEL_UPDATE_HZ,
        None => return,
    };

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

/// Update mel buffer with a new frame (currently unused)
#[allow(dead_code)]
fn update_mel_buffer(frame: &[f32], mel: &mut DisplayMel) {
    mel.push_frame(frame);
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

/// Draw mel spectrogram using DisplayMel buffer (currently unused)
#[allow(dead_code)]
fn draw_mel1(mel: &mut DisplayMel, ui: &mut egui::Ui) {
    let DisplayMel {
        buffer,
        texture,
        image,
        ..
    } = mel;

    // Initialize image if needed
    let image = image.get_or_insert_with(|| {
        let black = egui::Color32::from_black_alpha(0);
        egui::ColorImage::filled([PARTIAL_MEL_BINS, LEVEL_BUFFER_SIZE], black)
    });

    // Initialize texture if needed
    let texture = texture.get_or_insert_with(|| {
        ui.ctx().load_texture(
            "mel_spectrogram",
            image.clone(),
            egui::TextureOptions::default(),
        )
    });

    let xmax = buffer.len();
    let mut pixels: Vec<egui::Color32> = vec![egui::Color32::from_gray(0); PARTIAL_MEL_BINS * xmax];

    // Convert buffer to pixels
    for (x, frame) in buffer.iter().enumerate() {
        for (y, &value) in frame.iter().enumerate() {
            pixels[x + y * xmax] = egui::Color32::from_gray(value);
        }
    }

    image.pixels = pixels;
    image.size = [xmax, PARTIAL_MEL_BINS];
    texture.set(image.clone(), egui::TextureOptions::default());

    ui.add(
        egui::Image::from_texture(&*texture)
            .corner_radius(10.0)
            .maintain_aspect_ratio(false)
            .fit_to_exact_size(egui::vec2(
                buffer.len() as f32 * 4.0,
                PARTIAL_MEL_BINS as f32,
            )),
    );
}

/// Draw mel spectrogram from Tensor
fn draw_mel2(
    mel2: &Tensor,
    display: &mut DisplayMel,
    ui: &mut egui::Ui,
) -> Result<(), anyhow::Error> {
    let shape = mel2.shape();
    if shape.rank() != 2 {
        anyhow::bail!(
            "Unexpected tensor rank, expected: 2, got: {} ({:?})",
            shape.rank(),
            shape.dims()
        );
    }

    let n_frames = shape.dims()[1];
    if n_frames < 10 {
        return Ok(());
    }

    let mut mel_image = egui::ColorImage::filled(
        [n_frames, PARTIAL_MEL_BINS],
        egui::Color32::from_black_alpha(0),
    );

    let mel_min = mel2.min_all()?.to_scalar::<f32>()?;
    let mel_max = mel2.max_all()?.to_scalar::<f32>()? + 0.01;
    let mel_data = mel2.to_vec2::<f32>()?;

    // Convert tensor data to image pixels
    for f in 0..n_frames {
        for b in 0..PARTIAL_MEL_BINS {
            let value = mel_data[b][f];
            let color_value =
                ((value - mel_min) / (mel_max - mel_min) * 255.0).clamp(0.0, 255.0) as u8;
            mel_image.pixels[b * n_frames + f] =
                egui::Color32::from_rgb(color_value, color_value, color_value);
        }
    }

    // Create or update texture
    if display.texture.is_none() {
        let tex = ui.ctx().load_texture(
            "mel_spectrogram2",
            mel_image.clone(),
            egui::TextureOptions::default(),
        );
        display.texture = Some(tex);
    }

    let tex = display.texture.as_mut().unwrap();
    tex.set(mel_image, egui::TextureOptions::default());

    ui.add(
        egui::Image::from_texture(&*tex)
            .corner_radius(10.0)
            .maintain_aspect_ratio(false),
    );

    Ok(())
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
