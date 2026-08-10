use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, TryRecvError},
        Arc,
    },
    thread::JoinHandle,
    time::Duration,
};

use candle_core::Tensor;
use egui_extras::{Column, TableBuilder};
use egui_plot::{HLine, Line, Plot, PlotPoints};

use crate::{
    audio::{
        check_device_enumeration, get_devices, AppDevice, DeviceEnumerationReceiver, PcmAudio,
    },
    file_transcription::{self, FileUpdate},
    partial::PARTIAL_MEL_BINS,
    summary,
    whisper::{WhichModel, WhisperParams},
    whisper_word_align::AlignedWord,
    workers::{self, AudioWorker, WhisperWorker},
};

// UI constants
const LEVEL_PLOT_WIDTH: f32 = 100.0;
const LEVEL_PLOT_HEIGHT: f32 = 30.0;
const LEVEL_BUFFER_SIZE: usize = 100;
const MEL_DISPLAY_SECS: f32 = 30.0;
const REPAINT_INTERVAL_MS: u64 = 100;
const ALIGNED_WORD_ROWS: usize = 6;
const ALIGNED_WORD_ROW_HEIGHT: f32 = 12.0;
const WORD_CHAR_WIDTH: f32 = 7.0;
const DEFAULT_SILENCE_TIMEOUT_MS: f32 = 1000.0; // Pause before transcription triggers

fn default_silence_timeout() -> f32 {
    DEFAULT_SILENCE_TIMEOUT_MS
}

/// Application tab selection
#[derive(Debug, PartialEq, serde::Deserialize, serde::Serialize)]
enum AppTab {
    Transcript,
    TranscriptionHistory,
    StatisticalSummary,
    AISummary,
    FileTranscription,
    Devices,
    IgnoredPhrases,
    Settings,
    Log,
}

const MAX_MEL_FRAMES: usize = 3000; // 30 seconds at 100 Hz

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
            tex.set_partial(
                [self.frame_count, 0],
                column,
                egui::TextureOptions::default(),
            );
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
    selected_device_names: HashSet<String>,

    // Model configuration
    selected_model: usize,
    accuracy: usize,

    // Feature flags
    autotype: bool,
    partials: bool,
    echo_cancel: bool,
    #[serde(skip)]
    always_on_top: bool,

    // Audio subsystem (capture + VAD) — can run independently of whisper
    #[serde(skip)]
    audio_worker: Option<AudioWorker>,
    // Whisper subsystem (transcription) — can start/stop without restarting audio
    #[serde(skip)]
    whisper_worker: Option<WhisperWorker>,
    // Held when audio is running but whisper isn't (e.g. between model changes)
    #[serde(skip)]
    pending_filtered_rx: Option<Receiver<PcmAudio>>,
    #[serde(skip)]
    whisper_paused: Arc<AtomicBool>,
    #[serde(skip)]
    from_whisper: mpsc::Receiver<WhisperUpdate>,
    #[serde(skip)]
    whisper_tx: mpsc::Sender<WhisperUpdate>,

    // Visualization state
    #[serde(skip)]
    device_levels: HashMap<String, VecDeque<f32>>,
    #[serde(skip)]
    device_vad_levels: HashMap<String, VecDeque<f32>>,
    #[serde(skip)]
    device_muted: HashMap<String, bool>,
    device_thresholds: HashMap<String, f32>,
    #[serde(default = "default_silence_timeout")]
    silence_timeout_ms: f32,
    #[serde(skip)]
    mel: DisplayMel,

    // UI state
    #[serde(skip)]
    changed: bool,
    #[serde(skip)]
    tab: AppTab,
    #[serde(skip)]
    was_focused: bool,
    #[serde(skip)]
    last_device_refresh: std::time::Instant,

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

    // Ignored phrases
    ignored_phrases: Vec<String>,
    #[serde(skip)]
    new_ignored_phrase: String,
}

impl Default for MubblesApp {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        let (devices, device_enumeration_rx) = get_devices();
        let (file_tx, file_rx) = mpsc::channel();

        let selected_device_names = Self::default_device_names(&devices);

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
            selected_device_names,

            // Model configuration
            selected_model: 1,
            accuracy: 1,

            // Feature flags
            autotype: false,
            partials: false,
            echo_cancel: false,
            always_on_top: false,

            // Worker threads
            audio_worker: None,
            whisper_worker: None,
            pending_filtered_rx: None,
            whisper_paused: Arc::new(AtomicBool::new(false)),
            from_whisper: rx,
            whisper_tx: tx,

            // Visualization state
            device_levels: HashMap::new(),
            device_vad_levels: HashMap::new(),
            device_muted: HashMap::new(),
            device_thresholds: HashMap::new(),
            silence_timeout_ms: DEFAULT_SILENCE_TIMEOUT_MS,
            mel: DisplayMel::new(),

            // UI state
            changed: false,
            tab: AppTab::Transcript,
            was_focused: false,
            last_device_refresh: std::time::Instant::now(),

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

            // Ignored phrases
            ignored_phrases: Vec::new(),
            new_ignored_phrase: String::new(),
        }
    }
}

impl MubblesApp {
    /// Find default device names to select initially.
    fn default_device_names(devices: &[AppDevice]) -> HashSet<String> {
        let mut names = HashSet::new();

        #[cfg(target_os = "linux")]
        {
            if let Some(d) = devices.iter().find(|d| d.name.contains("Monitor")) {
                names.insert(d.name.clone());
            } else if let Some(d) = devices.first() {
                names.insert(d.name.clone());
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            use cpal::traits::{DeviceTrait, HostTrait};
            let host = cpal::default_host();
            let default_device_name = host
                .default_output_device()
                .and_then(|d| d.name().ok())
                .unwrap_or_else(|| "Unknown".to_owned());

            if let Some(d) = devices.iter().find(|d| d.name == default_device_name) {
                names.insert(d.name.clone());
            } else if let Some(d) = devices.first() {
                names.insert(d.name.clone());
            }
        }

        names
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
                    // Skip ignored phrases (case-insensitive comparison)
                    let is_ignored = self
                        .ignored_phrases
                        .iter()
                        .any(|ignored| trimmed.eq_ignore_ascii_case(ignored.trim()));
                    if is_ignored {
                        tracing::debug!("Ignored phrase: {}", trimmed);
                    } else {
                        self.text.push_str(&trimmed);
                        self.text.push('\n');
                        self.changed = true;
                        self.transcription_logger
                            .append(&self.transcription_folder, &trimmed);
                        if self.autotype {
                            crate::autotype::type_text(&trimmed);
                        }
                    }
                    self.mel.reset();
                }
                WhisperUpdate::Recording(r) => self.recording = r,
                WhisperUpdate::Transcribing(t) => self.transcribing = t,
                WhisperUpdate::Level {
                    device,
                    level,
                    vad_prob,
                    muted,
                } => {
                    let dev_level = self
                        .device_levels
                        .entry(device.clone())
                        .or_insert_with(|| VecDeque::with_capacity(LEVEL_BUFFER_SIZE));
                    if dev_level.len() >= LEVEL_BUFFER_SIZE {
                        dev_level.pop_front();
                    }
                    dev_level.push_back(level);
                    let dev_vad = self
                        .device_vad_levels
                        .entry(device.clone())
                        .or_insert_with(|| VecDeque::with_capacity(LEVEL_BUFFER_SIZE));
                    if dev_vad.len() >= LEVEL_BUFFER_SIZE {
                        dev_vad.pop_front();
                    }
                    dev_vad.push_back(vad_prob);
                    self.device_muted.insert(device, muted);
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
            tracing::info!(
                "Device enumeration complete, found {} devices",
                new_devices.len()
            );
            self.devices = new_devices;

            // If no selected devices exist in the new list, pick defaults
            if !self
                .devices
                .iter()
                .any(|d| self.selected_device_names.contains(&d.name))
            {
                self.selected_device_names = Self::default_device_names(&self.devices);
            }

            self.status = format!("Found {} audio devices", self.devices.len());
        }
    }

    /// Re-enumerate audio devices. On Windows/macOS this is synchronous; on
    /// Linux it kicks off async PipeWire enumeration (results arrive via
    /// `check_device_enumeration`). Does not interrupt an active recording.
    /// Skips if a previous enumeration is still in flight.
    fn refresh_devices(&mut self) {
        // Don't start a new enumeration if one is already pending (Linux)
        if self.device_enumeration_rx.is_some() {
            return;
        }
        let (devices, rx) = get_devices();
        self.devices = devices;
        self.device_enumeration_rx = rx;
        self.last_device_refresh = std::time::Instant::now();
        tracing::info!("Device refresh triggered, {} devices", self.devices.len());
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
                plot_levels(&self.device_levels, ui);

                // Start/Stop button
                let started = self.audio_worker.is_some();
                let button_text = if started { "Stop" } else { "Start" };
                if ui.button(button_text).clicked() {
                    self.toggle_recording();
                }

                // Model selector
                let model = egui::ComboBox::from_label("Model")
                    .selected_text(WhichModel::from(self.selected_model).to_string())
                    .show_index(ui, &mut self.selected_model, WhichModel::len(), |i| {
                        WhichModel::from(i).to_string()
                    });
                if model.changed() {
                    self.restart_whisper();
                }

                if ui.button("Devices…").clicked() {
                    self.tab = AppTab::Devices;
                }
            },
        );
    }

    /// Toggle recording on/off. Starts or stops both audio capture and
    /// whisper transcription together. Audio and whisper can also be
    /// controlled independently: dropping and recreating the whisper worker
    /// (e.g. after a model change) does not interrupt audio capture.
    fn toggle_recording(&mut self) {
        if self.audio_worker.is_some() {
            // Stop whisper first so it cleanly exits before audio stops
            self.whisper_worker = None;
            self.audio_worker = None;
            self.pending_filtered_rx = None;
            self.ai_summary.whisper_paused = None;
        } else {
            let device_refs: Vec<&AppDevice> = self
                .devices
                .iter()
                .filter(|d| self.selected_device_names.contains(&d.name))
                .collect();
            if device_refs.is_empty() {
                self.status =
                    "No devices selected. Go to the Devices tab to select devices.".to_string();
                return;
            }

            // Start audio
            match workers::start_audio(
                &self.whisper_tx,
                &device_refs,
                self.echo_cancel,
                &self.device_thresholds,
                self.silence_timeout_ms,
                self.partials,
            ) {
                Ok((audio_worker, filtered_rx)) => {
                    self.audio_worker = Some(audio_worker);
                    self.pending_filtered_rx = Some(filtered_rx);
                }
                Err(e) => {
                    tracing::error!("Failed to start audio: {}", e);
                    self.audio_worker = None;
                    self.pending_filtered_rx = None;
                    return;
                }
            }

            // Start whisper
            let filtered_rx = self.pending_filtered_rx.take().unwrap();
            match workers::start_whisper(
                &self.whisper_tx,
                filtered_rx,
                WhisperParams {
                    accuracy: self.accuracy,
                    model: WhichModel::from(self.selected_model),
                    partials: self.partials,
                },
                self.whisper_paused.clone(),
            ) {
                Ok(whisper_worker) => {
                    self.whisper_worker = Some(whisper_worker);
                    self.ai_summary.whisper_paused = Some(self.whisper_paused.clone());
                }
                Err(e) => {
                    tracing::error!("Failed to start whisper: {}", e);
                    self.audio_worker = None;
                }
            }
        }
    }

    /// Restart whisper transcription without restarting audio.
    /// Used when model, accuracy, or other whisper-only settings change.
    fn restart_whisper(&mut self) {
        if let Some(whisper) = self.whisper_worker.take() {
            self.pending_filtered_rx = Some(whisper.stop());
        }
        // If audio isn't running, nothing to do
        let filtered_rx = match self.pending_filtered_rx.take() {
            Some(rx) => rx,
            None => return,
        };
        match workers::start_whisper(
            &self.whisper_tx,
            filtered_rx,
            WhisperParams {
                accuracy: self.accuracy,
                model: WhichModel::from(self.selected_model),
                partials: self.partials,
            },
            self.whisper_paused.clone(),
        ) {
            Ok(whisper_worker) => {
                self.whisper_worker = Some(whisper_worker);
            }
            Err(e) => {
                tracing::error!("Failed to restart whisper: {}", e);
                // Audio is still running; whisper just failed to restart.
                // Drop audio too since transcription is unavailable.
                self.audio_worker = None;
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

        Some(
            ui.with_layout(
                egui::Layout::left_to_right(egui::Align::LEFT)
                    .with_main_wrap(true)
                    .with_cross_align(egui::Align::TOP),
                |ui| {
                    draw_mel(&mut self.mel, ui);
                },
            ),
        )
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
                    self.restart_whisper();
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
                    self.restart_whisper();
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
            .add_filter(
                "Audio Files",
                &["wav", "mp3", "flac", "ogg", "m4a", "aac", "wma"],
            )
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
            // Always poll for AI summary updates, even when not on the AI tab
            if let Some(status) = summary::poll_ai_updates(&mut self.ai_summary, &mut self.text) {
                self.status = status;
            }

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
            ui.selectable_value(&mut self.tab, AppTab::Devices, "Devices");
            ui.selectable_value(&mut self.tab, AppTab::IgnoredPhrases, "Filters");
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
                        if self.ai_summary.provider == summary::ApiProvider::Ollama {
                            self.ai_summary.ollama_models =
                                summary::fetch_ollama_models(&self.ai_summary.api_url);
                            if let Some(first) = self.ai_summary.ollama_models.first() {
                                self.ai_summary.model = first.clone();
                            }
                            self.ai_summary.ollama_model_ctx = summary::fetch_ollama_model_ctx(
                                &self.ai_summary.api_url,
                                &self.ai_summary.model,
                            );
                        } else {
                            self.ai_summary.ollama_model_ctx = None;
                        }
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
                    if self.ai_summary.provider == summary::ApiProvider::Ollama {
                        let prev_model = self.ai_summary.model.clone();
                        egui::ComboBox::from_id_salt("ollama_model")
                            .selected_text(&self.ai_summary.model)
                            .width(ui.available_width() - 90.0)
                            .show_ui(ui, |ui| {
                                for m in &self.ai_summary.ollama_models.clone() {
                                    ui.selectable_value(&mut self.ai_summary.model, m.clone(), m);
                                }
                            });
                        if ui.button("Refresh").clicked() {
                            self.ai_summary.ollama_models =
                                summary::fetch_ollama_models(&self.ai_summary.api_url);
                            if !self.ai_summary.ollama_models.is_empty()
                                && !self
                                    .ai_summary
                                    .ollama_models
                                    .contains(&self.ai_summary.model)
                            {
                                self.ai_summary.model =
                                    self.ai_summary.ollama_models[0].clone();
                            }
                        }
                        if self.ai_summary.model != prev_model {
                            self.ai_summary.ollama_model_ctx = summary::fetch_ollama_model_ctx(
                                &self.ai_summary.api_url,
                                &self.ai_summary.model,
                            );
                        }
                    } else {
                        ui.add_sized(
                            egui::vec2(ui.available_width(), 20.0),
                            egui::TextEdit::singleline(&mut self.ai_summary.model)
                                .hint_text("gpt-4o-mini"),
                        );
                    }
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

                ui.add(
                    egui::Slider::new(&mut self.ai_summary.ai_input_chars, 500..=50000)
                        .text("Characters per AI chunk")
                        .logarithmic(true),
                );

                ui.add(
                    egui::Slider::new(&mut self.ai_summary.summary_context_lines, 0..=50)
                        .text("Summary context lines sent to AI"),
                );

                ui.horizontal(|ui| {
                    ui.add(
                        egui::Slider::new(&mut self.ai_summary.max_tokens, 256..=32768)
                            .text("Max response tokens")
                            .logarithmic(true),
                    );
                    if let Some(ctx) = self.ai_summary.ollama_model_ctx {
                        if ui.button(format!("Use model ctx: {}", ctx)).clicked() {
                            self.ai_summary.max_tokens = ctx;
                        }
                    }
                });

                ui.add(
                    egui::Slider::new(&mut self.ai_summary.thinking_budget, 0..=8192)
                        .text("Thinking budget (chars, 0 = disable)")
                        .logarithmic(true),
                );

                ui.checkbox(
                    &mut self.ai_summary.free_gpu,
                    "Pause whisper during AI summary (free GPU for Ollama)",
                );

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

                ui.heading("Speech Detection");
                ui.add_space(10.0);

                ui.add(
                    egui::Slider::new(&mut self.silence_timeout_ms, 200.0..=5000.0)
                        .text("Pause before transcription")
                        .custom_formatter(|n, _| format!("{:.1}s", n / 1000.0))
                        .custom_parser(|s| {
                            s.trim_end_matches('s')
                                .parse::<f64>()
                                .ok()
                                .map(|v| v * 1000.0)
                        }),
                );
                ui.label(
                    "How long to wait after you stop speaking before triggering transcription.\n\
                     Lower = snappier but may split utterances. Higher = more thinking time.",
                );
                if ui
                    .button("Restart to apply")
                    .on_hover_text("Changes take effect next time you start recording")
                    .clicked()
                {
                    self.whisper_worker = None;
                    self.audio_worker = None;
                    self.pending_filtered_rx = None;
                }

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
            AppTab::Devices => {
                self.render_devices_tab(ui);
            }
            AppTab::IgnoredPhrases => {
                ui.heading("Ignored Phrases");
                ui.add_space(5.0);
                ui.label("Transcriptions matching these phrases will be silently discarded (case-insensitive).");
                ui.add_space(10.0);

                // Add new phrase row
                ui.horizontal(|ui| {
                    let response = ui.add_sized(
                        egui::vec2(ui.available_width() - 80.0, 20.0),
                        egui::TextEdit::singleline(&mut self.new_ignored_phrase)
                            .hint_text("e.g., Thank you."),
                    );
                    let add_clicked = ui.button("Add").clicked();
                    let enter_pressed = response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if (add_clicked || enter_pressed) && !self.new_ignored_phrase.trim().is_empty() {
                        let phrase = self.new_ignored_phrase.trim().to_string();
                        if !self.ignored_phrases.contains(&phrase) {
                            self.ignored_phrases.push(phrase);
                            tracing::info!("Added ignored phrase: {}", self.new_ignored_phrase.trim());
                        }
                        self.new_ignored_phrase.clear();
                    }
                });

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(5.0);

                if self.ignored_phrases.is_empty() {
                    ui.label("No ignored phrases yet. Add one above.");
                } else {
                    ui.label(format!("{} phrase(s) ignored:", self.ignored_phrases.len()));
                    ui.add_space(5.0);

                    let mut remove_idx: Option<usize> = None;
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for (i, phrase) in self.ignored_phrases.iter().enumerate() {
                            ui.horizontal(|ui| {
                                ui.label(phrase);
                                if ui.button("✕").clicked() {
                                    remove_idx = Some(i);
                                }
                            });
                        }
                    });
                    if let Some(i) = remove_idx {
                        let removed = self.ignored_phrases.remove(i);
                        tracing::info!("Removed ignored phrase: {}", removed);
                    }
                }
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

    /// Render the devices tab with checkboxes for each audio device
    fn render_devices_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Audio Devices");
        ui.add_space(10.0);

        ui.horizontal(|ui| {
            if ui.button("Refresh Devices").clicked() {
                self.refresh_devices();
            }
            ui.label(format!("{} devices found", self.devices.len()));
        });

        if ui
            .checkbox(&mut self.echo_cancel, "Echo cancellation")
            .on_hover_text("Mute mic input while speaker output is active")
            .changed()
        {
            if let Some(audio) = &self.audio_worker {
                audio.echo_cancel.store(self.echo_cancel, Ordering::Relaxed);
            }
        }

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(5.0);

        let device_names: Vec<(String, bool)> = self
            .devices
            .iter()
            .map(|d| (d.name.clone(), d.is_output))
            .collect();

        for (name, is_output) in &device_names {
            ui.horizontal(|ui| {
                let mut selected = self.selected_device_names.contains(name);
                if ui.checkbox(&mut selected, "").changed() {
                    if selected {
                        self.selected_device_names.insert(name.clone());
                        if let Some(audio) = &mut self.audio_worker {
                            if let Some(device) = self.devices.iter().find(|d| &d.name == name) {
                                let threshold = self
                                    .device_thresholds
                                    .get(name)
                                    .copied()
                                    .unwrap_or(workers::DEFAULT_THRESHOLD);
                                audio.add_device(device, threshold);
                            }
                        }
                    } else {
                        self.selected_device_names.remove(name);
                        if let Some(audio) = &mut self.audio_worker {
                            audio.remove_device(name);
                        }
                    }
                }

                // Muted indicator
                if let Some(&muted) = self.device_muted.get(name.as_str()) {
                    if muted {
                        ui.label(
                            egui::RichText::new("MUTED")
                                .color(egui::Color32::from_rgb(255, 100, 100))
                                .strong(),
                        );
                    }
                }

                // Per-device VAD sensitivity slider (speech probability threshold)
                let threshold = self
                    .device_thresholds
                    .entry(name.clone())
                    .or_insert(workers::DEFAULT_THRESHOLD);
                if ui
                    .add(egui::Slider::new(threshold, 0.1..=0.9).text("VAD sensitivity"))
                    .changed()
                {
                    // Push updated threshold to the running audio thread
                    if let Some(audio) = &self.audio_worker {
                        if let Some(atom) = audio.device_threshold_atoms.get(name.as_str()) {
                            atom.store(threshold.to_bits(), Ordering::Relaxed);
                        }
                    }
                }

                // Per-device chart: bars = audio level, line = VAD probability
                let dev_level = self.device_levels.get(name.as_str());
                let dev_vad = self.device_vad_levels.get(name.as_str());
                plot_level_with_vad(dev_level, dev_vad, *threshold, ui, name);

                // Device name comes last so varying lengths don't misalign controls
                let label = if *is_output {
                    format!("{} [output]", name)
                } else {
                    name.clone()
                };
                ui.label(label);
            });
        }
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
                header.col(|ui| {
                    ui.strong("Time");
                });
                header.col(|ui| {
                    ui.strong("Words");
                });
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
    Level {
        device: String,
        level: f32,
        vad_prob: f32,
        muted: bool,
    },
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
        // Keep the UI responsive with periodic repaints.
        // Use a faster rate on the Devices tab so charts look smooth.
        let interval = if self.tab == AppTab::Devices {
            16
        } else {
            REPAINT_INTERVAL_MS
        };
        ctx.request_repaint_after(Duration::from_millis(interval));

        // Check for thread panics
        if let Some(whisper) = &mut self.whisper_worker {
            whisper.check_panic();
        }
        check_thread_error(&mut self.file_transcription_thread);

        // Check if device enumeration completed
        self.check_device_enumeration();

        // Auto-refresh devices on window focus gain or periodically
        let focused = ctx.input(|i| i.focused);
        let focus_gained = focused && !self.was_focused;
        self.was_focused = focused;
        let refresh_interval = Duration::from_secs(30);
        let interval_elapsed = self.last_device_refresh.elapsed() >= refresh_interval;
        if focus_gained || interval_elapsed {
            self.refresh_devices();
            // If focus gained, also re-sync selected devices with worker
            if focus_gained {
                if let Some(audio) = &mut self.audio_worker {
                    for name in &self.selected_device_names {
                        if let Some(device) = self.devices.iter().find(|d| &d.name == name) {
                            let threshold = self
                                .device_thresholds
                                .get(name)
                                .copied()
                                .unwrap_or(workers::DEFAULT_THRESHOLD);
                            audio.add_device(device, threshold);
                        }
                    }
                }
            }
        }

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
    _display: &DisplayMel,
) {
    if aligned_words.is_empty() {
        return;
    }

    let rx = mel_response.response.rect;
    let seconds_to_pixels = rx.width() / MEL_DISPLAY_SECS;

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

/// Convert linear amplitude (0.0..1.0) to logarithmic dB scale (0.0..1.0)
/// Maps -60 dB to 0 dB range into 0.0..1.0
fn amplitude_to_log(v: f32) -> f64 {
    if v <= 0.0 {
        return 0.0;
    }
    // 20*log10(v) gives dB, range is roughly -60..0 for 0.001..1.0
    let db = 20.0 * (v as f64).log10();
    // Map -60..0 to 0..1
    ((db + 60.0) / 60.0).clamp(0.0, 1.0)
}

/// Distinct colors for device lines
const DEVICE_COLORS: &[egui::Color32] = &[
    egui::Color32::from_rgb(100, 180, 255),
    egui::Color32::from_rgb(255, 150, 80),
    egui::Color32::from_rgb(100, 220, 100),
    egui::Color32::from_rgb(220, 100, 220),
    egui::Color32::from_rgb(255, 220, 80),
    egui::Color32::from_rgb(80, 220, 220),
];

/// Plot all device levels as separate colored lines (logarithmic scale)
fn plot_levels(device_levels: &HashMap<String, VecDeque<f32>>, ui: &mut egui::Ui) {
    ui.add_enabled_ui(false, |ui| {
        Plot::new("level_plot")
            .width(LEVEL_PLOT_WIDTH)
            .height(LEVEL_PLOT_HEIGHT)
            .include_y(0.0)
            .include_y(1.0)
            .view_aspect(2.0)
            .show(ui, |plot_ui| {
                for (i, (name, level)) in device_levels.iter().enumerate() {
                    let points: PlotPoints<'_> = level
                        .iter()
                        .enumerate()
                        .map(|(j, v)| [j as f64, amplitude_to_log(*v)])
                        .collect();
                    let color = DEVICE_COLORS[i % DEVICE_COLORS.len()];
                    let line = Line::new(name.as_str(), points).color(color);
                    plot_ui.line(line);
                }
            });
    });
}

/// Plot audio level (bars) + VAD probability (line) with threshold marker.
/// Audio level is log-scaled like a traditional meter; VAD prob is linear [0,1].
fn plot_level_with_vad(
    level: Option<&VecDeque<f32>>,
    vad: Option<&VecDeque<f32>>,
    threshold: f32,
    ui: &mut egui::Ui,
    name: &str,
) {
    ui.add_enabled_ui(false, |ui| {
        Plot::new(format!("dev_level_{}", name))
            .width(LEVEL_PLOT_WIDTH)
            .height(LEVEL_PLOT_HEIGHT)
            .include_y(0.0)
            .include_y(1.0)
            .view_aspect(2.0)
            .show(ui, |plot_ui| {
                // Audio level as bars — discrete, chunky, like a traditional meter
                if let Some(level) = level {
                    let bars: Vec<egui_plot::Bar> = level
                        .iter()
                        .enumerate()
                        .map(|(i, v)| {
                            egui_plot::Bar::new(i as f64, amplitude_to_log(*v))
                                .fill(egui::Color32::from_rgba_premultiplied(100, 180, 255, 120))
                                .width(0.9)
                        })
                        .collect();
                    let bar_chart = egui_plot::BarChart::new("level", bars);
                    plot_ui.bar_chart(bar_chart);
                }

                // VAD probability as a line — smooth, continuous, visually distinct from bars
                if let Some(vad) = vad {
                    let vad_points: egui_plot::PlotPoints<'_> = vad
                        .iter()
                        .enumerate()
                        .map(|(i, v)| [i as f64, *v as f64])
                        .collect();
                    let vad_line = Line::new("vad", vad_points)
                        .color(egui::Color32::from_rgb(255, 200, 60))
                        .width(1.5_f32);
                    plot_ui.line(vad_line);
                }

                // Threshold line
                let thresh_line = HLine::new("threshold", threshold as f64)
                    .color(egui::Color32::from_rgb(255, 80, 80))
                    .style(egui_plot::LineStyle::Dashed { length: 4.0 });
                plot_ui.hline(thresh_line);
            });
    });
}

/// Draw mel spectrogram from pre-allocated texture, sized to fit available width
fn draw_mel(mel: &DisplayMel, ui: &mut egui::Ui) {
    let tex = match &mel.texture {
        Some(t) => t,
        None => return,
    };
    let width = ui.available_width().max(1.0);
    ui.add(
        egui::Image::from_texture(egui::load::SizedTexture::from_handle(tex))
            .maintain_aspect_ratio(false)
            .fit_to_exact_size(egui::vec2(width, PARTIAL_MEL_BINS as f32)),
    );
}
