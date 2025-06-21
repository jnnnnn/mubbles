use std::{
    collections::VecDeque, sync::mpsc::{self, Sender, TryRecvError}, thread::JoinHandle, time::Duration
};

use candle_core::{Tensor};
use cpal::traits::{DeviceTrait, HostTrait};
use crate::{audio::{get_devices, AppDevice, StreamState}, partial::{PARTIAL_MEL_BINS}, whisper::{
    WhichModel, WhisperParams,
}, whisper_word_align::AlignedWord};

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

struct DisplayMel {
    buffer: VecDeque<[u8; PARTIAL_MEL_BINS]>,
    texture: Option<egui::TextureHandle>,
    image: Option<egui::ColorImage>,
    min: f32,
    max: f32,
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
    worker: Option<Worker>,

    #[serde(skip)]
    devices: Vec<AppDevice>,

    #[serde(skip)]
    selected_device1: usize,
    selected_device2: usize,

    #[serde(skip)]
    selected_model: usize,

    #[serde(skip)]
    whisper_tx: mpsc::Sender<WhisperUpdate>,

    #[serde(skip)]
    level: VecDeque<f32>,

    #[serde(skip)]
    mel1: DisplayMel,
    // todo: remove mel2, it is less optimized (but always shows the full mel)
    #[serde(skip)]
    mel2: Tensor,

    #[serde(skip)]
    aligned_words: Vec<AlignedWord>,

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

    #[serde(skip)]
    status: String,
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
            worker: None,
            devices: devices,
            selected_device1: selected_device,
            selected_device2: selected_device,
            selected_model: 1,
            whisper_tx: tx,
            level: VecDeque::with_capacity(100),
            mel1: DisplayMel {
                buffer: VecDeque::with_capacity(100),
                texture: None,
                image: None,
                min: -10.0,
                max: -0.0,
            },
            mel2: Tensor::zeros((2, 3), candle_core::DType::F32, &candle_core::Device::Cpu).expect("Failed to create mel tensor"),
            aligned_words: vec![],
            autotype: false,
            partials: false,
            always_on_top: false,
            changed: false,
            tab: AppTab::Transcript,
            accuracy: 1,
            status: "Init".to_owned(),
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



#[derive(Debug, Clone)]
pub enum WhisperUpdate {
    Recording(bool),
    Transcribing(bool),
    Transcription(String),
    Alignment(Vec<AlignedWord>),
    Level(f32),
    Mel(Tensor), // [f32; (bin, frame)]
    Status(String),
    MelFrame(Vec<f32>),
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
            selected_device1,
            selected_device2,
            selected_model,
            worker: stream,
            whisper_tx,
            level,
            mel1,
            mel2,
            autotype,
            partials,
            accuracy,
            changed,
            aligned_words,
            status,
            ..
        } = self;
        
        // eframe will go to sleep when data is waiting.. this is a hack to keep it awake.
        // it would be better for the channel to call this when it has posted data.
        ctx.request_repaint_after(Duration::from_millis(100));

        // check for thread panics
        if let Some(worker) = stream {
            check_thread_error(&mut worker.whisper_thread);
        }

        // drain from_whisper channel
        loop {
            let whisper_update_result = from_whisper.try_recv();
            let span = tracing::span!(tracing::Level::TRACE, "whisper_update", ?whisper_update_result);
            let _enter = span.enter();
            match whisper_update_result {
                Ok(WhisperUpdate::Transcription(t)) => {
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
                Ok(WhisperUpdate::Alignment(a)) => {
                    *aligned_words = a;
                }
                Ok(WhisperUpdate::MelFrame(frame)) => {
                    //update_mel_buffer(&frame, mel1);
                }
                Ok(WhisperUpdate::Mel(m)) => {
                    //*mel2 = m;
                    tracing::debug!("App received mel spectrogram with shape: {:?}", mel2.shape());
                }
                Ok(WhisperUpdate::Status(s)) => {
                    *status = s;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => panic!("Whisper channel disconnected"),
            }
        }

        // Draw the UI
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.with_layout(
                egui::Layout::left_to_right(egui::Align::LEFT)
                    .with_main_wrap(true)
                    .with_cross_align(egui::Align::TOP),
                |ui| {
                    plot_level(level, ui);
                    
                    let started = stream.is_some();
                    let buttontext = if started { "Stop" } else { "Start" };
                    if ui.add(egui::Button::new(buttontext)).clicked() {
                        if started {
                            // stop the stream
                            *stream = None;
                        } else {
                            match start_listening(
                        whisper_tx,
                        &devices[*selected_device1],
                        &devices[*selected_device2],
                        WhisperParams {
                            accuracy: *accuracy,
                            model: WhichModel::from(*selected_model),
                            partials: *partials,
                        },
                    ) {
                        Ok(new_stream) => {
                            *stream = Some(new_stream);
                        }
                        Err(e) => {
                            tracing::error!("Failed to start listening: {}", e);
                            *stream = None
                        }
                    }
                        }
                    }

                    let source1 = egui::ComboBox::from_label("Sound device").show_index(
                        ui,
                        selected_device1,
                        devices.len(),
                        |i| devices[i].name.clone(),
                    );
                    if source1.changed() {
                        *stream = None;
                    }
                    let source2 = egui::ComboBox::from_label("Sound device 2").show_index(
                        ui,
                        selected_device2,
                        devices.len(),
                        |i| devices[i].name.clone(),
                    );
                    if source2.changed() {
                        *stream = None;
                    }
                    let model = egui::ComboBox::from_label("Model")
                        .selected_text(WhichModel::from(*selected_model).to_string())
                        .show_index(ui, selected_model, WhichModel::len(), |i| {
                            WhichModel::from(i).to_string()
                        });
                    if model.changed() {
                        *stream = None;
                    }
                },
            );
            ui.label(format!("Status: {}", status));
            let mel = ui.with_layout(
                egui::Layout::left_to_right(egui::Align::LEFT)
                    .with_main_wrap(true)
                    .with_cross_align(egui::Align::TOP), | ui| {
                        if let Err(e) = draw_mel2(mel2, mel1, ui) {
                            tracing::error!("Error drawing mel spectrogram: {}", e);
                        }
                    }
            );

            draw_aligned_words(ctx, aligned_words, ui, mel, mel1);

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
                        *stream = None;
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
                    let p = ui.checkbox(partials, "Partials").on_hover_text(
                        "Show partials as a block is dictated, erasing with the full model once it is done",
                    );
                    if p.changed() {
                        *stream = None;
                    }
                    if ui.button("Clear").clicked() {
                        text.clear();
                        // log the time as well as a message
                        tracing::info!("Cleared text, time: {}", chrono::Local::now());
                        *mel2 = Tensor::zeros(
                            (2, 3),
                            candle_core::DType::F32,
                            &candle_core::Device::Cpu,
                        ).expect("Failed to create mel tensor");
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

fn check_thread_error(join: &mut Option<JoinHandle<()>>) {
    if let Some(thread) = join.take() {
        if thread.is_finished() {
            if let Err(e) = thread.join() {
                if let Some(es) = (&e).downcast_ref::<&'static str>() {
                    tracing::error!("Thread panicked with error: {}", es);
                } else if let Some(es) = (&e).downcast_ref::<String>() {
                    tracing::error!("Thread panicked with error: {}", es);
                } else {
                    tracing::error!("Thread panicked with unknown error: {:?}", e);
                }
            }
        } else {
            *join = Some(thread);
        }
    }
}

fn draw_aligned_words(ctx: &egui::Context, aligned_words: &mut Vec<AlignedWord>, ui: &mut egui::Ui, mel: egui::InnerResponse<()>, display: &DisplayMel) {
    let mel_seconds = match &display.texture {
        Some(tex) => tex.size()[0] as f32 / 100.0, // mel is 100Hz (100 pixels per second)
        None => 0.0,
    };
    if mel_seconds < 0.1 {
        return;
    }

    const ALIGNED_ROWS: usize = 6; // spread words over n rows so that they don't overlap too much
    let rx = mel.response.rect;
    for (i, word) in aligned_words.iter().enumerate() {
        let row = (i%ALIGNED_ROWS) as f32 * 12.0;
        let word_pixels = (7 * word.word.len()) as f32;
        let seconds_to_pixels = rx.width() / mel_seconds;
        let rect = egui::Rect::from_min_max(
            egui::pos2(rx.left() + seconds_to_pixels * word.start as f32 , rx.top()+row),
            egui::pos2(rx.left() + seconds_to_pixels * word.end as f32 + word_pixels, rx.top() + 12.0 + row),
        );
        ui.painter().rect_filled( rect, 0.0, egui::Color32::from_rgb(0,0,0));
        ui.painter().text(
            rect.left_center(),
            egui::Align2::LEFT_CENTER,
            word.word.clone(),
            egui::TextStyle::Body.resolve(&ctx.style()),
            egui::Color32::from_rgb(
                (255) as u8, // even low probability words stay red, not black
                (word.probability * 255.0) as u8,
                (word.probability * 255.0) as u8,
            )
        );
    }
}

fn update_mel_buffer(
    frame: &Vec<f32>,
    mel: &mut DisplayMel,
) {
    mel.min = mel.min.min(frame.iter().cloned().fold(f32::INFINITY, f32::min));
    mel.max = mel.max.max(frame.iter().cloned().fold(f32::NEG_INFINITY, f32::max))+0.01;
    
    let bytes: Vec<u8> = frame
        .iter()
        .map(|&x| {
            let x = (x - mel.min) * (255.0 / (mel.max - mel.min));
            let x = x.clamp(0.0, 255.0);
            x as u8
        })
        .collect();

    let mut arr = [0u8; PARTIAL_MEL_BINS];
    let len = bytes.len().min(PARTIAL_MEL_BINS);
    arr[..len].copy_from_slice(&bytes[..len]);

    if mel.buffer.len() >= 5 * 100 {
        mel.buffer.pop_front();
    }
    mel.buffer.push_back(arr);
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

fn draw_mel1(mel: &mut DisplayMel, ui: &mut egui::Ui) {
    let DisplayMel {
        buffer,
        texture,
        image,
        min: _, max:_,
    } = mel;

    let image = if let Some(image) = image {
        image
    } else {
        let black = egui::Color32::from_black_alpha(0);
        let img = egui::ColorImage::new( [PARTIAL_MEL_BINS, 100], black);
        *image = Some(img);
        image.as_mut().unwrap()
    };

    let texture = if let Some(texture) = texture {
        texture
    } else {
        let tex = ui.ctx().load_texture(
            "mel_spectrogram",
            image.clone(),
            egui::TextureOptions::default(),
        );
        *texture = Some(tex);
        texture.as_mut().unwrap()
    };

    let xmax = buffer.len();
    let mut pixels: Vec<egui::Color32> = vec![egui::Color32::from_gray(0); PARTIAL_MEL_BINS * xmax];
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
            .fit_to_exact_size(egui::vec2(buffer.len() as f32 * 4.0, PARTIAL_MEL_BINS as f32)),
    );
}

fn draw_mel2(mel2: &mut Tensor, display: &mut DisplayMel, ui: &mut egui::Ui) -> Result<(), anyhow::Error> {
    let shape = mel2.shape();
    if shape.rank() != 2 {
        anyhow::bail!("unexpected rank, expected: 2, got: {} ({:?})", shape.rank(), shape.dims());
    }

    let n_frames = shape.dims()[1];
    if n_frames < 10 {
        return Ok(());
    }

    let mut mel_image = egui::ColorImage::new(
        [n_frames, PARTIAL_MEL_BINS],
        egui::Color32::from_black_alpha(0),
    );
    let mel_min = mel2.min_all()?.to_scalar::<f32>()?;
    let mel_max = mel2.max_all()?.to_scalar::<f32>()? + 0.01;

    let mel_data = mel2.to_vec2::<f32>()?;
    for f in 0..n_frames {
        for b in 0..PARTIAL_MEL_BINS {
            let value = mel_data[b][f];
            let color_value = ((value - mel_min) / (mel_max - mel_min) * 255.0)
                .clamp(0.0, 255.0) as u8;
            mel_image.pixels[b * n_frames + f] =
                egui::Color32::from_rgb(color_value, color_value, color_value);
        }
    }

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

// Needed to hold a handle to keep the audio stream alive
struct Worker {
    audio: StreamState,
    audio2: Option<StreamState>, // for the second device, if used
    // Keeping thread handles allows us to look for panics in the threads
    partial_thread: Option<JoinHandle<()>>,
    whisper_thread: Option<JoinHandle<()>>,
}

fn start_listening(
    app: &Sender<WhisperUpdate>,
    app_device: &AppDevice,
    app_device2: &AppDevice,
    params: WhisperParams,
) -> Result<Worker, anyhow::Error> {
    // cleanup: when the StreamState is dropped, the audio thread will stop, closing its sender, which will close the receiver in the chained threads.
    let (partial, partial_tx) = if params.partials {
        let (partial_tx, partial_rx) = mpsc::channel();
        (Some(crate::partial::start_partial_thread(app.clone(), partial_rx)?), Some(partial_tx))
    } else {
        (None, None)
    };

    let (filtered_tx, filtered_rx) = mpsc::channel();
        
    let stream = crate::audio::start_audio_thread(app.clone(), app_device, filtered_tx.clone(), partial_tx)?;

    let audio2 = if app_device2.name != app_device.name {
        Some(crate::audio::start_audio_thread(app.clone(), app_device2, filtered_tx, None)?)
    } else {
        None
    };

    let whisper = crate::whisper::start_whisper_thread(app.clone(), filtered_rx, params)?;

    Ok(Worker{audio: stream, audio2: audio2, 
        partial_thread: partial,
        whisper_thread: Some(whisper),
    })
}
