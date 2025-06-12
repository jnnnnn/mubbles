use std::{
    collections::VecDeque,
    sync::mpsc::{self, Sender, TryRecvError},
    time::Duration,
};

use candle_core::Tensor;
use cpal::traits::{DeviceTrait, HostTrait};

use crate::{audio::{get_devices, AppDevice, StreamState}, partial::{PARTIAL_LEN, PARTIAL_MEL_BINS}, whisper::{
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
    selected_device: usize,

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
            selected_device: selected_device,
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
            selected_device,
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

        // drain from_whisper channel
        loop {
            let whisper_update_result = from_whisper.try_recv();
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
                    *mel2 = m;
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
                    

                    let source = egui::ComboBox::from_label("Sound device").show_index(
                        ui,
                        selected_device,
                        devices.len(),
                        |i| devices[i].name.clone(),
                    );
                    if source.changed() {
                        let device = &devices[*selected_device];
                        *stream = start_listening(
                            whisper_tx,
                            device,
                            WhisperParams {
                                accuracy: *accuracy,
                                model: WhichModel::from(*selected_model),
                                partials: *partials,
                            },
                        );
                    }
                    let model = egui::ComboBox::from_label("Model")
                        .selected_text(WhichModel::from(*selected_model).to_string())
                        .show_index(ui, selected_model, WhichModel::len(), |i| {
                            WhichModel::from(i).to_string()
                        });
                    if model.changed() {
                        *stream = start_listening(
                            whisper_tx,
                            &devices[*selected_device],
                            WhisperParams {
                                accuracy: *accuracy,
                                model: WhichModel::from(*selected_model),
                                partials: *partials,
                            },
                        );
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

            draw_aligned_words(ctx, aligned_words, ui, mel);

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
                        *stream = start_listening(
                            whisper_tx,
                            &devices[*selected_device],
                            WhisperParams {
                                accuracy: *accuracy,
                                model: WhichModel::from(*selected_model),
                                partials: *partials,
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

const MEL_SECONDS: usize = PARTIAL_LEN;

fn draw_aligned_words(ctx: &egui::Context, aligned_words: &mut Vec<AlignedWord>, ui: &mut egui::Ui, mel: egui::InnerResponse<()>) {
    let rx = mel.response.rect;
    for (i, word) in aligned_words.iter().enumerate() {
        let row = (i%10) as f32 * 12.0;
        let word_pixels = (10 * word.word.len()) as f32;
        let seconds_to_pixels = rx.width() / MEL_SECONDS as f32;
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
    mel.max = mel.max.max(frame.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    
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

    if mel.buffer.len() >= MEL_SECONDS * 100 {
        mel.buffer.pop_front();
    }
    mel.buffer.push_back(arr);
}

fn overwrite_mel_buffer(
    display: &mut DisplayMel,
    mel: Vec<f32>,
) {
    tracing::debug!("Overwrite mel buffer with {} frames", mel.len() / PARTIAL_MEL_BINS);
    display.min = f32::INFINITY;
    display.max = f32::NEG_INFINITY;
    let n_frames = mel.len() / PARTIAL_MEL_BINS;
    let mut frame = vec![0f32; PARTIAL_MEL_BINS];
    for f in 0..n_frames {
        for b in 0..PARTIAL_MEL_BINS {
            frame[b] = mel[b * n_frames + f];
        }
        update_mel_buffer(&frame, display);
    }
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

    let n_frames = shape.dims()[1]; // Access the second dimension
    if n_frames < 10 {
        tracing::warn!("Mel spectrogram has too few frames: {}", n_frames);
        return Ok(());
    }

    let mut mel_image = egui::ColorImage::new(
        [n_frames, PARTIAL_MEL_BINS],
        egui::Color32::from_black_alpha(0),
    );
    let mel_min = mel2.min_all()?.to_scalar::<f32>()?;
    let mel_max = mel2.max_all()?.to_scalar::<f32>()?;

    let mel_data = mel2.to_vec2::<f32>()?; // Adjusted to handle 2D tensors
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
}

fn start_listening(
    app: &Sender<WhisperUpdate>,
    app_device: &AppDevice,
    params: WhisperParams,
) -> Option<Worker> {
    let result = crate::audio::start_audio_thread(app.clone(), app_device);
    
    if result.is_err() {
        tracing::error!("Failed to start audio thread");
        return None;
    }
    let (stream, rx, rx_partial) = result.unwrap();

    if params.partials {
        crate::partial::start_partial_thread(app.clone(), rx_partial);
    }
    
    crate::whisper::start_whisper_thread(
        app.clone(),
        rx,
        params,
    );
    Some(Worker{audio: stream})
}
