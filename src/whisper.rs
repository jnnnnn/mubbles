use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, Sender},
        Arc,
    },
    thread::JoinHandle,
};

use anyhow::{Error as E, Result};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use candle_transformers::models::whisper::{self as m, Config};

use crate::{
    app::WhisperUpdate,
    audio::PcmAudio,
    mel::unpad_mel,
    whisper_model::{token_id, Decoder, Model},
};

#[derive(Clone, Copy, Debug)]
enum Task {
    Transcribe,
    Translate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhichModel {
    Tiny,
    TinyEn,
    Base,
    BaseEn,
    Small,
    SmallEn,
    Medium,
    MediumEn,
    Large,
    LargeV2,
    LargeV3,
    LargeV3Turbo,
    DistilMediumEn,
    DistilLargeV2,
    DistilLargeV3,
}

impl WhichModel {
    fn is_multilingual(&self) -> bool {
        match self {
            Self::Tiny
            | Self::Base
            | Self::Small
            | Self::Medium
            | Self::Large
            | Self::LargeV2
            | Self::LargeV3
            | Self::LargeV3Turbo
            | Self::DistilLargeV2
            | Self::DistilLargeV3 => true,
            Self::TinyEn | Self::BaseEn | Self::SmallEn | Self::MediumEn | Self::DistilMediumEn => {
                false
            }
        }
    }

    fn model_and_revision(&self) -> (&'static str, &'static str) {
        match self {
            Self::Tiny => ("openai/whisper-tiny", "main"),
            Self::TinyEn => ("openai/whisper-tiny.en", "refs/pr/15"),
            Self::Base => ("openai/whisper-base", "refs/pr/22"),
            Self::BaseEn => ("openai/whisper-base.en", "refs/pr/13"),
            Self::Small => ("openai/whisper-small", "main"),
            Self::SmallEn => ("openai/whisper-small.en", "refs/pr/10"),
            Self::Medium => ("openai/whisper-medium", "main"),
            Self::MediumEn => ("openai/whisper-medium.en", "main"),
            Self::Large => ("openai/whisper-large", "refs/pr/36"),
            Self::LargeV2 => ("openai/whisper-large-v2", "refs/pr/57"),
            Self::LargeV3 => ("openai/whisper-large-v3", "main"),
            Self::LargeV3Turbo => ("openai/whisper-large-v3-turbo", "main"),
            Self::DistilMediumEn => ("distil-whisper/distil-medium.en", "main"),
            Self::DistilLargeV2 => ("distil-whisper/distil-large-v2", "main"),
            Self::DistilLargeV3 => ("distil-whisper/distil-large-v3", "main"),
        }
    }

    pub fn len() -> usize {
        Self::iter().count()
    }

    pub fn from(index: usize) -> Self {
        Self::iter().nth(index).expect("Invalid index")
    }

    pub fn index(&self) -> usize {
        Self::iter()
            .position(|x| x == *self)
            .expect("Invalid model")
    }

    pub fn to_string(&self) -> String {
        return format!("{:?}", self);
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        [
            Self::Tiny,
            Self::TinyEn,
            Self::Base,
            Self::BaseEn,
            Self::Small,
            Self::SmallEn,
            Self::Medium,
            Self::MediumEn,
            Self::Large,
            Self::LargeV2,
            Self::LargeV3,
            Self::LargeV3Turbo,
            Self::DistilMediumEn,
            Self::DistilLargeV2,
            Self::DistilLargeV3,
        ]
        .into_iter()
    }
}

const FIRST_TIMESTAMP_TOKEN: usize = 50364; // <|0.00|>

pub struct WhisperParams {
    pub accuracy: usize, // 1 for greedy, more for beam search
    pub model: WhichModel,
    pub partials: bool,
}

pub struct WhisperContext {
    pub decoder: crate::whisper_model::Decoder,
    pub device: candle_core::Device,
    pub config: Config,
    pub mel_filters: Vec<f32>,
    pub previous_content_tokens: Vec<u32>, // Added to store context
}

impl std::fmt::Debug for WhisperContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhisperContext").finish()
    }
}

struct MyProgress {
    status_fn: Box<dyn Fn(String) + Send>,
    total_size: usize,
    current: usize,
    fname: String,
}
impl MyProgress {
    fn new(status_fn: Box<dyn Fn(String) + Send>) -> Self {
        MyProgress {
            status_fn,
            total_size: 0,
            current: 0,
            fname: String::new(),
        }
    }
}
impl hf_hub::api::Progress for MyProgress {
    fn init(&mut self, size: usize, filename: &str) {
        self.total_size = size;
        self.fname = filename.to_string();
    }
    fn update(&mut self, size: usize) {
        self.current += size;
        let percentage = (self.current as f64 / self.total_size as f64) * 100.0;

        (self.status_fn)(format!(
            "Downloading {}: {} MiB ({percentage:.1}%)",
            self.fname,
            self.current / (1024 * 1024)
        ));
    }
    fn finish(&mut self) {}
}

fn get_with_progress(
    repo: hf_hub::Repo,
    status_fn: Box<dyn Fn(String) + Send>,
    filename: &str,
) -> Result<std::path::PathBuf, E> {
    // unfortunately, api::download_with_progress does not check cache first, so we do it manually
    // this is made harder by ApiRepo not exposing the cache, so we have to use the Cache directly
    let cache = hf_hub::Cache::default();
    let cached = cache.repo(repo.clone()).get(filename);
    let path = if let Some(path) = cached {
        tracing::info!("Using cached file: {}", path.display());
        path
    } else {
        tracing::info!("Downloading file: {}", filename);
        Api::new()?
            .repo(repo)
            .download_with_progress(filename, MyProgress::new(status_fn))?
    };
    Ok(path)
}

fn get_device() -> Result<candle_core::Device> {
    Ok(candle_core::Device::cuda_if_available(0)?)
}

pub fn load_whisper_model(
    model: WhichModel,
    status_fn: impl Fn(String) + Send + 'static,
) -> Result<WhisperContext> {
    tracing::info!("Loading whisper model: {:?}", model);
    let device = get_device()?;
    let (model_id, revision) = model.model_and_revision();
    //quantized let (model_id, revision) = ("lmz/candle-whisper", "main");
    // todo: find quantized models for not just tiny ? or quantize them ourselves? implement the script that does quantization ?
    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let inner_repo =
            Repo::with_revision(model_id.to_string(), RepoType::Model, revision.to_string());
        let repo_with_api = api.repo(inner_repo.clone());
        let config = repo_with_api.get("config.json")?;
        let tokenizer = repo_with_api.get("tokenizer.json")?;
        let model = get_with_progress(inner_repo, Box::new(status_fn), "model.safetensors")?;
        (config, tokenizer, model)
    };
    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
    tracing::info!("Config: {:?}", config);
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let mdl = {
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)? };
        Model::Normal(m::model::Whisper::load(&vb, config.clone())?)
        //quantized    unsafe { candle_transformers::models::whisper::VarBuilder::from_gguf(&[weights_filename], m::DTYPE, &device)? };
        //quantized Model::Quantized(m::quantized_model::Whisper::load(&vb, config.clone())?)
    };

    let heads = crate::whisper_model::get_alignment_heads(model, &config);
    let tokenizer1 = tokenizer.clone();
    let timestamps = match model {
        WhichModel::Tiny
        | WhichModel::TinyEn
        | WhichModel::Base
        | WhichModel::BaseEn
        | WhichModel::Small
        | WhichModel::SmallEn
        | WhichModel::Medium
        | WhichModel::MediumEn => false,
        _ => true,
    };
    let mut decoder = Decoder::new(
        mdl, tokenizer1, 0, &device, None, None, timestamps, false, heads,
    )?;

    decoder.set_language_token(if model.is_multilingual() {
        Some(token_id(&tokenizer, "<|en|>")?)
    } else {
        None
    });

    let mel_bytes = match config.num_mel_bins {
        80 => include_bytes!("melfilters.bytes").as_slice(),
        128 => include_bytes!("melfilters128.bytes").as_slice(),
        nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);

    tracing::info!("Model loaded");
    Ok(WhisperContext {
        decoder,
        device,
        config,
        mel_filters,
        previous_content_tokens: Vec::new(), // Initialize context
    })
}

// thread closes when the receiver is closed
pub fn start_whisper_thread(
    app: Sender<WhisperUpdate>,
    filtered_rx: Receiver<PcmAudio>,
    params: WhisperParams,
    paused: Arc<AtomicBool>,
) -> Result<JoinHandle<()>, anyhow::Error> {
    let result = std::thread::Builder::new()
        .name("whisper".to_string())
        .spawn(move || match whisper_loop(app, filtered_rx, params, paused) {
            Ok(_) => tracing::info!("Whisper thread finished successfully"),
            Err(e) => tracing::error!("Whisper thread failed: {:?}", e),
        });
    Ok(result?)
}

fn whisper_loop(
    app: Sender<WhisperUpdate>,
    filtered_rx: Receiver<PcmAudio>,
    params: WhisperParams,
    paused: Arc<AtomicBool>,
) -> Result<(), anyhow::Error> {
    app.send(WhisperUpdate::Status(
        "Loading whisper model...".to_string(),
    ))?;
    let app_status = app.clone();
    let mut ctx: Option<WhisperContext> = Some(load_whisper_model(params.model, move |s| {
        app_status.send(WhisperUpdate::Status(s)).ok();
    })?);
    app.send(WhisperUpdate::Status("Model loaded".to_string()))?;
    loop {
        // When paused, drop the model and stop reading audio (channel buffers it)
        while paused.load(Ordering::Relaxed) {
            if ctx.is_some() {
                tracing::info!("Whisper paused — dropping model to free GPU memory");
                ctx = None;
                app.send(WhisperUpdate::Status("Model unloaded (GPU freed)".to_string()))?;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        ensure_model_loaded(&mut ctx, &params, &app)?;

        // first recv needs to be blocking to prevent the thread from spinning
        let PcmAudio {
            data: mut aggregated,
            sample_rate,
        } = match filtered_rx.recv() {
            Ok(pcm) => pcm,
            Err(_) => {
                tracing::info!("Filtered stream closed");
                return Ok(());
            }
        };
        // Aggregate up to 30s of audio before processing
        while aggregated.len() < sample_rate * 30 {
            match filtered_rx.try_recv() {
                Ok(pcm) => aggregated.extend(pcm.data),
                Err(_) => break,
            }
        }
        let result = whisperize(ctx.as_mut().unwrap(), &aggregated, &app);
        if !result.is_ok() {
            tracing::error!("Failed to perform partial transcription: {:?}", result);
        }
    }
}

fn ensure_model_loaded(
    ctx: &mut Option<WhisperContext>,
    params: &WhisperParams,
    app: &Sender<WhisperUpdate>,
) -> Result<(), anyhow::Error> {
    if ctx.is_none() {
        app.send(WhisperUpdate::Status("Reloading whisper model...".to_string()))?;
        let app_status = app.clone();
        *ctx = Some(load_whisper_model(params.model, move |s| {
            app_status.send(WhisperUpdate::Status(s)).ok();
        })?);
        app.send(WhisperUpdate::Status("Model reloaded".to_string()))?;
    }
    Ok(())
}

#[tracing::instrument(skip(state, resampled, app))]
pub fn whisperize(
    state: &mut WhisperContext,
    resampled: &[f32],
    app: &Sender<WhisperUpdate>,
) -> Result<(), anyhow::Error> {
    if resampled.is_empty() {
        tracing::warn!("Received empty audio data, skipping transcription");
        return Ok(());
    }

    app.send(WhisperUpdate::Transcribing(true))
        .expect("Failed to send transcribing update");

    app.send(WhisperUpdate::Status("Levels...".to_string()))?;

    let mel_start = std::time::Instant::now();
    app.send(WhisperUpdate::Status("Mel spectrogram...".to_string()))?;
    // the mel pads out to 30s; keep track of how much actual audio we have for initializing the alignment calculation

    let mel_raw = crate::mel::pcm_to_mel(state.config.num_mel_bins, &resampled, &state.mel_filters);

    let arcmel = Arc::new(mel_raw);
    let mel_duration_secs = mel_start.elapsed().as_secs_f32();
    tracing::info!(
        "Mel spectrogram generation took {:.2} seconds",
        mel_duration_secs
    );

    let mel_len = arcmel.len();
    let num_bins = state.config.num_mel_bins;
    let num_mel_frames = mel_len / num_bins;
    // shape typically 1 batch, 80 frequency bins, 3000 10ms frames
    let mel_tensor = Tensor::from_slice(
        arcmel.as_slice(),
        (1, num_bins, num_mel_frames),
        &state.device,
    )?;
    app.send(WhisperUpdate::Mel(
        unpad_mel(mel_tensor.clone())?.squeeze(0)?,
    ))?;

    app.send(WhisperUpdate::Status(
        "Running Whisper decoder...".to_string(),
    ))?;

    let decode_start = std::time::Instant::now();

    let per_token_callback = Some(|text: String| {
        app.send(WhisperUpdate::Status(text.clone())).unwrap_or_default();
    });

    let (segments_results, last_segment_content_tokens) =
        state.decoder.run(&mel_tensor, None, None, &per_token_callback)?;
    state.previous_content_tokens = last_segment_content_tokens;

    for segment in segments_results.iter() {
        let text = &segment.dr.text;
        for phrase in text {
            app.send(WhisperUpdate::Transcription(phrase.clone()))?;
        }
        app.send(WhisperUpdate::Alignment(segment.dr.alignment.clone()))?;
        tracing::info!("Whisper segment: {:?}", segment.dr);

        #[cfg(debug_assertions)]
        if let Some(t) = segment.dr.text.first() {
            if t == ".com" {
                // write out wav file for testing. use timestamp to avoid overwriting
                let timestamp = chrono::Utc::now().format("%Y%m%d%H%M%S").to_string();
                let wav_path = std::path::PathBuf::from(format!("test_{}.wav", timestamp));
                tracing::info!("Writing test audio to: {}", wav_path.display());
                let mut wav_file = hound::WavWriter::create(
                    wav_path,
                    hound::WavSpec {
                        channels: 1,
                        sample_rate: crate::mel::SAMPLE_RATE as u32,
                        bits_per_sample: 32,
                        sample_format: hound::SampleFormat::Float,
                    },
                )?;
                for sample in resampled.iter() {
                    wav_file.write_sample(*sample)?;
                }
                wav_file.finalize()?;

                // save the mel as well, using bitcode
                let mel_path = std::path::PathBuf::from(format!("test_{}.mel", timestamp));
                tracing::info!("Writing mel spectrogram to: {}", mel_path.display());
                let mut mel_file = std::fs::File::create(mel_path)?;
                let encoded = bitcode::encode(arcmel.as_slice());
                std::io::Write::write_all(&mut mel_file, &encoded)?;
            }
        }

        let elapsed = decode_start.elapsed().as_secs_f64();
        let n_tokens = segment.dr.tokens.len();
        app.send(WhisperUpdate::Status(format!(
            "Time per token: {:.2}ms. Realtime factor: {:.2}",
            if n_tokens > 0 {
                elapsed * 1000.0 / n_tokens as f64
            } else {
                0.0
            },
            if resampled.len() > 0 {
                resampled.len() as f64 / crate::mel::SAMPLE_RATE as f64 / elapsed
            } else {
                0.0
            }
        )))?;
    }

    app.send(WhisperUpdate::Transcribing(false))?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_com_hound_file_decoding() -> Result<(), anyhow::Error> {
        // load pcm from test_20250612074049.wav using hound
        let wav_path = std::path::PathBuf::from("test_20250612074049.wav");
        let mut wav_reader = hound::WavReader::open(wav_path).expect("Failed to open wav file");
        let mut pcm_audio = PcmAudio {
            data: Vec::new(),
            sample_rate: wav_reader.spec().sample_rate as usize,
        };
        for sample in wav_reader.samples::<f32>() {
            match sample {
                Ok(s) => pcm_audio.data.push(s),
                Err(e) => panic!("Failed to read sample: {:?}", e),
            }
        }
        tracing::info!("Loaded {} samples from wav file", pcm_audio.data.len());

        let (tx, rx) = std::sync::mpsc::channel();
        let tx2 = tx.clone();
        let mut state = load_whisper_model(WhichModel::Tiny, move |s| {
            tx2.send(WhisperUpdate::Status(s)).ok();
        })?;

        whisperize(&mut state, &pcm_audio.data, &tx)?;
        whisperize(&mut state, &pcm_audio.data, &tx)?;
        whisperize(&mut state, &pcm_audio.data, &tx)?;

        loop {
            // blocks until a message is received
            match rx.recv() {
                Ok(update) => match update {
                    WhisperUpdate::Transcription(text) => {
                        assert_eq!(text, "This is a test transcription for .com");
                    }
                    _ => {}
                },
                Err(_) => break,
            }
        }

        Ok(())
    }

    #[test]
    fn test_com_mel_file_decoding() -> Result<(), anyhow::Error> {
        // load mel from test_20250612074049.mel using bitcode
        let mel_path = std::path::PathBuf::from("test_20250612110040.mel");
        let mel_bytes = std::fs::read(mel_path)?;
        let mel_data: Vec<f32> = bitcode::decode(&mel_bytes)?;
        let mel_tensor = Tensor::from_slice(
            mel_data.as_slice(),
            (1, 80, mel_data.len() / 80),
            &candle_core::Device::cuda_if_available(0)?,
        )?;

        let mut state = load_whisper_model(WhichModel::Tiny, |_| {})?;

        let token_callback = Some(|_: String| { });
        let (segments, _) = state.decoder.run(&mel_tensor, None, None, &token_callback)?;
        assert_eq!(
            segments.first().unwrap().dr.text,
            vec!["This is a test transcription for .com"]
        );

        Ok(())
    }
}
