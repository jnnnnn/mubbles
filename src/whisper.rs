use std::sync::{
    mpsc::{Receiver, Sender},
    Arc,
};

use anyhow::{Error as E, Result};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use candle_transformers::models::whisper::{self as m, Config};

use crate::{app::WhisperUpdate, whisper_model::{token_id, Decoder, Model}};


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

#[derive(Default)]
pub struct DisplayMel {
    pub mel: Arc<Vec<f32>>,
    pub num_bins: usize,
    pub num_frames: usize,
}

pub struct WhisperParams {
    pub accuracy: usize, // 1 for greedy, more for beam search
    pub model: WhichModel,
}

pub struct WhisperContext {
    decoder: crate::whisper_model::Decoder,
    device: candle_core::Device,
    config: Config,
    mel_filters: Vec<f32>,
    previous_content_tokens: Vec<u32>, // Added to store context
}

pub fn load_whisper_model(model: WhichModel) -> Result<WhisperContext> {
    tracing::info!("Loading whisper model: {:?}", model);
    let device = candle_core::Device::new_cuda(0)?;
    let (model_id, revision) = model.model_and_revision();
    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.to_string(),
        ));
        let config = repo.get("config.json")?;
        let tokenizer = repo.get("tokenizer.json")?;
        let model = repo.get("model.safetensors")?;
        (config, tokenizer, model)
    };
    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
    tracing::info!("Config: {:?}", config);
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let active_model = {
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)? };
        Model::Normal(m::model::Whisper::load(&vb, config.clone())?)
    };

    let mut decoder = Decoder::new(
        active_model,
        tokenizer.clone(),
        0,
        &device,
        None,
        None,
        false,
        false,
        crate::whisper_model::get_alignment_heads(model, &config),
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
    filtered_rx: Receiver<Vec<f32>>,
    params: WhisperParams,
) {
    let result = std::thread::Builder::new()
        .name("whisper".to_string())
        .spawn(move || match whisper_loop(app, filtered_rx, params) {
            Ok(_) => tracing::info!("Whisper thread finished successfully"),
            Err(e) => tracing::error!("Whisper thread failed: {:?}", e),
        });
    match result {
        Ok(_) => tracing::info!("Whisper thread started"),
        Err(e) => tracing::error!("Failed to start whisper thread: {:?}", e),
    }
}

fn whisper_loop(
    app: Sender<WhisperUpdate>,
    filtered_rx: Receiver<Vec<f32>>,
    params: WhisperParams,
) -> Result<(), anyhow::Error> {
    app.send(WhisperUpdate::Status(
        "Loading whisper model...".to_string(),
    ))?;
    let mut ctx: WhisperContext = load_whisper_model(params.model)?;
    app.send(WhisperUpdate::Status("Model loaded".to_string()))?;
    loop {
        // first recv needs to be blocking to prevent the thread from spinning
        let mut aggregated_data = match filtered_rx.recv() {
            Ok(data) => data,
            Err(_) => {
                tracing::info!("Filtered stream closed");
                return Ok(());
            }
        };
        // because whisper processes audio in chunks of 30 seconds (and takes a while to do so), it's
        // worth aggregating several chunks of audio before sending it to whisper (if they are available)
        // todo: 48000 is hardcoded here, see how candle whisper example handles it
        while aggregated_data.len() < 48000 * 30 {
            match filtered_rx.try_recv() {
                Ok(data) => aggregated_data.extend(data),
                Err(_) => break,
            }
        }
        whisperize(&mut ctx, &aggregated_data, &app)?;
    }
}

fn whisperize(
    state: &mut WhisperContext,
    resampled: &[f32],
    app: &Sender<WhisperUpdate>,
) -> Result<(), anyhow::Error> {
    let start = std::time::Instant::now();

    app.send(WhisperUpdate::Transcribing(true))
        .expect("Failed to send transcribing update");

    app.send(WhisperUpdate::Status("Levels...".to_string()))?;

    let mel_start = std::time::Instant::now();
    app.send(WhisperUpdate::Status("Mel spectrogram...".to_string()))?;
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

    app.send(WhisperUpdate::Mel(DisplayMel {
        mel: arcmel.clone(),
        num_bins,
        num_frames: num_mel_frames,
    }))
    .expect("Failed to send mel update");

    app.send(WhisperUpdate::Status(
        "Running Whisper decoder...".to_string(),
    ))?;

    let (segments_results, last_segment_content_tokens) =
        state.decoder.run(&mel_tensor, None, None)?;
    state.previous_content_tokens = last_segment_content_tokens;

    for segment in segments_results.iter() {
        let text = &segment.dr.text;
        let start_time_s = segment.start;
        let end_time_s = segment.start + segment.duration;

        app.send(WhisperUpdate::Transcription(format!(
            "[{:.2}s -> {:.2}s] {}",
            start_time_s, end_time_s, text
        )))?;
    }

    app.send(WhisperUpdate::Transcribing(false))?;
    let duration_secs = start.elapsed().as_secs_f64();
    let input_duration_secs = resampled.len() as f64 / 16000.0; // Assuming 16kHz
    tracing::info!(
        "Whisper processing took {:.2}s for {:.2}s audio (RTF: {:.2}x)",
        duration_secs,
        input_duration_secs,
        duration_secs / input_duration_secs
    );
    Ok(())
}
