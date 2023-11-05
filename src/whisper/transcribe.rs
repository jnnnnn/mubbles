use std::sync::mpsc::{Receiver, Sender};

use candle::Tensor;
use candle_transformers::models::whisper::{self as m, audio, Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use crate::whisper::candle_example::{Decoder, Model, Task, WhichModel};

struct WhisperState {
    decoder: Decoder,
    device: candle::Device,
    mel_filters: Vec<f32>,
    language_token: Option<u32>,
}

// thisError
#[derive(Debug, thiserror::Error)]
enum WhisperError {
    #[error("Failed to load whisper model: {0}")]
    LoadModelIO(#[from] std::io::Error),
    #[error("Load: {0}")]
    LoadModelCandle(#[from] candle::Error),
    #[error("Model Download: {0}")]
    LoadModelHub(#[from] hf_hub::api::sync::ApiError),
    #[error("JSON: {0}")]
    LoadModelJson(#[from] serde_json::Error),
    #[error("Failed to load whisper model: {0}")]
    LoadModelOther(#[from] Box<dyn std::error::Error + Send + Sync>),
    // from anyhow
    #[error("Failed to load whisper model: {0}")]
    LoadModelAnyhow(#[from] anyhow::Error),
}

pub(crate) fn whisper_loop(
    app: Sender<super::WhisperUpdate>,
    filtered_rx: Receiver<Vec<f32>>,
    params: super::WhisperParams,
) {
    let mut state = match load_whisper_model(params) {
        Ok(ctx) => ctx,
        Err(err) => {
            tracing::error!("Failed to load whisper model: {}", err);
            return;
        }
    };
    loop {
        // first recv needs to be blocking to prevent the thread from spinning
        let mut aggregated_data = match filtered_rx.recv() {
            Ok(data) => data,
            Err(_) => {
                tracing::info!("Filtered stream closed");
                return;
            }
        };
        // because whisper processes audio in chunks of 30 seconds (and takes a while to do so), it's
        // worth aggregating several chunks of audio before sending it to whisper (if they are available)
        loop {
            match filtered_rx.try_recv() {
                Ok(data) => aggregated_data.extend(data),
                Err(_) => break,
            }
        }
        app.send(super::WhisperUpdate::BufferSize(
            aggregated_data.len() / 16000,
        ))
        .expect("Failed to send update");
        app.send(super::WhisperUpdate::Transcribing(true))
            .expect("Failed to send update");
        if let Err(err) = whisperize(&mut state, &aggregated_data, &app) {
            tracing::error!("Failed to transcribe: {}", err);
        }
        app.send(super::WhisperUpdate::Transcribing(false))
            .expect("Failed to send update");
    }
}

fn whisperize(
    state: &mut WhisperState,
    pcm_data: &[f32],
    app: &Sender<super::WhisperUpdate>,
) -> Result<(), String> {
    // record the start time so that we can measure how long it takes to generate the mel
    let start = std::time::Instant::now();
    let mel = audio::pcm_to_mel(&pcm_data, &state.mel_filters);
    let mel_len = mel.len();
    tracing::info!(
        "Mel generation for {:.0}s of audio took {:?}",
        pcm_data.len() / 16000,
        start.elapsed()
    );
    let mel = match Tensor::from_vec(mel, (1, m::N_MELS, mel_len / m::N_MELS), &state.device) {
        Ok(mel) => mel,
        Err(err) => {
            return Err(format!("Failed to create mel tensor: {}", err.to_string()));
        }
    };

    let start = std::time::Instant::now();
    let result = state.decoder.run(&mel);
    tracing::info!("Decoder run took {:?}", start.elapsed());
    match result {
        Err(err) => {
            return Err(format!("Failed to run decoder: {}", err.to_string()));
        }
        Ok(result) => {
            for segment in result.iter() {
                match app.send(super::WhisperUpdate::Transcript(segment.dr.text.clone())) {
                    Ok(_) => (),
                    Err(err) => {
                        return Err(format!("Failed to send transcription: {}", err.to_string()));
                    }
                }
            }
        }
    }
    Ok(())
}

fn load_whisper_model(params: super::WhisperParams) -> Result<WhisperState, WhisperError> {
    let device = candle::Device::cuda_if_available(0)?;
    tracing::info!(
        "Using {}",
        match device {
            candle::Device::Cpu => "CPU",
            candle::Device::Cuda(_) => "CUDA",
        }
    );
    let (model_id, revision) = if params.quantized {
        ("lmz/candle-whisper", "main")
    } else {
        params.model.model_and_revision()
    };

    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            model_id.to_owned(),
            RepoType::Model,
            revision.to_owned(),
        ));

        let (config, tokenizer, model) = if params.quantized {
            let ext = match params.model {
                WhichModel::TinyEn => "tiny-en",
                WhichModel::Tiny => "tiny",
                _ => unimplemented!("no quantized support for {:?}", params.model),
            };
            (
                repo.get(&format!("config-{ext}.json"))?,
                repo.get(&format!("tokenizer-{ext}.json"))?,
                repo.get(&format!("model-{ext}-q80.gguf"))?,
            )
        } else {
            (
                repo.get("config.json")?,
                repo.get("tokenizer.json")?,
                repo.get("model.safetensors")?,
            )
        };
        (config, tokenizer, model)
    };
    let tokenizer = Tokenizer::from_file(tokenizer_filename)?;

    let mel_bytes = include_bytes!("melfilters.bytes");
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);

    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
    let model = if params.quantized {
        let vb =
            candle_transformers::quantized_var_builder::VarBuilder::from_gguf(&weights_filename)?;
        Model::Quantized(m::quantized_model::Whisper::load(&vb, config)?)
    } else {
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[weights_filename],
                candle::DType::F32,
                &device,
            )?
        };
        Model::Normal(m::model::Whisper::load(&vb, config)?)
    };

    // Detected language: <|en|> u32:50259
    let language_token = match params.model.is_multilingual() {
        true => Some(50259),
        false => None,
    };
    let dc = Decoder::new(
        model,
        tokenizer,
        0,
        &device,
        language_token,
        Some(Task::Transcribe),
        false,
        false,
    )?;
    Ok(WhisperState {
        decoder: dc,
        device,
        mel_filters,
        language_token,
    })
}
