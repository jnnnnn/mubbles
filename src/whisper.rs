use std::{
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    SupportedStreamConfig,
};
use rubato::Resampler;

use anyhow::{Error as E, Result};
use candle_core::{IndexOp, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use hf_hub::{api::sync::Api, Repo, RepoType};
use rand::{distr::Distribution, SeedableRng};
use tokenizers::Tokenizer;

use candle_transformers::models::whisper::{self as m, audio, Config};
use crate::mel::log_mel_spectrogram; // Import the new log_mel_spectrogram function

pub enum Model {
    Normal(m::model::Whisper),
    Quantized(m::quantized_model::Whisper),
}

impl Model {
    pub fn config(&self) -> &Config {
        match self {
            Self::Normal(m) => &m.config,
            Self::Quantized(m) => &m.config,
        }
    }

    pub fn encoder_forward(&mut self, x: &Tensor, flush: bool) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(m) => m.encoder.forward(x, flush),
            Self::Quantized(m) => m.encoder.forward(x, flush),
        }
    }

    pub fn decoder_forward(
        &mut self,
        x: &Tensor,
        xa: &Tensor,
        flush: bool,
    ) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.forward(x, xa, flush),
            Self::Quantized(m) => m.decoder.forward(x, xa, flush),
        }
    }

    pub fn decoder_final_linear(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.final_linear(x),
            Self::Quantized(m) => m.decoder.final_linear(x),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct DecodingResult {
    tokens: Vec<u32>,
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Segment {
    start: f64,
    duration: f64,
    dr: DecodingResult,
}

struct Decoder {
    model: Model,
    rng: rand::rngs::StdRng,
    task: Option<Task>,
    timestamps: bool,
    verbose: bool,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
}

impl Decoder {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        device: &candle_core::Device,
        language_token: Option<u32>,
        task: Option<Task>,
        timestamps: bool,
        verbose: bool,
    ) -> Result<Self> {
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        // Suppress the notimestamps token when in timestamps mode.
        // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
        let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if model.config().suppress_tokens.contains(&i)
                    || timestamps && i == no_timestamps_token
                {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(&tokenizer, m::TRANSLATE_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok());
        let no_speech_token = match no_speech_token {
            None => anyhow::bail!("unable to find any non-speech token"),
            Some(n) => n,
        };
        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            tokenizer,
            task,
            timestamps,
            verbose,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            language_token,
            no_timestamps_token,
        })
    }

    fn decode(&mut self, mel: &Tensor, t: f64) -> Result<DecodingResult> {
        let model = &mut self.model;
        let audio_features = model.encoder_forward(mel, true)?;
        if self.verbose {
            println!("audio features: {:?}", audio_features.dims());
        }
        let sample_len = model.config().max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        match self.task {
            None | Some(Task::Transcribe) => tokens.push(self.transcribe_token),
            Some(Task::Translate) => tokens.push(self.translate_token),
        }
        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            // The model expects a batch dim but this inference loop does not handle
            // it so we add it at this point.
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = model.decoder_forward(&tokens_t, &audio_features, i == 0)?;

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                let logits = model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = model
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;
            // TODO: Besides suppress tokens, we should apply the heuristics from
            // ApplyTimestampRules, i.e.:
            // - Timestamps come in pairs, except before EOT.
            // - Timestamps should be non-decreasing.
            // - If the sum of the probabilities of timestamps is higher than any other tokens,
            //   only consider timestamps when sampling.
            // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L439
            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = rand::distr::weighted::WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let prob = softmax(&logits, candle_core::D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == self.eot_token || tokens.len() > model.config().max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    fn decode_with_fallback(&mut self, segment: &Tensor) -> Result<DecodingResult> {
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult> = self.decode(segment, t);
            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                        && dr.avg_logprob < m::LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    println!("Error running at {t}: {err}")
                }
            }
        }
        unreachable!()
    }

    fn run(&mut self, mel: &Tensor, _times: Option<(f64, f64)>) -> Result<Vec<Segment>> {
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];
        while seek < content_frames {
            let time_offset = (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration = (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment)?;
            seek += segment_size;
            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                println!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };
            segments.push(segment)
        }
        Ok(segments)
    }

    fn set_language_token(&mut self, language_token: Option<u32>) {
        self.language_token = language_token;
    }

    #[allow(dead_code)]
    fn reset_kv_cache(&mut self) {
        match &mut self.model {
            Model::Normal(m) => m.reset_kv_cache(),
            Model::Quantized(m) => m.reset_kv_cache(),
        }
    }

    fn model(&mut self) -> &mut Model {
        &mut self.model
    }
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> candle_core::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => candle_core::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

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
        Self::iter().position(|x| x == *self).expect("Invalid model")
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

pub enum WhisperUpdate {
    Recording(bool),
    Transcribing(bool),
    Transcript(String),
    Level(f32),
}

pub struct StreamState {
    // the app holds this handle to keep the stream open (and the whisper context / thread alive)
    #[allow(dead_code)]
    stream: cpal::Stream,
}

pub struct AppDevice {
    pub name: String,
    device: cpal::Device,
    config: cpal::SupportedStreamConfig,
}

pub struct WhisperParams {
    pub accuracy: usize, // 1 for greedy, more for beam search
    pub model: WhichModel,
}

pub fn get_devices() -> Vec<AppDevice> {
    let host = cpal::default_host();
    let mut all = Vec::new();
    let input_devices = match host.input_devices() {
        Ok(devices) => devices.collect(),
        Err(whatever) => {
            tracing::warn!("Failed to get input devices: {}", whatever);
            Vec::new()
        }
    };
    for device in input_devices {
        let config = match device.default_input_config() {
            Ok(config) => config,
            Err(whatever) => {
                tracing::info!("Failed to get config for {:?}: {}", device.name(), whatever);
                continue;
            }
        };
        let name = device.name().unwrap_or("Unknown".to_string());
        all.push(AppDevice {
            name,
            device,
            config,
        });
    }
    let output_devices = match host.output_devices() {
        Ok(devices) => devices.collect(),
        Err(whatever) => {
            tracing::warn!("Failed to get output devices: {}", whatever);
            Vec::new()
        }
    };
    for device in output_devices {
        let config = match device.default_output_config() {
            Ok(config) => config,
            Err(whatever) => {
                tracing::info!("Failed to get config for {:?}: {}", device.name(), whatever);
                continue;
            }
        };
        let name = device.name().unwrap_or("Unknown".to_string());
        all.push(AppDevice {
            name,
            device,
            config,
        });
    }

    all
}

pub struct WhisperContext {
    decoder: Decoder,
    device: candle_core::Device,
    config: Config,
    mel_filters: Vec<f32>,
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
    })
}

// once the return value is dropped, listening stops
// and the sender is closed
pub fn start_listening(
    app: &Sender<WhisperUpdate>,
    app_device: &AppDevice,
    params: WhisperParams,
) -> Option<StreamState> {
    tracing::info!(
        "Listening on device: {}",
        app_device.device.name().expect("device name")
    );

    let (audio_tx, audio_rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = mpsc::channel();

    let err_fn = move |err| tracing::error!("an error occurred on stream: {}", err);

    let audio_config = &app_device.config;
    let channel_count = audio_config.channels() as usize;
    let data_callback = move |data: &[f32], _: &_| {
        let singlechannel = data
            .iter()
            .step_by(channel_count)
            .copied()
            .collect::<Vec<f32>>();
        audio_tx.send(singlechannel).expect("Failed to send data");
    };
    let stream = app_device
        .device
        .build_input_stream(
            &app_device.config.clone().into(),
            data_callback,
            err_fn,
            None,
        )
        .expect("Failed to build input stream");

    stream.play().expect("Failed to play stream");

    let app2 = app.clone();
    let (filtered_tx, filtered_rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = mpsc::channel();
    let config2 = app_device.config.clone();
    thread::spawn(move || {
        filter_audio_loop(app2, audio_rx, filtered_tx, config2);
    });

    let app2 = app.clone();
    thread::spawn(move || match whisper_loop(app2, filtered_rx, params) {
        Ok(_) => {
            tracing::info!("Whisper loop finished");
        }
        Err(err) => {
            tracing::error!("Whisper loop error: {}", err);
        }
    });

    Some(StreamState { stream })
}

fn whisper_loop(
    app: Sender<WhisperUpdate>,
    filtered_rx: Receiver<Vec<f32>>,
    params: WhisperParams,
) -> Result<(), anyhow::Error> {
    let mut ctx: WhisperContext = load_whisper_model(params.model)?;
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
        whisperize(&mut ctx, &aggregated_data, &app, &params)?;
    }
}

// a thread that collects non-silent audio samples and sends them on
fn filter_audio_loop(
    app: Sender<WhisperUpdate>,
    audio_rx: Receiver<Vec<f32>>,
    filtered_tx: Sender<Vec<f32>>,
    device_config: SupportedStreamConfig,
) {
    // here's the basic idea: receive 480 samples at a time (48000 / 100 = 480). If the max value
    // of the samples is above a threshold, then we know that there is a sound. If there is a sound,
    // then we can start recording the audio. Once we stop recording, we can send the recorded audio to Whisper.
    let mut under_threshold_count = 101;
    let mut recording_buffer: Vec<f32> = Vec::new();

    // a dynamic threshold (or something like silero-vad) would be better
    let threshold = 0.02f32;

    // accumulate data until we've been under the threshold for 100 samples
    loop {
        let data = match audio_rx.recv() {
            Ok(data) => data,
            Err(_) => {
                tracing::info!("Audio stream closed");
                return;
            }
        };

        let mut max = 0.0;
        for sample in data.iter() {
            if *sample > max {
                max = *sample;
            }
        }
        app.send(WhisperUpdate::Level(max))
            .expect("Failed to send level update");

        let sample_rate = device_config.sample_rate().0 as usize;
        let full_whisper_buffer = 30/*seconds*/ * sample_rate /*samples per second*/;

        if max > threshold {
            if under_threshold_count > 100 {
                // we've been listening to silence for a while, so we stopped recording. Indicate that we're listening again.
                app.send(WhisperUpdate::Recording(true))
                    .expect("Failed to send recording update");
            }
            recording_buffer.extend_from_slice(&data);
            under_threshold_count = 0;
        } else {
            // the incoming audio is below the threshold. Check how long it's been silent for.
            under_threshold_count += 1;
            if under_threshold_count < 100 {
                // not long enough, keep listening
                recording_buffer.extend_from_slice(&data);
            } else {
                if recording_buffer.len() > 0 {
                    app.send(WhisperUpdate::Recording(false))
                        .expect("Failed to send recording update");
                    let resampled = convert_sample_rate(&recording_buffer, sample_rate).unwrap();
                    filtered_tx.send(resampled).expect("Send filtered");
                    recording_buffer.clear();
                }
            }
        }

        // Whisper is trained to process 30 seconds at a time. So if we've got that much, send it now.
        if recording_buffer.len() >= full_whisper_buffer {
            let resampled = convert_sample_rate(&recording_buffer, sample_rate).unwrap();
            filtered_tx.send(resampled).expect("Send filtered");
            recording_buffer.clear();
        }
    }
}

fn whisperize(
    state: &mut WhisperContext,
    resampled: &[f32],
    app: &Sender<WhisperUpdate>,
    _params: &WhisperParams,
) -> Result<(), anyhow::Error> {
    let start = std::time::Instant::now();

    app.send(WhisperUpdate::Transcribing(true))
        .expect("Failed to send transcribing update");

    // boost the levels to at least 50% of the max
    let max = resampled.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    let boost = 0.5f32 / max;
    let resampled: Vec<f32> = resampled.iter().map(|x| x * boost).collect();

    let mel_start = std::time::Instant::now();
    let mel = log_mel_spectrogram(&resampled, state.config.num_mel_bins); // Use the new function
    let mel_duration = mel_start.elapsed().as_secs_f32();
    tracing::info!("Mel spectrogram generation took {:.2} seconds", mel_duration);

    let mel_len = mel.len();
    let num_bins = state.config.num_mel_bins;
    let mel = Tensor::from_vec(mel, (1, num_bins, mel_len / num_bins), &state.device)?;

    let segments = state.decoder.run(&mel, None)?;

    for segment in segments.iter() {
        let text = segment.dr.text.clone();
        tracing::info!("Transcribed segment: {:?}", segment);
        const NO_SPEECH_THRESHOLD: f64 = 0.05;
        const LOGPROB_THRESHOLD: f64 = -0.1;
        if segment.dr.no_speech_prob > NO_SPEECH_THRESHOLD
            && segment.dr.avg_logprob < LOGPROB_THRESHOLD
        {
            tracing::info!("No speech detected, skipping");
            continue;
        }
        app.send(WhisperUpdate::Transcript(text))?;
    }

    app.send(WhisperUpdate::Transcribing(false))?;

    // trace how long it took and how long the input was
    let duration = start.elapsed().as_secs_f32() as f64;
    let input_duration = resampled.len() as f64 / 16000.;
    tracing::info!(
        "Transcribed {} seconds of audio in {} seconds",
        input_duration,
        duration
    );
    Ok(())
}

fn convert_sample_rate(
    audio: &[f32],
    original_sample_rate: usize,
) -> Result<Vec<f32>, &'static str> {
    let params = rubato::SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: rubato::SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: rubato::WindowFunction::BlackmanHarris2,
    };
    const TARGET_SAMPLE_RATE: f64 = 16000.;
    let ratio = TARGET_SAMPLE_RATE / original_sample_rate as f64;
    let mut resampler =
        rubato::SincFixedIn::<f32>::new(ratio, 2.0, params, audio.len(), 1).unwrap();

    let waves_in = vec![audio; 1];
    let waves_out = resampler.process(&waves_in, None).unwrap();
    Ok(waves_out[0].to_vec())
}
