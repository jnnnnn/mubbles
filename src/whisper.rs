use std::sync::{
    mpsc::{Receiver, Sender},
    Arc,
};

use anyhow::{Error as E, Result};
use candle_core::{IndexOp, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use hf_hub::{api::sync::Api, Repo, RepoType};
use rand::{distr::Distribution, SeedableRng};
use tokenizers::Tokenizer;

use candle_transformers::models::whisper::{self as m, Config};

use crate::{app::WhisperUpdate, whisper_word_align::{clear_attention_hooks, set_attention_hooks}};

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
    startofprev_token: u32,
    startoflm_token: u32,
    timestamp_begin_token: u32,
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
        let startofprev_token = token_id(&tokenizer, "<|startofprev|>").unwrap_or(0);
        let startoflm_token = token_id(&tokenizer, "<|startoflm|>").unwrap_or(0);
        let timestamp_begin_token = token_id(&tokenizer, "<|0.00|>").unwrap_or(0);
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
            startofprev_token,
            startoflm_token,
            timestamp_begin_token,
        })
    }

    fn decode(
        &mut self,
        mel: &Tensor,
        temperature: f64,
        prompt_content_tokens: Option<&[u32]>,
    ) -> Result<DecodingResult> {
        let model = &mut self.model;
        // todo: running the encoder is expensive! Do it once (separately to decode) and reuse the result for each temperature.
        let audio_features = model.encoder_forward(mel, true)?;
        if self.verbose {
            println!("audio features: {:?}", audio_features.dims());
        }
        let sample_len = model.config().max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;

        let mut tokens = vec![];

        if let Some(prompt_content_tokens) = prompt_content_tokens {
            if !prompt_content_tokens.is_empty() {
                tokens.push(self.startofprev_token);
                tokens.extend_from_slice(prompt_content_tokens);
            }
        }

        let prefix_len = tokens.len();

        tokens.push(self.sot_token);

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
            let decoder = match model {
                Model::Normal(m) => &mut m.decoder,
                Model::Quantized(m) => anyhow::bail!("Quantized timestamping not implemented"),
            };
            let qk_receiver = set_attention_hooks(decoder)?;
            let ys = model.decoder_forward(&tokens_t, &audio_features, i == 0)?;
            // clear hooks to close all senders, so the receiver can be dropped
            let decoder = match model {
                Model::Normal(m) => &mut m.decoder,
                Model::Quantized(m) => anyhow::bail!("Quantized timestamping not implemented"),
            };
            clear_attention_hooks(decoder);
            let query_key_tensors: Vec<(usize, Tensor)> = qk_receiver
                .into_iter()
                .collect();
            // log the indices to check they're in the right order
            tracing::info!(
                "QK indices: {:?}",
                query_key_tensors.iter().map(|(i, _)| i).collect::<Vec<_>>()
            );
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
            let next_token = if temperature > 0f64 {
                let prs = softmax(&(&logits / temperature)?, 0)?;
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

        // drop prefix tokens from start of vec
        let tokens = tokens
            .iter()
            .skip(prefix_len)
            .copied()
            .collect::<Vec<u32>>();

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature,
            compression_ratio: f64::NAN,
        })
    }

    // fn find_alignment() {
    //     let mut all_word_timings: Vec<WordTiming> = Vec::new();

    //     // Word alignment parameters
    //     let word_timestamps = true; // Assuming we want word timestamps
    //     let qk_scale = 1.0f32;
    //     let medfilt_width = 7;
    //     let prepend_punctuations = "\"'“¿([{-";
    //     let append_punctuations = "\"'.。,，!！?？:：”)]}、";

    //     // Collect all text tokens from all segments for alignment
    //     let mut text_tokens_for_alignment = Vec::new();
    //     for seg_res in &segments_results {
    //         // Assuming DecodingResult has a `tokens` field that are *text* tokens (post-SOT, etc.)
    //         // and *before* EOT. The `find_alignment` Python code filters for token < tokenizer.eot.
    //         // This needs to match how `seg_res.tokens` are structured.
    //         let eot_id = state
    //             .decoder
    //             .tokenizer
    //             .token_to_id(m::EOT_TOKEN)
    //             .unwrap_or(u32::MAX);
    //         text_tokens_for_alignment.extend(
    //             seg_res
    //                 .dr
    //                 .tokens
    //                 .iter()
    //                 .filter(|&&tok| tok < eot_id)
    //                 .cloned(),
    //         );
    //     }

    //     if !text_tokens_for_alignment.is_empty() {
    //         match &mut state.decoder.model {
    //             Model::Normal(normal_model) => {
    //                 match find_alignment(
    //                     &mut normal_model.decoder,
    //                     &state.config,
    //                     &state.decoder.tokenizer,
    //                     &text_tokens_for_alignment,
    //                     &audio_features,
    //                     num_mel_frames,
    //                     qk_scale,
    //                     medfilt_width,
    //                     &state.device,
    //                 ) {
    //                     Ok(mut timings) => {
    //                         merge_punctuations(
    //                             &mut timings,
    //                             prepend_punctuations,
    //                             append_punctuations,
    //                         );
    //                         all_word_timings.extend(timings);
    //                     }
    //                     Err(e) => {
    //                         tracing::error!("Word alignment failed: {:?}", e);
    //                         app.send(WhisperUpdate::Status(format!("Alignment error: {}", e)))?;
    //                     }
    //                 }
    //             }
    //             Model::Quantized(_) => {
    //                 // find_alignment might need to be generic or have a quantized path
    //                 tracing::warn!("Word alignment not implemented for quantized models yet.");
    //                 app.send(WhisperUpdate::Status(
    //                     "Alignment not for quantized".to_string(),
    //                 ))?;
    //             }
    //         }
    //     }
    // }
    fn decode_with_fallback(
        &mut self,
        segment: &Tensor,
        prompt_content_tokens: Option<&[u32]>,
    ) -> Result<DecodingResult> {
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult> = self.decode(segment, t, prompt_content_tokens);
            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            // todo: unimplemented. currently the only way for an error to happen is if an allocation fails. 
            // compression_ratio is not calculated, so very repetitive outputs are not recalculated with a higher temperature.
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

    fn run(
        &mut self,
        mel: &Tensor,
        _times: Option<(f64, f64)>,
        prompt_content_tokens: Option<&[u32]>,
    ) -> Result<(Vec<Segment>, Vec<u32>)> {
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];
        let mut last_segment_content_tokens = vec![];
        while seek < content_frames {
            let time_offset = (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration = (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;

            let dr = self.decode_with_fallback(&mel_segment, prompt_content_tokens)?;

            seek += segment_size;
            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                println!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            last_segment_content_tokens = dr.tokens.clone();
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };
            segments.push(segment)
        }
        Ok((segments, last_segment_content_tokens))
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
    decoder: Decoder,
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
