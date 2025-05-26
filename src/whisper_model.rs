use anyhow::{Error as E, Result};
use candle_core::{IndexOp, Tensor};
use candle_nn::ops::softmax;
use rand::{distr::Distribution, SeedableRng};
use tokenizers::Tokenizer;

use candle_transformers::models::whisper::{self as m, Config};

use crate::{
    whisper::WhichModel,
    whisper_word_align::{clear_attention_hooks, dtw_path_from_matrix, median_filter, set_attention_hooks},
};

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
            // todo: candle_nn::ops::sdpa would probably speed this up a lot
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
pub(crate) struct DecodingResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub avg_logprob: f64,
    pub no_speech_prob: f64,
    pub temperature: f64,
    pub compression_ratio: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct Segment {
    pub start: f64,
    pub duration: f64,
    pub dr: DecodingResult,
}

pub struct AlignmentHead {
    pub layer: usize,
    pub head: usize,
}

// indexes for (n_layers, n_heads) the cross-attention heads that are highly
// correlated to the word-level timing, i.e. the alignment between audio and
// text tokens.
#[rustfmt::skip]
pub(crate) static ALIGNMENT_HEADS: &[(WhichModel, &[usize])] = &[
    (WhichModel::TinyEn, &[6, 12, 17, 18, 19, 20, 21, 22]),
    (WhichModel::Tiny, &[14, 18, 20, 21, 22, 23]),
    (WhichModel::BaseEn, &[27, 39, 41, 45, 47]), // [[3, 3], [4, 7], [5, 1], [5, 5], [5, 7]], layers 6, attention_heads 8
    (WhichModel::Base, &[25, 34, 35, 39, 41, 42, 44, 46]),
    (WhichModel::SmallEn, &[78, 84, 87, 92, 98, 101, 103, 108, 112, 116, 118, 120, 121, 122, 123, 126, 131, 134, 136]),
    (WhichModel::Small, &[63, 69, 96, 100, 103, 104, 108, 115, 117, 125]),
    (WhichModel::MediumEn, &[180, 225, 236, 238, 244, 256, 260, 265, 284, 286, 295, 298, 303, 320, 323, 329, 334, 348]),
    (WhichModel::Medium, &[223, 244, 255, 257, 320, 372]),
    (WhichModel::Large, &[199, 222, 224, 237, 447, 451, 457, 462, 475]),
    (WhichModel::LargeV2, &[212, 277, 331, 332, 333, 355, 356, 364, 371, 379, 391, 422, 423, 443, 449, 452, 465, 467, 473, 505, 521, 532, 555]),
    (WhichModel::LargeV3, &[140, 217, 258, 272, 321, 354, 391, 424, 481, 506]),
    (WhichModel::Large, &[140, 217, 258, 272, 321, 354, 391, 424, 481, 506]),
    (WhichModel::LargeV3Turbo, &[44, 51, 63, 66, 71, 74]),
    //(WhichModel::Turbo, &[44, 51, 63, 66, 71, 74]),
];

pub fn get_alignment_heads(model: WhichModel, config: &Config) -> Vec<AlignmentHead> {
    let heads = ALIGNMENT_HEADS
        .iter()
        .find(|(m, _)| *m == model)
        .map(|(_, h)| *h)
        .unwrap_or(&[]);
    heads
        .iter()
        .map(|&index| AlignmentHead {
            layer: index / config.encoder_attention_heads,
            head: index % config.encoder_attention_heads,
        })
        .collect()
}

pub(crate) struct Decoder {
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
    alignment_heads: Vec<AlignmentHead>,
}

impl Decoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        device: &candle_core::Device,
        language_token: Option<u32>,
        task: Option<Task>,
        timestamps: bool,
        verbose: bool,
        alignment_heads: Vec<AlignmentHead>,
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
            alignment_heads,
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

        // Iteratively produce text tokens from encoded audio tokens. Stop when the model generates EOT.
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            // The model expects a batch dim but this inference loop does not handle
            // it so we add it at this point.
            let tokens_t = tokens_t.unsqueeze(0)?;
            let decoder = match model {
                Model::Normal(m) => &mut m.decoder,
                Model::Quantized(_) => anyhow::bail!("Quantized timestamping not implemented"),
            };
            let qk_receiver = set_attention_hooks(decoder)?;
            let ys = model.decoder_forward(&tokens_t, &audio_features, i == 0)?;
            // clear hooks to close all senders, so the receiver can be dropped
            let decoder = match model {
                Model::Normal(m) => &mut m.decoder,
                Model::Quantized(_) => anyhow::bail!("Quantized timestamping not implemented"),
            };
            clear_attention_hooks(decoder);
            let query_key_tensors: Vec<Tensor> = qk_receiver.into_iter().collect();

            // print shape of first query_key_tensor
            tracing::info!(
                "query_key_tensors shape: {:?} {:?}",
                query_key_tensors.len(),
                query_key_tensors[0].dims(),
            );
            // What is qk? Each element `qk[b, h, i, j]` represents the
            // attention score between the `i`-th query token (from the text
            // token sequence (so far)) and the `j`-th key token (from the key
            // (audio token) sequence) for the `h`-th attention head in the
            // `b`-th batch. qk_receiver gets a qk for each block (layer), so
            // query_key_tensors is [layer][batch, head, text_query_token,
            // audio_key_token].
            // For example, 6 [8, 14, 1500] for base.en with 6 layers, 8 heads per layer, 14 tokens generated so far, and 1500 audio tokens representing 30s of audio.

            // we now squash all this together into a single tensor of only the bits we care about: [usefulhead, i, j]
            // py: # heads * tokens * frames
            // py: weights = torch.stack([QKs[_l][_h] for _l, _h in model.alignment_heads.indices().T])
            let useful_slices: Vec<Tensor> = self
                .alignment_heads
                .iter()
                .map(|head| -> Result<Tensor> {
                    let layer = head.layer;
                    let head = head.head;
                    Ok(query_key_tensors[layer].i((0, head, prefix_len.., ..))?)
                })
                .collect::<Result<_>>()?;
            // the python shape is now [head, i, j]
            // but we stack it to [i, j, head] shape as candlenn wants to softmax the last dim
            let weights = Tensor::stack(&useful_slices, 2)?;
            // py: weights = weights[:, :, : num_frames // 2]
            // todo: cut down to actual length of audio, not the full padded 30s
            // py: weights = (weights * qk_scale).softmax(dim=-1)
            // qk_scale always 1
            let weights = candle_nn::ops::softmax_last_dim(&weights)?;
            // py: std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
            // this is a mean across dim `i`, the text query tokens
            let text_query_mean = weights.mean_keepdim(1)?;
            let text_query_std = weights.var_keepdim(1)?.sqrt()?;
            // py: weights = (weights - mean) / std
            let weights = (weights.sub(&text_query_mean))?.div(&text_query_std)?;
            // py: weights = median_filter(weights, medfilt_width)
            // pytorch median filters along the last dimension (j), and medfilt_width is 7
            // candle doesn't have a median filter, so we do it ourselves
            // this just smooths the attention weights a bit, removing outliers
            let weights = median_filter(&weights, 7, 1)?;
            // py: matrix = weights.mean(axis=0)
            // this is the mean across all the attention heads. 
            // we now have [i, j] shape, where i is the text query token and j is the audio key token.
            let matrix = weights.mean(2)?;
            // py: matrix = matrix[len(tokenizer.sot_sequence) : -1]
            // we did this earlier
            // py: text_indices, time_indices = dtw(-matrix)
            let text_token_timesteps = dtw_path_from_matrix(&matrix)?;

            tracing::info!(
                "text_token_timesteps: {:?} {:?}",
                text_token_timesteps.len(),
                text_token_timesteps,
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

    pub fn decode_with_fallback(
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

    pub fn run(
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

    pub fn set_language_token(&mut self, language_token: Option<u32>) {
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
pub enum Task {
    Transcribe,
    Translate,
}
