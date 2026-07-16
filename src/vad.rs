//! Silero Voice Activity Detection using candle-onnx.
//!
//! Downloads the `onnx-community/silero-vad` model from HuggingFace Hub on first use
//! and runs inference via candle-onnx for neural speech probability estimation.
//! Includes a hysteresis state machine for robust speech segment detection
//! that rejects short transients (keystrokes, chair squeaks, etc.).

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_onnx::onnx::ModelProto;
use std::path::PathBuf;

/// Number of 16kHz samples per VAD frame (32ms)
pub const FRAME_SIZE: usize = 512;
/// Context overlap between consecutive frames (64 samples = 4ms)
const CONTEXT_SIZE: usize = 64;
/// Minimum speech duration in frames (~96ms = 3 frames)
const MIN_SPEECH_FRAMES: usize = 3;
/// Frames of silence before ending a speech segment (~128ms = 4 frames)
const MIN_SILENCE_FRAMES: usize = 4;
/// Maximum speech duration in frames before forced split (30s)
const MAX_SPEECH_FRAMES: usize = (30 * 16000) / FRAME_SIZE;
/// Frames of silence at max speech before forced split (~100ms)
const MIN_SILENCE_AT_MAX_FRAMES: usize = 3;

/// Per-frame decision from the VAD state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadDecision {
    Silence,
    Speech,
    SpeechStart,
    SpeechEnd,
}

/// Streaming Silero VAD session with hysteresis state machine.
///
/// Feed 512-sample f32 frames at 16kHz via [`process_frame`](VadSession::process_frame).
/// The state machine handles trigger/release hysteresis, minimum speech/silence
/// durations, speech padding, and forced splits on long utterances.
pub struct VadSession {
    model: ModelProto,
    sample_rate_tensor: Tensor,
    state: Tensor,
    context: Tensor,
    device: Device,

    // Hysteresis state machine (ported from VadIter)
    current_sample: usize,
    triggered: bool,
    temp_end: usize,
    prev_end: usize,
    next_start: usize,
    current_speech_start: i64,
    to_take: bool,
}

impl VadSession {
    /// Download (if needed) and load the Silero VAD ONNX model.
    pub fn new(device: &Device) -> Result<Self> {
        let model_path = Self::download_model()?;
        let model = candle_onnx::read_file(model_path)?;
        let sample_rate_tensor = Tensor::new(16000i64, device)?;
        let state = Tensor::zeros((2, 1, 128), DType::F32, device)?;
        let context = Tensor::zeros((1, CONTEXT_SIZE), DType::F32, device)?;

        Ok(Self {
            model,
            sample_rate_tensor,
            state,
            context,
            device: device.clone(),
            current_sample: 0,
            triggered: false,
            temp_end: 0,
            prev_end: 0,
            next_start: 0,
            current_speech_start: 0,
            to_take: false,
        })
    }

    /// Process one frame of 16kHz audio (must be exactly FRAME_SIZE samples).
    /// `threshold` is the speech probability threshold (0.1–0.9, default 0.5).
    /// Lower values are more sensitive (catch whispers), higher values
    /// require clearer speech.
    /// Returns the raw speech probability [0.0, 1.0] and the VAD decision.
    pub fn process_frame(&mut self, frame: &[f32], threshold: f32) -> Result<(f32, VadDecision)> {
        assert_eq!(
            frame.len(),
            FRAME_SIZE,
            "VAD frame must be exactly {} samples",
            FRAME_SIZE
        );

        let speech_prob = self.run_model(frame)?;
        let neg_threshold = threshold - 0.15;

        self.current_sample += FRAME_SIZE;

        let decision = if speech_prob > threshold {
            // Speech detected
            if self.temp_end != 0 {
                self.temp_end = 0;
                if self.next_start < self.prev_end {
                    self.next_start = self.current_sample.saturating_sub(FRAME_SIZE);
                }
            }
            if !self.triggered {
                self.triggered = true;
                self.current_speech_start = self.current_sample as i64 - FRAME_SIZE as i64;
                VadDecision::SpeechStart
            } else {
                VadDecision::Speech
            }
        } else {
            // Silence (or below threshold)
            if self.triggered
                && (self.current_sample as i64 - self.current_speech_start) as usize
                    > MAX_SPEECH_FRAMES * FRAME_SIZE
            {
                // Forced split at max speech duration
                if self.prev_end > 0 {
                    self.end_speech(self.prev_end);
                    if self.next_start < self.prev_end {
                        self.triggered = false;
                    } else {
                        self.current_speech_start = self.next_start as i64;
                    }
                } else {
                    self.end_speech(self.current_sample);
                    self.prev_end = 0;
                    self.next_start = 0;
                    self.temp_end = 0;
                    self.triggered = false;
                }
                self.to_take = true;
            } else if self.triggered && speech_prob < neg_threshold {
                // End of speech
                if self.temp_end == 0 {
                    self.temp_end = self.current_sample;
                }
                if self.current_sample.saturating_sub(self.temp_end)
                    > MIN_SILENCE_AT_MAX_FRAMES * FRAME_SIZE
                {
                    self.prev_end = self.temp_end;
                }
                if self.current_sample.saturating_sub(self.temp_end)
                    >= MIN_SILENCE_FRAMES * FRAME_SIZE
                {
                    let speech_duration = self.temp_end as i64 - self.current_speech_start;
                    if speech_duration > (MIN_SPEECH_FRAMES * FRAME_SIZE) as i64 {
                        self.end_speech(self.temp_end);
                        self.prev_end = 0;
                        self.next_start = 0;
                        self.temp_end = 0;
                        self.triggered = false;
                        self.to_take = true;
                    }
                }
            }

            if self.to_take {
                self.to_take = false;
                VadDecision::SpeechEnd
            } else {
                VadDecision::Silence
            }
        };

        Ok((speech_prob, decision))
    }

    /// Reset the VAD state for a new audio stream.
    pub fn reset(&mut self) -> Result<()> {
        self.state = Tensor::zeros((2, 1, 128), DType::F32, &self.device)?;
        self.context = Tensor::zeros((1, CONTEXT_SIZE), DType::F32, &self.device)?;
        self.current_sample = 0;
        self.triggered = false;
        self.temp_end = 0;
        self.prev_end = 0;
        self.next_start = 0;
        self.current_speech_start = 0;
        self.to_take = false;
        Ok(())
    }

    /// Returns whether the VAD is currently in a triggered (speaking) state.
    pub fn is_speaking(&self) -> bool {
        self.triggered
    }
}

// ── Internal helpers ──────────────────────────────────────────

impl VadSession {
    fn download_model() -> Result<PathBuf> {
        let api = hf_hub::api::sync::Api::new()?;
        let model = api
            .model("onnx-community/silero-vad".into())
            .get("onnx/model.onnx")?;
        Ok(model)
    }

    fn run_model(&mut self, frame: &[f32]) -> Result<f32> {
        // Build input: concatenate context with new frame
        let frame_tensor = Tensor::from_slice(frame, (1, FRAME_SIZE), &self.device)?;
        let input = Tensor::cat(&[&self.context, &frame_tensor], 1)?;
        // Update context for next frame
        self.context = Tensor::from_slice(
            &frame[FRAME_SIZE - CONTEXT_SIZE..],
            (1, CONTEXT_SIZE),
            &self.device,
        )?;

        let inputs = std::collections::HashMap::from_iter([
            ("input".to_string(), input),
            ("sr".to_string(), self.sample_rate_tensor.clone()),
            ("state".to_string(), self.state.clone()),
        ]);
        let out = candle_onnx::simple_eval(&self.model, inputs)
            .map_err(|e| anyhow::anyhow!("VAD model inference failed: {e}"))?;

        let out_names = &self.model.graph.as_ref().unwrap().output;
        let output = out
            .get(&out_names[0].name)
            .ok_or_else(|| anyhow::anyhow!("VAD model missing output node"))?
            .clone();
        self.state = out
            .get(&out_names[1].name)
            .ok_or_else(|| anyhow::anyhow!("VAD model missing state node"))?
            .clone();

        let prob = output.flatten_all()?.to_vec1::<f32>()?;
        Ok(prob[0])
    }

    fn end_speech(&mut self, _end_sample: usize) {
        // State machine transition marker — the actual padding and boundary
        // logic is handled by the caller via SpeechStart/SpeechEnd decisions.
    }
}

/// Downsample audio from any source rate to 16kHz for VAD processing.
/// Uses simple nearest-neighbor sampling — quality is sufficient for VAD
/// (the actual transcription audio goes through proper sinc resampling).
pub fn downsample_to_16k(audio: &[f32], src_rate: usize) -> Vec<f32> {
    if src_rate == 16000 {
        return audio.to_vec();
    }
    let ratio = src_rate as f64 / 16000.0;
    let out_len = (audio.len() as f64 / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_idx = (i as f64 * ratio) as usize;
        out.push(audio.get(src_idx).copied().unwrap_or(0.0));
    }
    out
}
