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
/// Minimum speech duration in frames (2s = 63 frames at 32ms/frame)
const MIN_SPEECH_FRAMES: usize = 63;
/// Frames of silence before ending a speech segment (256ms = 8 frames)
const MIN_SILENCE_FRAMES: usize = 8;
/// Maximum speech duration in frames before forced split (30s)
const MAX_SPEECH_FRAMES: usize = (30 * 16000) / FRAME_SIZE;

/// Per-frame decision from the VAD state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadDecision {
    Silence,
    Speech,
    SpeechStart,
    /// Valid utterance ended — flush to transcription.
    SpeechEnd,
    /// False trigger (keystroke, too-short utterance) — discard buffer, don't transcribe.
    SpeechRejected,
}

/// Streaming Silero VAD session with hysteresis state machine.
///
/// Feed 512-sample f32 frames at 16kHz via [`process_frame`](VadSession::process_frame).
/// Speech must persist for [`MIN_SPEECH_FRAMES`] (2s) to be considered valid.
/// Silence for [`MIN_SILENCE_FRAMES`] (256ms) ends a speech segment.
pub struct VadSession {
    model: ModelProto,
    sample_rate_tensor: Tensor,
    state: Tensor,
    context: Tensor,
    device: Device,

    // State machine
    current_sample: usize,
    triggered: bool,
    /// Sample index where the current silence period began
    temp_end: usize,
    /// Sample index where speech started
    current_speech_start: i64,
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
            current_speech_start: 0,
        })
    }

    /// Process one frame of 16kHz audio (must be exactly FRAME_SIZE samples).
    /// `threshold` is the speech probability threshold (0.1–0.9, default 0.5).
    /// Returns the raw speech probability [0.0, 1.0] and the VAD decision.
    pub fn process_frame(&mut self, frame: &[f32], threshold: f32) -> Result<(f32, VadDecision)> {
        assert_eq!(
            frame.len(),
            FRAME_SIZE,
            "VAD frame must be exactly {} samples",
            FRAME_SIZE
        );

        let speech_prob = self.run_model(frame)?;
        self.current_sample += FRAME_SIZE;

        let decision = if speech_prob > threshold {
            // ── Speech frame ──────────────────────────────
            if self.temp_end != 0 {
                // Speech resumed after silence — reset the silence counter
                self.temp_end = 0;
            }
            if !self.triggered {
                self.triggered = true;
                self.current_speech_start = self.current_sample as i64 - FRAME_SIZE as i64;
                VadDecision::SpeechStart
            } else {
                VadDecision::Speech
            }
        } else if self.triggered {
            // ── Silence while triggered ───────────────────
            if self.temp_end == 0 {
                // First silence frame after speech — mark the boundary
                self.temp_end = self.current_sample;
            }

            let total_duration = (self.current_sample as i64 - self.current_speech_start) as usize;

            if total_duration > MAX_SPEECH_FRAMES * FRAME_SIZE {
                // Forced split at 30s max speech
                self.reset_state();
                VadDecision::SpeechEnd
            } else if self.current_sample.saturating_sub(self.temp_end)
                >= MIN_SILENCE_FRAMES * FRAME_SIZE
            {
                // Enough consecutive silence — evaluate the utterance
                let speech_duration = self.temp_end as i64 - self.current_speech_start;
                if speech_duration > (MIN_SPEECH_FRAMES * FRAME_SIZE) as i64 {
                    // Valid utterance (≥ 2s speech)
                    self.reset_state();
                    VadDecision::SpeechEnd
                } else {
                    // False trigger (< 2s speech) — discard
                    self.reset_state();
                    VadDecision::SpeechRejected
                }
            } else {
                // Still waiting for enough silence
                VadDecision::Silence
            }
        } else {
            // ── Silence, not triggered ────────────────────
            VadDecision::Silence
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
        self.current_speech_start = 0;
        Ok(())
    }

    /// Returns whether the VAD is currently in a triggered (speaking) state.
    pub fn is_speaking(&self) -> bool {
        self.triggered
    }
}

// ── Internal helpers ──────────────────────────────────────────

impl VadSession {
    fn reset_state(&mut self) {
        self.temp_end = 0;
        self.triggered = false;
    }

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
