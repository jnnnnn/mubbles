//! Earshot Voice Activity Detection.
//!
//! Uses the `earshot` crate for blazingly-fast neural VAD (40x faster than Silero).
//! Weights are embedded in the binary — no model download needed, no ONNX runtime.
//! Includes a hysteresis state machine for robust speech segment detection
//! that rejects short transients (keystrokes, chair squeaks, etc.).

use earshot::Detector;

/// Number of 16kHz samples per VAD frame (16ms)
pub const FRAME_SIZE: usize = 256;
/// Minimum speech duration in frames (192ms = 12 frames) — rejects keystroke transients
/// but accepts single words like "yes" or "no".
const MIN_SPEECH_FRAMES: usize = 12;
/// Frames of silence before ending a speech segment (256ms = 16 frames)
const MIN_SILENCE_FRAMES: usize = 16;
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

/// Streaming Earshot VAD session with hysteresis state machine.
///
/// Feed 256-sample f32 frames at 16kHz via [`process_frame`](VadSession::process_frame).
/// Speech must persist for [`MIN_SPEECH_FRAMES`] (192ms) to be considered valid.
/// Silence for [`MIN_SILENCE_FRAMES`] (256ms) ends a speech segment.
pub struct VadSession {
    detector: Detector,

    // State machine
    current_sample: usize,
    triggered: bool,
    /// Sample index where the current silence period began
    temp_end: usize,
    /// Sample index where speech started
    current_speech_start: i64,
}

impl VadSession {
    /// Create a new VAD session. No model download — weights are embedded.
    pub fn new() -> Self {
        Self {
            detector: Detector::default(),
            current_sample: 0,
            triggered: false,
            temp_end: 0,
            current_speech_start: 0,
        }
    }

    /// Process one frame of 16kHz audio (must be exactly FRAME_SIZE samples).
    /// `threshold` is the speech probability threshold (0.1–0.9, default 0.5).
    /// Returns the raw speech probability [0.0, 1.0] and the VAD decision.
    pub fn process_frame(&mut self, frame: &[f32], threshold: f32) -> (f32, VadDecision) {
        assert_eq!(
            frame.len(),
            FRAME_SIZE,
            "VAD frame must be exactly {} samples",
            FRAME_SIZE
        );

        let speech_prob = self.detector.predict_f32(frame);
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
                    // Valid utterance (≥ 192ms speech)
                    self.reset_state();
                    VadDecision::SpeechEnd
                } else {
                    // False trigger (< 192ms speech) — discard
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

        (speech_prob, decision)
    }

    /// Reset the VAD state for a new audio stream.
    pub fn reset(&mut self) {
        self.detector.reset();
        self.current_sample = 0;
        self.triggered = false;
        self.temp_end = 0;
        self.current_speech_start = 0;
    }

    /// Returns whether the VAD is currently in a triggered (speaking) state.
    pub fn is_speaking(&self) -> bool {
        self.triggered
    }
}

impl VadSession {
    fn reset_state(&mut self) {
        self.temp_end = 0;
        self.triggered = false;
    }
}
