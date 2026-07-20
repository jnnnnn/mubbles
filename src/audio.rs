//! Audio capture module with platform-specific backends.
//!
//! On Linux, this uses PipeWire for better support of monitor sources (desktop audio capture).
//! On other platforms, it uses cpal.

use rubato::Resampler;
use std::{
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc,
    },
    thread,
};

use crate::app::WhisperUpdate;

// whisper is trained on 16kHz audio
pub(crate) const TARGET_SAMPLE_RATE: usize = crate::mel::SAMPLE_RATE;
const MINIMUM_AUDIO_LENGTH: f32 = 0.3; // 300ms — enough to reject keystroke false triggers (< 96ms speech)

/// Audio data with sample rate information
pub struct PcmAudio {
    pub data: Vec<f32>,
    pub sample_rate: usize,
}

/// Convert audio sample rate to the target rate for Whisper
pub(crate) fn convert_sample_rate(
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
    let ratio = TARGET_SAMPLE_RATE as f64 / original_sample_rate as f64;
    let mut resampler =
        rubato::SincFixedIn::<f32>::new(ratio, 2.0, params, audio.len(), 1).unwrap();

    let waves_in = vec![audio; 1];
    let waves_out = resampler.process(&waves_in, None).unwrap();
    Ok(waves_out[0].to_vec())
}

/// Filter audio to detect speech and accumulate non-silent segments.
/// This is shared across all platforms.
///
/// Uses Earshot neural VAD for speech detection, which reliably
/// distinguishes speech from keystrokes, chair noises, and other
/// transient sounds that fool energy-based detectors.
/// Output devices fall back to a simple energy gate.
pub fn filter_audio_loop(
    app: Sender<WhisperUpdate>,
    audio_rx: Receiver<PcmAudio>,
    filtered_tx: Sender<PcmAudio>,
    device_name: String,
    is_output: bool,
    output_active: Arc<AtomicBool>,
    echo_cancel: Arc<AtomicBool>,
    threshold_bits: Arc<AtomicU32>,
) -> Result<(), anyhow::Error> {
    use crate::vad::{self, VadDecision, VadSession};

    // Output devices use a simple energy gate (VAD is overkill for speakers)
    if is_output {
        return filter_audio_loop_output(
            app,
            audio_rx,
            filtered_tx,
            device_name,
            output_active,
            echo_cancel,
            threshold_bits,
        );
    }

    // ── Input devices: Earshot VAD gated buffer ────────────

    let mut vad = VadSession::new();
    let mut recording_buffer: Vec<f32> = Vec::new();
    let mut vad_buffer: Vec<f32> = Vec::new(); // 16kHz samples waiting to be framed
    let mut was_speaking = false;
    let mut silence_since_speech: usize = 0; // native-rate samples of silence after last speech
    const SILENCE_FLUSH_SAMPLES: usize = 10;

    loop {
        let PcmAudio { data, sample_rate } = match audio_rx.recv() {
            Ok(pcmaudio) => pcmaudio,
            Err(_) => {
                tracing::info!("Audio stream closed");
                app.send(WhisperUpdate::Recording(false))?;
                return Ok(());
            }
        };

        // Compute RMS for the level display
        let rms = if data.is_empty() {
            0.0
        } else {
            let sum_sq: f32 = data.iter().map(|s| s * s).sum();
            (sum_sq / data.len() as f32).sqrt()
        };

        // Echo cancellation: skip when output is active
        let muted = echo_cancel.load(Ordering::Relaxed) && output_active.load(Ordering::Relaxed);
        app.send(WhisperUpdate::Level {
            device: device_name.clone(),
            level: rms,
            muted,
        })?;
        if muted {
            continue;
        }

        // Downsample chunk to 16kHz and feed into VAD frame buffer
        let downsampled = vad::downsample_to_16k(&data, sample_rate);
        vad_buffer.extend_from_slice(&downsampled);

        let threshold = f32::from_bits(threshold_bits.load(Ordering::Relaxed));
        let mut speech_end = false;
        let mut speech_rejected = false;
        while vad_buffer.len() >= vad::FRAME_SIZE {
            let frame: Vec<f32> = vad_buffer.drain(..vad::FRAME_SIZE).collect();
            let (_prob, decision) = vad.process_frame(&frame, threshold);
            match decision {
                VadDecision::SpeechEnd => speech_end = true,
                VadDecision::SpeechRejected => speech_rejected = true,
                VadDecision::SpeechStart => {}
                _ => {}
            }
        }
        let is_speaking = vad.is_speaking();

        // Gate: buffer audio when speaking, flush on silence after speech
        if is_speaking {
            if !was_speaking {
                // Just started speaking
                app.send(WhisperUpdate::Recording(true))?;
            }
            recording_buffer.extend_from_slice(&data);
            silence_since_speech = 0;
        } else if was_speaking && speech_end {
            // Valid utterance ended (≥ 2s speech) — always transcribe
            app.send(WhisperUpdate::Recording(false))?;
            let resampled = convert_sample_rate(&recording_buffer, sample_rate).unwrap();
            filtered_tx.send(PcmAudio {
                data: resampled,
                sample_rate: TARGET_SAMPLE_RATE,
            })?;
            recording_buffer.clear();
        } else if was_speaking && speech_rejected {
            // False trigger (< 2s speech) — discard, don't transcribe
            app.send(WhisperUpdate::Recording(false))?;
            recording_buffer.clear();
        } else if !recording_buffer.is_empty() {
            // Safety net: buffer has data but VAD hasn't signaled end
            silence_since_speech += 1;
            recording_buffer.extend_from_slice(&data);
            if silence_since_speech >= SILENCE_FLUSH_SAMPLES {
                app.send(WhisperUpdate::Recording(false))?;
                if recording_buffer.len() >= (MINIMUM_AUDIO_LENGTH * sample_rate as f32) as usize {
                    let resampled = convert_sample_rate(&recording_buffer, sample_rate).unwrap();
                    filtered_tx.send(PcmAudio {
                        data: resampled,
                        sample_rate: TARGET_SAMPLE_RATE,
                    })?;
                }
                recording_buffer.clear();
            }
        }
        was_speaking = is_speaking;

        // if we've got more than 15 seconds of audio, find the lowest-energy point and send everything up to that point
        let full_whisper_buffer = 15/*seconds*/ * sample_rate /*samples per second*/;
        if recording_buffer.len() > full_whisper_buffer {
            let chunk_size = crate::mel::FFT_STEP * sample_rate / TARGET_SAMPLE_RATE; // 160 resamples per chunk (1 mel frame)

            let energies = recording_buffer
                .chunks(chunk_size)
                .map(|chunk| chunk.iter().fold(0.0f32, |acc, &x| acc.max(x.abs())))
                .collect::<Vec<f32>>();
            // Skip the first 300ms to avoid cutting mid-transient
            let skip_chunks = (0.3 * sample_rate as f32 / chunk_size as f32) as usize;
            let low_energy_index = energies
                .iter()
                .enumerate()
                .skip(skip_chunks)
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            let start_index = low_energy_index * chunk_size;
            let resampled =
                convert_sample_rate(&recording_buffer[..start_index], sample_rate).unwrap();
            filtered_tx.send(PcmAudio {
                data: resampled,
                sample_rate: TARGET_SAMPLE_RATE,
            })?;
            recording_buffer = recording_buffer[start_index..].to_vec();
        }
    }
}

/// Simple energy-gate filter for output devices (speakers).
/// VAD is unnecessary for speaker output — a basic amplitude threshold suffices
/// to detect when audio is playing through the device.
fn filter_audio_loop_output(
    app: Sender<WhisperUpdate>,
    audio_rx: Receiver<PcmAudio>,
    filtered_tx: Sender<PcmAudio>,
    device_name: String,
    output_active: Arc<AtomicBool>,
    echo_cancel: Arc<AtomicBool>,
    threshold_bits: Arc<AtomicU32>,
) -> Result<(), anyhow::Error> {
    let mut recording_buffer: Vec<f32> = Vec::new();

    loop {
        let PcmAudio { data, sample_rate } = match audio_rx.recv() {
            Ok(pcmaudio) => pcmaudio,
            Err(_) => {
                tracing::info!("Audio stream closed");
                app.send(WhisperUpdate::Recording(false))?;
                return Ok(());
            }
        };

        let threshold = f32::from_bits(threshold_bits.load(Ordering::Relaxed));

        let rms = if data.is_empty() {
            0.0
        } else {
            let sum_sq: f32 = data.iter().map(|s| s * s).sum();
            (sum_sq / data.len() as f32).sqrt()
        };

        let active = rms > threshold;
        output_active.store(active, Ordering::Relaxed);

        let muted = echo_cancel.load(Ordering::Relaxed) && active;
        app.send(WhisperUpdate::Level {
            device: device_name.clone(),
            level: rms,
            muted,
        })?;

        if active {
            app.send(WhisperUpdate::Recording(true))?;
            recording_buffer.extend_from_slice(&data);
        } else if !recording_buffer.is_empty() {
            app.send(WhisperUpdate::Recording(false))?;
            if recording_buffer.len() >= (MINIMUM_AUDIO_LENGTH * sample_rate as f32) as usize {
                let resampled = convert_sample_rate(&recording_buffer, sample_rate).unwrap();
                filtered_tx.send(PcmAudio {
                    data: resampled,
                    sample_rate: TARGET_SAMPLE_RATE,
                })?;
            }
            recording_buffer.clear();
        }

        // 15-second max buffer split
        let full_whisper_buffer = 15 * sample_rate;
        if recording_buffer.len() > full_whisper_buffer {
            let chunk_size = crate::mel::FFT_STEP * sample_rate / TARGET_SAMPLE_RATE;
            let energies = recording_buffer
                .chunks(chunk_size)
                .map(|chunk| chunk.iter().fold(0.0f32, |acc, &x| acc.max(x.abs())))
                .collect::<Vec<f32>>();
            let low_energy_index = energies
                .iter()
                .enumerate()
                .skip((MINIMUM_AUDIO_LENGTH * sample_rate as f32 / chunk_size as f32) as usize)
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            let start_index = low_energy_index * chunk_size;
            let resampled =
                convert_sample_rate(&recording_buffer[..start_index], sample_rate).unwrap();
            filtered_tx.send(PcmAudio {
                data: resampled,
                sample_rate: TARGET_SAMPLE_RATE,
            })?;
            recording_buffer = recording_buffer[start_index..].to_vec();
        }
    }
}

// ============================================================================
// Platform-specific implementations
// ============================================================================

#[cfg(target_os = "linux")]
mod platform {
    use super::*;
    use crate::audio_pipewire::{
        get_default_devices, start_device_enumeration, start_pipewire_stream, DeviceReceiver,
        PwDevice, PwStreamState,
    };

    /// Unified audio device representation
    pub struct AppDevice {
        pub name: String,
        pub is_output: bool,
        pub(crate) inner: PwDevice,
    }

    impl AppDevice {
        pub fn sample_rate(&self) -> usize {
            self.inner.sample_rate as usize
        }
    }

    /// Stream handle that keeps audio capture alive
    pub struct StreamState {
        #[allow(dead_code)]
        pub(crate) inner: PwStreamState,
    }

    /// Receiver for async device enumeration results
    pub type DeviceEnumerationReceiver = Option<DeviceReceiver>;

    /// Start async device enumeration and return initial placeholder devices.
    /// Call `check_device_enumeration` periodically to get the full list when ready.
    pub fn get_devices() -> (Vec<AppDevice>, DeviceEnumerationReceiver) {
        // Start background enumeration
        let rx = start_device_enumeration();

        // Return placeholder devices immediately
        let devices = get_default_devices()
            .into_iter()
            .map(|pw| AppDevice {
                name: pw.name.clone(),
                is_output: pw.is_monitor,
                inner: pw,
            })
            .collect();

        (devices, Some(rx))
    }

    /// Check if device enumeration has completed. Returns Some(devices) if ready.
    pub fn check_device_enumeration(rx: &mut DeviceEnumerationReceiver) -> Option<Vec<AppDevice>> {
        if let Some(receiver) = rx {
            match receiver.try_recv() {
                Ok(pw_devices) => {
                    *rx = None; // Enumeration complete, clear receiver
                    Some(
                        pw_devices
                            .into_iter()
                            .map(|pw| AppDevice {
                                name: pw.name.clone(),
                                is_output: pw.is_monitor,
                                inner: pw,
                            })
                            .collect(),
                    )
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => None, // Still enumerating
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    *rx = None; // Thread finished unexpectedly
                    None
                }
            }
        } else {
            None
        }
    }

    /// Start capturing audio from a device
    pub fn start_audio_thread(
        app: Sender<WhisperUpdate>,
        app_device: &AppDevice,
        filtered_tx: Sender<PcmAudio>,
        partial_tx: Option<Sender<PcmAudio>>,
        output_active: Arc<AtomicBool>,
        echo_cancel: Arc<AtomicBool>,
        threshold_bits: Arc<AtomicU32>,
    ) -> anyhow::Result<StreamState> {
        tracing::info!("Starting PipeWire audio capture on: {}", app_device.name);

        let (audio_tx, audio_rx) = mpsc::channel::<PcmAudio>();

        // Start the PipeWire stream
        let pw_state =
            start_pipewire_stream(app.clone(), &app_device.inner, audio_tx, partial_tx.clone())?;

        // Start the filter thread
        let app2 = app.clone();
        let device_name = app_device.name.clone();
        let is_output = app_device.is_output;
        thread::spawn(move || {
            match filter_audio_loop(
                app2,
                audio_rx,
                filtered_tx,
                device_name,
                is_output,
                output_active,
                echo_cancel,
                threshold_bits,
            ) {
                Ok(_) => tracing::info!("Audio filter thread finished successfully"),
                Err(e) => tracing::error!("Audio filter thread failed: {:?}", e),
            }
        });

        Ok(StreamState { inner: pw_state })
    }
}

#[cfg(not(target_os = "linux"))]
mod platform {
    use super::*;
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    /// Unified audio device representation
    pub struct AppDevice {
        pub name: String,
        pub is_output: bool,
        pub(crate) device: cpal::Device,
        pub(crate) config: cpal::SupportedStreamConfig,
    }

    impl AppDevice {
        pub fn sample_rate(&self) -> usize {
            self.config.sample_rate().0 as usize
        }
    }

    /// Stream handle that keeps audio capture alive
    pub struct StreamState {
        #[allow(dead_code)]
        pub(crate) stream: cpal::Stream,
    }

    /// Receiver for async device enumeration results (not used on non-Linux)
    pub type DeviceEnumerationReceiver = Option<()>;

    /// Get available audio devices (input and output)
    pub fn get_devices() -> (Vec<AppDevice>, DeviceEnumerationReceiver) {
        let host = cpal::default_host();
        let mut all = Vec::new();

        // Get input devices (microphones)
        let input_devices: Vec<_> = match host.input_devices() {
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
            let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
            all.push(AppDevice {
                name,
                is_output: false,
                device,
                config,
            });
        }

        // Get output devices (for loopback/monitor on platforms that support it)
        let output_devices: Vec<_> = match host.output_devices() {
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
            let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
            all.push(AppDevice {
                name,
                is_output: true,
                device,
                config,
            });
        }

        // On non-Linux, device enumeration is synchronous, so no receiver needed
        (all, None)
    }

    /// Check if device enumeration has completed (always returns None on non-Linux)
    pub fn check_device_enumeration(_rx: &mut DeviceEnumerationReceiver) -> Option<Vec<AppDevice>> {
        None
    }

    /// Start capturing audio from a device
    pub fn start_audio_thread(
        app: Sender<WhisperUpdate>,
        app_device: &AppDevice,
        filtered_tx: Sender<PcmAudio>,
        partial_tx: Option<Sender<PcmAudio>>,
        output_active: Arc<AtomicBool>,
        echo_cancel: Arc<AtomicBool>,
        threshold_bits: Arc<AtomicU32>,
    ) -> anyhow::Result<StreamState> {
        tracing::info!(
            "Listening on device: {}",
            app_device
                .device
                .name()
                .unwrap_or_else(|_| "Unknown".to_string())
        );

        let (audio_tx, audio_rx) = mpsc::channel::<PcmAudio>();

        let err_fn = move |err| tracing::error!("an error occurred on stream: {}", err);

        let audio_config = &app_device.config;
        let channel_count = audio_config.channels() as usize;
        let sample_rate = audio_config.sample_rate().0 as usize;
        let data_callback = move |raw: &[f32], _: &_| {
            let data = raw
                .iter()
                .step_by(channel_count)
                .copied()
                .collect::<Vec<f32>>();
            audio_tx
                .send(PcmAudio {
                    data: data.clone(),
                    sample_rate,
                })
                .unwrap_or_else(|_| {
                    // this is too noisy.
                    // tracing::debug!("Audio channel closed, can't send audio data");
                });
            if let Some(partial_tx) = &partial_tx {
                partial_tx
                    .send(PcmAudio { data, sample_rate })
                    .unwrap_or_else(|_| {
                        // this is too noisy.
                        // tracing::debug!("Partial channel closed, can't send partial audio data");
                    });
            }
        };
        let config2 = app_device.config.clone();
        let stream =
            app_device
                .device
                .build_input_stream(&config2.into(), data_callback, err_fn, None)?;

        stream.play()?;

        let app2 = app.clone();
        let device_name = app_device.name.clone();
        let is_output = app_device.is_output;
        thread::spawn(move || {
            match filter_audio_loop(
                app2,
                audio_rx,
                filtered_tx,
                device_name,
                is_output,
                output_active,
                echo_cancel,
                threshold_bits,
            ) {
                Ok(_) => tracing::info!("Audio filter thread finished successfully"),
                Err(e) => tracing::error!("Audio filter thread failed: {:?}", e),
            }
        });

        Ok(StreamState { stream })
    }
}

// Re-export platform-specific types
pub use platform::{
    check_device_enumeration, get_devices, start_audio_thread, AppDevice,
    DeviceEnumerationReceiver, StreamState,
};
