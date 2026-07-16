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
const MINIMUM_AUDIO_LENGTH: usize = 3; // 3 seconds

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
/// Uses adaptive noise floor tracking: computes RMS energy per chunk,
/// tracks ambient noise level via exponential moving average during silence,
/// and detects speech when energy exceeds the noise floor by the configured
/// SNR ratio. This handles low-volume mics far better than a fixed amplitude threshold.
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
    // Noise floor estimation constants
    const NOISE_FLOOR_EMA_ALPHA: f32 = 0.02; // smoothing factor for noise floor tracking
    const MIN_ABSOLUTE_FLOOR: f32 = 0.0001; // absolute minimum to avoid detecting circuit noise
    const SILENCE_HOLD_COUNT: usize = 50; // chunks of silence before ending a speech segment
    const SILENCE_START_COUNT: usize = 100; // chunks of silence before considering "not speaking"

    let mut under_threshold_count = SILENCE_START_COUNT + 1;
    let mut recording_buffer: Vec<f32> = Vec::new();
    let mut noise_floor: f32 = 0.001; // initial noise floor estimate

    // accumulate data until we've been under the threshold for 100 samples
    loop {
        let PcmAudio { data, sample_rate } = match audio_rx.recv() {
            Ok(pcmaudio) => pcmaudio,
            Err(_) => {
                tracing::info!("Audio stream closed");
                // end thread because there's no more work to do
                app.send(WhisperUpdate::Recording(false))?;
                return Ok(());
            }
        };

        // The threshold control from the UI represents the SNR ratio:
        // a value of 1.0 means speech must be 2x noise floor (3 dB above),
        // a value of 3.0 means speech must be 8x noise floor (9 dB above).
        // Scale: 0.001..=0.5 in old UI maps to ~0.1..=50.0 SNR ratio.
        let snr_ratio = f32::from_bits(threshold_bits.load(Ordering::Relaxed)) * 100.0;

        // Compute RMS energy (more robust than peak amplitude for speech detection)
        let rms = if data.is_empty() {
            0.0
        } else {
            let sum_sq: f32 = data.iter().map(|s| s * s).sum();
            (sum_sq / data.len() as f32).sqrt()
        };

        // Update noise floor estimate during silence (when below the adaptive threshold)
        let adaptive_threshold = (noise_floor * snr_ratio).max(MIN_ABSOLUTE_FLOOR);
        if rms < adaptive_threshold {
            // Slowly adapt noise floor toward current ambient level
            noise_floor = (1.0 - NOISE_FLOOR_EMA_ALPHA) * noise_floor + NOISE_FLOOR_EMA_ALPHA * rms;
        }

        // Output devices: signal when active so input devices can mute
        if is_output {
            output_active.store(rms > adaptive_threshold, Ordering::Relaxed);
        }

        // Input devices: skip audio when output is active (echo cancellation)
        let muted = !is_output
            && echo_cancel.load(Ordering::Relaxed)
            && output_active.load(Ordering::Relaxed);
        app.send(WhisperUpdate::Level {
            device: device_name.clone(),
            level: rms,
            muted,
        })?;
        if muted {
            continue;
        }

        if rms > adaptive_threshold {
            if under_threshold_count > SILENCE_START_COUNT {
                // we've been listening to silence for a while, so we stopped recording. Indicate that we're listening again.
                app.send(WhisperUpdate::Recording(true))?;
            }
            recording_buffer.extend_from_slice(&data);
            under_threshold_count = 0;
        } else if !recording_buffer.is_empty() {
            // the incoming audio is back under the threshold. Check how long it's been silent for.
            under_threshold_count += 1;
            if under_threshold_count < SILENCE_HOLD_COUNT
                || recording_buffer.len() < MINIMUM_AUDIO_LENGTH * sample_rate
            {
                // not long enough, keep listening
                recording_buffer.extend_from_slice(&data);
            } else {
                app.send(WhisperUpdate::Recording(false))?;
                let resampled = convert_sample_rate(&recording_buffer, sample_rate).unwrap();
                filtered_tx.send(PcmAudio {
                    data: resampled,
                    sample_rate: TARGET_SAMPLE_RATE,
                })?;
                recording_buffer.clear();
            }
        }

        // if we've got more than 15 seconds of audio, find the lowest-energy point and send everything up to that point
        let full_whisper_buffer = 15/*seconds*/ * sample_rate /*samples per second*/;
        if recording_buffer.len() > full_whisper_buffer {
            let chunk_size = crate::mel::FFT_STEP * sample_rate / TARGET_SAMPLE_RATE; // 160 resamples per chunk (1 mel frame)

            let energies = recording_buffer
                .chunks(chunk_size)
                .map(|chunk| chunk.iter().fold(0.0f32, |acc, &x| acc.max(x.abs())))
                .collect::<Vec<f32>>();
            let low_energy_index = energies
                .iter()
                .enumerate()
                .skip(MINIMUM_AUDIO_LENGTH * sample_rate / chunk_size)
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
