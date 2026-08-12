use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc,
    },
    thread::JoinHandle,
};

use crate::{
    app::WhisperUpdate,
    audio::{self, AppDevice, PcmAudio, StreamState},
    partial,
    whisper::{self, WhisperParams},
};

/// Default Earshot VAD speech probability threshold
pub(crate) const DEFAULT_THRESHOLD: f32 = 0.75;

// =============================================================================
// AudioWorker
// =============================================================================

/// Holds handles to keep audio capture streams and filter threads alive.
/// Drop = stop all audio capture. The filtered_tx is dropped, which closes the
/// channel and causes the whisper thread to exit naturally.
pub(crate) struct AudioWorker {
    pub(crate) audio_streams: HashMap<String, StreamState>,
    pub(crate) app_tx: Sender<WhisperUpdate>,
    pub(crate) filtered_tx: Sender<PcmAudio>,
    pub(crate) output_active: Arc<AtomicBool>,
    pub(crate) echo_cancel: Arc<AtomicBool>,
    pub(crate) device_threshold_atoms: HashMap<String, Arc<AtomicU32>>,
    pub(crate) silence_timeout_atom: Arc<AtomicU32>,
}

impl AudioWorker {
    /// Add an audio stream for a device
    pub(crate) fn add_device(&mut self, device: &AppDevice, threshold: f32) {
        if self.audio_streams.contains_key(&device.name) {
            return;
        }
        let threshold_atom = Arc::new(AtomicU32::new(threshold.to_bits()));
        match audio::start_audio_thread(
            self.app_tx.clone(),
            device,
            self.filtered_tx.clone(),
            None, // partial_tx — only first device gets it, handled at setup
            self.output_active.clone(),
            self.echo_cancel.clone(),
            threshold_atom.clone(),
            self.silence_timeout_atom.clone(),
        ) {
            Ok(stream) => {
                tracing::info!("Started audio stream for: {}", device.name);
                self.audio_streams.insert(device.name.clone(), stream);
                self.device_threshold_atoms
                    .insert(device.name.clone(), threshold_atom);
            }
            Err(e) => {
                tracing::error!("Failed to start audio for {}: {}", device.name, e);
            }
        }
    }

    /// Remove an audio stream for a device (dropping it stops capture)
    pub(crate) fn remove_device(&mut self, name: &str) {
        if self.audio_streams.remove(name).is_some() {
            tracing::info!("Stopped audio stream for: {}", name);
        }
    }

    /// Check all filter threads for panics. Removes dead devices from the stream map.
    pub(crate) fn check_panic(&mut self, app_tx: &Sender<WhisperUpdate>) {
        let dead: Vec<String> = self
            .audio_streams
            .iter()
            .filter(|(_, s)| s.filter_thread.is_finished())
            .map(|(name, _)| name.clone())
            .collect();
        for name in dead {
            if let Some(stream) = self.audio_streams.remove(&name) {
                match stream.filter_thread.join() {
                    Ok(()) => {
                        tracing::warn!("Filter thread for '{}' exited", name);
                    }
                    Err(e) => {
                        let msg = if let Some(s) = e.downcast_ref::<&'static str>() {
                            s.to_string()
                        } else if let Some(s) = e.downcast_ref::<String>() {
                            s.clone()
                        } else {
                            format!("{:?}", e)
                        };
                        tracing::error!("Filter thread for '{}' panicked: {}", name, msg);
                    }
                }
                let _ = app_tx.send(WhisperUpdate::Recording(false));
            }
        }
    }
}

// =============================================================================
// WhisperWorker
// =============================================================================

/// Holds handles to keep whisper and partial transcription threads alive.
/// When stopped via [`WhisperWorker::stop`], returns the utterance receiver
/// so it can be reused by a new whisper instance without restarting audio.
pub(crate) struct WhisperWorker {
    pub(crate) whisper_thread: Option<JoinHandle<Receiver<PcmAudio>>>,
    pub(crate) whisper_stop: Arc<AtomicBool>,
    pub(crate) partial_thread: Option<JoinHandle<()>>,
    pub(crate) partial_tx: Option<Sender<PcmAudio>>,
}

impl WhisperWorker {
    /// Check if the whisper thread has panicked and log the error.
    pub(crate) fn check_panic(&mut self) {
        if let Some(thread) = self.whisper_thread.as_mut() {
            if thread.is_finished() {
                let handle = self.whisper_thread.take().unwrap();
                if let Err(e) = handle.join() {
                    let msg = if let Some(s) = e.downcast_ref::<&'static str>() {
                        s.to_string()
                    } else if let Some(s) = e.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        format!("{:?}", e)
                    };
                    tracing::error!("Whisper thread panicked: {}", msg);
                }
            }
        }
    }

    /// Stop whisper transcription, wait for threads to exit, and recover the
    /// utterance receiver so it can be reused by a new whisper instance.
    pub(crate) fn stop(mut self) -> Receiver<PcmAudio> {
        self.whisper_stop.store(true, Ordering::Relaxed);
        // Drop partial_tx to signal the partial thread
        drop(self.partial_tx.take());
        // Wait for partial thread
        if let Some(pt) = self.partial_thread.take() {
            let _ = pt.join();
        }
        // Wait for whisper thread and recover the rx
        if let Some(thread) = self.whisper_thread.take() {
            match thread.join() {
                Ok(rx) => rx,
                Err(_) => {
                    let (_, rx) = mpsc::channel();
                    rx
                }
            }
        } else {
            let (_, rx) = mpsc::channel();
            rx
        }
    }
}

impl Drop for WhisperWorker {
    fn drop(&mut self) {
        self.whisper_stop.store(true, Ordering::Relaxed);
        drop(self.partial_tx.take());
        if let Some(pt) = self.partial_thread.take() {
            let _ = pt.join();
        }
        if let Some(thread) = self.whisper_thread.take() {
            let _ = thread.join();
        }
    }
}

// =============================================================================
// Free functions
// =============================================================================

/// Start audio capture for the selected devices.
/// Returns the audio worker and the utterance receiver for whisper to consume.
pub(crate) fn start_audio(
    app: &Sender<WhisperUpdate>,
    devices: &[&AppDevice],
    echo_cancel_enabled: bool,
    device_thresholds: &HashMap<String, f32>,
    silence_timeout_ms: f32,
    partials_enabled: bool,
) -> Result<(AudioWorker, Receiver<PcmAudio>), anyhow::Error> {
    // Shared filtered audio channel: filter threads send complete utterances here
    let (filtered_tx, filtered_rx) = mpsc::channel();

    // Partial channel: first device sends raw audio here for real-time preview
    let (partial_tx, _partial_rx) = mpsc::channel();

    // Shared flags
    let output_active = Arc::new(AtomicBool::new(false));
    let echo_cancel = Arc::new(AtomicBool::new(echo_cancel_enabled));
    let silence_timeout_atom = Arc::new(AtomicU32::new(silence_timeout_ms.to_bits()));

    let mut audio_streams = HashMap::new();
    let mut device_threshold_atoms = HashMap::new();

    for (i, device) in devices.iter().enumerate() {
        let ptx = if partials_enabled && i == 0 {
            Some(partial_tx.clone())
        } else {
            None
        };
        let threshold = device_thresholds
            .get(&device.name)
            .copied()
            .unwrap_or(DEFAULT_THRESHOLD);
        let threshold_atom = Arc::new(AtomicU32::new(threshold.to_bits()));
        let stream = audio::start_audio_thread(
            app.clone(),
            device,
            filtered_tx.clone(),
            ptx,
            output_active.clone(),
            echo_cancel.clone(),
            threshold_atom.clone(),
            silence_timeout_atom.clone(),
        )?;
        audio_streams.insert(device.name.clone(), stream);
        device_threshold_atoms.insert(device.name.clone(), threshold_atom);
    }

    Ok((
        AudioWorker {
            audio_streams,
            app_tx: app.clone(),
            filtered_tx,
            output_active,
            echo_cancel,
            device_threshold_atoms,
            silence_timeout_atom,
        },
        filtered_rx,
    ))
}

/// Start whisper transcription, consuming the utterance receiver.
pub(crate) fn start_whisper(
    app: &Sender<WhisperUpdate>,
    filtered_rx: Receiver<PcmAudio>,
    params: WhisperParams,
    paused: Arc<AtomicBool>,
) -> Result<WhisperWorker, anyhow::Error> {
    let (partial_thread, partial_tx) = if params.partials {
        let (tx, rx) = mpsc::channel();
        let thread = partial::start_partial_thread(app.clone(), rx)?;
        (Some(thread), Some(tx))
    } else {
        (None, None)
    };

    let whisper_stop = Arc::new(AtomicBool::new(false));
    let whisper_thread = Some(whisper::start_whisper_thread(
        app.clone(),
        filtered_rx,
        params,
        paused,
        whisper_stop.clone(),
    )?);

    Ok(WhisperWorker {
        whisper_thread,
        whisper_stop,
        partial_thread,
        partial_tx,
    })
}
