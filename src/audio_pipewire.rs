//! PipeWire audio backend for Linux.
//!
//! This module provides audio capture using PipeWire, supporting both input devices
//! (microphones) and monitor sources (desktop/application audio capture).
//!
//! Device enumeration runs on a background thread to avoid blocking the UI.

use std::{
    convert::TryInto,
    mem,
    sync::mpsc::{self, Receiver, Sender},
    thread::{self, JoinHandle},
};

use pipewire as pw;
use pw::{properties::properties, spa};
use spa::param::format::{MediaSubtype, MediaType};
use spa::param::format_utils;
use spa::pod::Pod;

use crate::app::WhisperUpdate;
use crate::audio::PcmAudio;

const DEFAULT_SAMPLE_RATE: u32 = 48000;

/// A PipeWire audio device (node) that can be used for capture
#[derive(Clone)]
pub struct PwDevice {
    pub name: String,
    pub node_id: Option<u32>,
    pub is_monitor: bool,
    pub sample_rate: u32,
}

impl std::fmt::Debug for PwDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PwDevice {{ name: {}, node_id: {:?}, is_monitor: {} }}",
            self.name, self.node_id, self.is_monitor
        )
    }
}

/// Holds the state for an active PipeWire audio stream
pub struct PwStreamState {
    /// Handle to the thread running the main loop - dropping this stops capture
    pub thread: JoinHandle<()>,
    /// Sender to signal the thread to stop
    pub stop_tx: Sender<()>,
}

impl Drop for PwStreamState {
    fn drop(&mut self) {
        // Signal the thread to stop
        let _ = self.stop_tx.send(());
    }
}

/// Channel receiver for device enumeration results
pub type DeviceReceiver = Receiver<Vec<PwDevice>>;

/// Start device enumeration on a background thread.
/// Returns a receiver that will receive the device list when enumeration completes.
pub fn start_device_enumeration() -> DeviceReceiver {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let devices = enumerate_devices_blocking();
        let _ = tx.send(devices);
    });

    rx
}

/// Get default placeholder devices while real enumeration happens in background
pub fn get_default_devices() -> Vec<PwDevice> {
    vec![
        PwDevice {
            name: "Default (Auto-connect)".to_string(),
            node_id: None,
            is_monitor: false,
            sample_rate: DEFAULT_SAMPLE_RATE,
        },
        PwDevice {
            name: "Default Monitor (Desktop Audio)".to_string(),
            node_id: None,
            is_monitor: true,
            sample_rate: DEFAULT_SAMPLE_RATE,
        },
    ]
}

/// Internal: Enumerate devices synchronously (runs on background thread)
fn enumerate_devices_blocking() -> Vec<PwDevice> {
    let mut devices = Vec::new();

    // Initialize PipeWire
    pw::init();

    // Create a temporary main loop to enumerate devices
    let mainloop = match pw::main_loop::MainLoop::new(None) {
        Ok(ml) => ml,
        Err(e) => {
            tracing::error!("Failed to create PipeWire main loop: {:?}", e);
            return devices;
        }
    };

    let context = match pw::context::Context::new(&mainloop) {
        Ok(ctx) => ctx,
        Err(e) => {
            tracing::error!("Failed to create PipeWire context: {:?}", e);
            return devices;
        }
    };

    let core = match context.connect(None) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("Failed to connect to PipeWire: {:?}", e);
            return devices;
        }
    };

    let registry = match core.get_registry() {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("Failed to get PipeWire registry: {:?}", e);
            return devices;
        }
    };

    // Collect devices in a channel since the callback borrows
    let (device_tx, device_rx) = mpsc::channel();

    let _listener = registry
        .add_listener_local()
        .global(move |global| {
            // We're interested in nodes with media.class = Audio/Source or Audio/Sink
            if global.type_ != pw::types::ObjectType::Node {
                return;
            }

            let props = match global.props {
                Some(p) => p,
                None => return,
            };

            let media_class = props.get("media.class").unwrap_or("");
            let node_name = props.get("node.name").unwrap_or("Unknown");
            let node_description = props.get("node.description").unwrap_or(node_name);

            // Audio/Source = microphone/input device
            // Audio/Sink = output device (we want its monitor)
            let (is_valid, is_monitor) = match media_class {
                "Audio/Source" => (true, false),
                "Audio/Sink" => (true, true),
                _ => (false, false),
            };

            if is_valid {
                let display_name = if is_monitor {
                    format!("{} (Monitor)", node_description)
                } else {
                    node_description.to_string()
                };

                let device = PwDevice {
                    name: display_name,
                    node_id: Some(global.id),
                    is_monitor,
                    sample_rate: DEFAULT_SAMPLE_RATE,
                };

                tracing::info!("Found device: {:?}", device);
                let _ = device_tx.send(device);
            }
        })
        .register();

    // Run the main loop briefly to enumerate devices
    // We need to do a sync roundtrip to ensure we've received all globals
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let done_clone = done.clone();

    let _core_listener = core
        .add_listener_local()
        .done(move |_, _| {
            done_clone.store(true, std::sync::atomic::Ordering::SeqCst);
        })
        .register();

    // Trigger a sync - the done callback will fire when complete
    core.sync(0).expect("Failed to sync with PipeWire core");

    // Process events until sync is complete
    let loop_ = mainloop.loop_();
    let mut count = 0;
    while !done.load(std::sync::atomic::Ordering::SeqCst) {
        tracing::debug!("Iteration {:?}", count);
        count += 1;
        loop_.iterate(std::time::Duration::from_millis(100));
    }

    // Collect all devices from the channel
    while let Ok(device) = device_rx.try_recv() {
        devices.push(device);
    }

    // Sort: monitors first, then by name
    devices.sort_by(|a, b| {
        if a.is_monitor != b.is_monitor {
            b.is_monitor.cmp(&a.is_monitor) // monitors first
        } else {
            a.name.cmp(&b.name)
        }
    });

    // Prepend default devices
    let mut result = get_default_devices();
    result.extend(devices);

    tracing::info!("Found {} PipeWire audio devices", result.len());
    for device in &result {
        tracing::debug!("  {:?}", device);
    }

    result
}

/// User data passed to stream callbacks
struct StreamUserData {
    format: spa::param::audio::AudioInfoRaw,
    audio_tx: Sender<PcmAudio>,
    partial_tx: Option<Sender<PcmAudio>>,
}

/// Start capturing audio from a PipeWire device
pub fn start_pipewire_stream(
    app: Sender<WhisperUpdate>,
    device: &PwDevice,
    audio_tx: Sender<PcmAudio>,
    partial_tx: Option<Sender<PcmAudio>>,
) -> anyhow::Result<PwStreamState> {
    let device_name = device.name.clone();
    let node_id = device.node_id;
    let is_monitor = device.is_monitor;

    let (stop_tx, stop_rx) = mpsc::channel();

    let thread = thread::spawn(move || {
        if let Err(e) = run_pipewire_stream(
            app,
            &device_name,
            node_id,
            is_monitor,
            audio_tx,
            partial_tx,
            stop_rx,
        ) {
            tracing::error!("PipeWire stream error: {:?}", e);
        }
    });

    Ok(PwStreamState { thread, stop_tx })
}

/// Internal function that runs the PipeWire main loop
fn run_pipewire_stream(
    app: Sender<WhisperUpdate>,
    device_name: &str,
    node_id: Option<u32>,
    is_monitor: bool,
    audio_tx: Sender<PcmAudio>,
    partial_tx: Option<Sender<PcmAudio>>,
    stop_rx: Receiver<()>,
) -> anyhow::Result<()> {
    tracing::info!(
        "Starting PipeWire capture on {} (node_id={:?}, monitor={})",
        device_name,
        node_id,
        is_monitor
    );

    pw::init();

    let mainloop = pw::main_loop::MainLoop::new(None)?;
    let context = pw::context::Context::new(&mainloop)?;
    let core = context.connect(None)?;

    // Build stream properties
    let mut props = properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Communication",
    };

    // For monitor sources, we need to set stream.capture.sink to capture from a sink
    if is_monitor {
        props.insert(*pw::keys::STREAM_CAPTURE_SINK, "true");
    }

    // If we have a specific target node, set it
    if let Some(id) = node_id {
        props.insert(*pw::keys::TARGET_OBJECT, id.to_string());
    }

    // Create the stream with user data for callbacks
    let user_data = StreamUserData {
        format: Default::default(),
        audio_tx,
        partial_tx,
    };

    let stream = pw::stream::Stream::new(&core, "mubbles-audio-capture", props)?;

    // Set up the stream listener for callbacks
    let app_clone = app.clone();
    let _listener = stream
        .add_local_listener_with_user_data(user_data)
        .param_changed(|_stream, user_data, id, param| {
            // NULL means to clear the format
            let Some(param) = param else {
                return;
            };
            if id != pw::spa::param::ParamType::Format.as_raw() {
                return;
            }

            let (media_type, media_subtype) = match format_utils::parse_format(param) {
                Ok(v) => v,
                Err(_) => return,
            };

            // only accept raw audio
            if media_type != MediaType::Audio || media_subtype != MediaSubtype::Raw {
                return;
            }

            // call a helper function to parse the format for us.
            user_data
                .format
                .parse(param)
                .expect("Failed to parse param changed to AudioInfoRaw");

            tracing::info!(
                "PipeWire capturing rate:{} channels:{}",
                user_data.format.rate(),
                user_data.format.channels()
            );
        })
        .process(move |stream, user_data| {
            process_audio_buffer(stream, user_data, &app_clone);
        })
        .register()?;

    // Build audio format parameters - request F32LE mono or stereo at native rate
    let mut audio_info = spa::param::audio::AudioInfoRaw::new();
    audio_info.set_format(spa::param::audio::AudioFormat::F32LE);

    let obj = pw::spa::pod::Object {
        type_: pw::spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: pw::spa::param::ParamType::EnumFormat.as_raw(),
        properties: audio_info.into(),
    };
    let values: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(obj),
    )
    .unwrap()
    .0
    .into_inner();

    let mut params = [Pod::from_bytes(&values).unwrap()];

    // Connect the stream
    let flags = pw::stream::StreamFlags::AUTOCONNECT
        | pw::stream::StreamFlags::MAP_BUFFERS
        | pw::stream::StreamFlags::RT_PROCESS;

    stream.connect(spa::utils::Direction::Input, node_id, flags, &mut params)?;

    // Notify that we've started
    let _ = app.send(WhisperUpdate::Status(
        "PipeWire stream connected".to_string(),
    ));

    // Run the main loop, checking for stop signal
    let loop_ = mainloop.loop_();

    // Add a timer to periodically check for stop signal
    let stop_rx_arc = std::sync::Arc::new(std::sync::Mutex::new(stop_rx));
    let stop_rx_clone = stop_rx_arc.clone();
    let mainloop_weak = mainloop.downgrade();

    let _timer = loop_.add_timer(move |_| {
        if let Ok(rx) = stop_rx_clone.lock() {
            if rx.try_recv().is_ok() {
                tracing::info!("PipeWire stream received stop signal");
                if let Some(ml) = mainloop_weak.upgrade() {
                    ml.quit();
                }
            }
        }
    });
    _timer
        .update_timer(
            Some(std::time::Duration::from_millis(100)),
            Some(std::time::Duration::from_millis(100)),
        )
        .into_result()?;

    // Run the main loop
    mainloop.run();

    // Notify that we've stopped recording
    let _ = app.send(WhisperUpdate::Recording(false));

    tracing::info!("PipeWire stream ended");
    Ok(())
}

/// Process an incoming audio buffer from PipeWire
fn process_audio_buffer(
    stream: &pw::stream::StreamRef,
    user_data: &mut StreamUserData,
    _app: &Sender<WhisperUpdate>,
) {
    let mut buffer = match stream.dequeue_buffer() {
        Some(b) => b,
        None => return,
    };

    let datas = buffer.datas_mut();
    if datas.is_empty() {
        return;
    }

    let data = &mut datas[0];
    let n_channels = user_data.format.channels().max(1) as usize;
    let chunk_size = data.chunk().size() as usize;
    let n_samples = chunk_size / mem::size_of::<f32>();

    if n_samples == 0 {
        return;
    }

    let sample_rate = user_data.format.rate() as usize;
    if sample_rate == 0 {
        return;
    }

    // Get the raw audio data
    let samples = match data.data() {
        Some(s) => s,
        None => return,
    };

    // Convert bytes to f32 samples, taking first channel only (mono)
    let mut audio_data = Vec::with_capacity(n_samples / n_channels);
    for n in (0..n_samples).step_by(n_channels) {
        let start = n * mem::size_of::<f32>();
        let end = start + mem::size_of::<f32>();
        if end <= samples.len() {
            let sample_bytes: [u8; 4] = samples[start..end].try_into().unwrap_or([0; 4]);
            let sample = f32::from_le_bytes(sample_bytes);
            audio_data.push(sample);
        }
    }

    if audio_data.is_empty() {
        return;
    }

    // Send to main audio processing pipeline
    let _ = user_data.audio_tx.send(PcmAudio {
        data: audio_data.clone(),
        sample_rate,
    });

    // Send to partial processing if enabled
    if let Some(ref partial_tx) = user_data.partial_tx {
        let _ = partial_tx.send(PcmAudio {
            data: audio_data,
            sample_rate,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_enumeration() {
        // This test requires PipeWire to be running
        let devices = enumerate_devices_blocking();
        println!("Found {} devices:", devices.len());
        for d in &devices {
            println!("  {:?}", d);
        }
    }

    #[test]
    fn test_async_device_enumeration() {
        let rx = start_device_enumeration();
        let devices = rx.recv().expect("Should receive devices");
        println!("Found {} devices:", devices.len());
        for d in &devices {
            println!("  {:?}", d);
        }
    }
}
