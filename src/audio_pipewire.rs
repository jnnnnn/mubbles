use libpulse_binding::{
    context::{Context, FlagSet as ContextFlagSet},
    mainloop::standard::{Mainloop, IterateResult},
    proplist::Proplist,
    sample::{Format, Spec},
    stream::{Direction, FlagSet as StreamFlagSet, State, Stream},
    def::BufferAttr,
};
use std::{
    cell::RefCell,
    ops::Deref,
    rc::Rc,
    sync::mpsc::{self, Receiver, Sender},
    thread,
    time::Duration,
};
use rubato::Resampler;

use crate::app::WhisperUpdate;

pub struct AudioChunk {
    data: Vec<f32>,
}

// whisper is trained on 16kHz audio
pub(crate) const TARGET_SAMPLE_RATE: usize = crate::mel::SAMPLE_RATE;
const MINIMUM_AUDIO_LENGTH: usize = 3; // 3 seconds

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

// a thread that collects non-silent audio samples and sends them on
fn filter_audio_loop(
    app: Sender<WhisperUpdate>,
    audio_rx: Receiver<PcmAudio>,
    filtered_tx: Sender<PcmAudio>,
) -> Result<(), anyhow::Error> {
    // here's the basic idea: receive 480 samples at a time (48000 / 100 = 480). If the max value
    // of the samples is above a threshold, then we know that there is a sound. If there is a sound,
    // then we can start recording the audio. Once we stop recording, we can send the recorded audio to Whisper.
    let mut under_threshold_count = 101;
    let mut recording_buffer: Vec<f32> = Vec::new();

    // a dynamic threshold (or something like silero-vad) would be better
    // something like, threshold = 2 * lowest-level-in-last-ten-seconds
    let threshold = 0.05f32;

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

        let mut max = 0.0;
        for sample in data.iter() {
            if *sample > max {
                max = *sample;
            }
        }
        app.send(WhisperUpdate::Level(max))?;

        if max > threshold {
            if under_threshold_count > 100 {
                // we've been listening to silence for a while, so we stopped recording. Indicate that we're listening again.
                app.send(WhisperUpdate::Recording(true))?;
            }
            recording_buffer.extend_from_slice(&data);
            under_threshold_count = 0;
        } else if recording_buffer.len() > 0 {
            // the incoming audio is back under the threshold. Check how long it's been silent for.
            under_threshold_count += 1;
            if under_threshold_count < 50
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

#[derive(Clone)]
pub struct PulseDevice {
    pub name: String,
    pub description: String,
    pub is_capture: bool,
}

impl std::fmt::Display for PulseDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.description, if self.is_capture { "Input" } else { "Output" })
    }
}

pub fn get_devices() -> Vec<PulseDevice> {
    // For simplicity, return some common device configurations
    // In a full implementation, you'd use introspection to get actual devices
    vec![
        PulseDevice {
            name: "@DEFAULT_SOURCE@".to_string(),
            description: "Default Input".to_string(),
            is_capture: true,
        },
        PulseDevice {
            name: "@DEFAULT_SINK@".to_string(),
            description: "Default Output".to_string(),
            is_capture: false,
        },
    ]
}

pub struct PcmAudio {
    pub data: Vec<f32>,
    pub sample_rate: usize,
}

pub struct StreamState {
    #[allow(dead_code)]
    pub join_handle: Option<thread::JoinHandle<()>>,
}

impl Drop for StreamState {
    fn drop(&mut self) {
        if let Some(handle) = self.join_handle.take() {
            // The thread should exit when the stream is dropped
            let _ = handle.join();
        }
    }
}

// PulseAudio stream management (works with PipeWire via compatibility layer)
pub fn start_audio_thread(
    app: Sender<WhisperUpdate>,
    device: &PulseDevice,
    filtered_tx: Sender<PcmAudio>,
    partial_tx: Option<Sender<PcmAudio>>,
) -> anyhow::Result<StreamState> {
    tracing::info!("Starting PulseAudio/PipeWire audio capture on device: {}", device.description);

    let (audio_tx, audio_rx) = mpsc::channel::<PcmAudio>();
    let device_clone = device.clone();
    let app_clone = app.clone();

    // Start the audio filtering thread
    let app2 = app.clone();
    thread::spawn(
        move || match filter_audio_loop(app2, audio_rx, filtered_tx) {
            Ok(_) => tracing::info!("Audio filter thread finished successfully"),
            Err(e) => tracing::error!("Audio filter thread failed: {:?}", e),
        },
    );

    // Start PulseAudio audio capture thread
    let join_handle = thread::spawn(move || {
        if let Err(e) = run_pulse_capture(device_clone, audio_tx, partial_tx) {
            tracing::error!("PulseAudio capture failed: {:?}", e);
            let _ = app_clone.send(WhisperUpdate::Recording(false));
        }
    });

    Ok(StreamState {
        join_handle: Some(join_handle),
    })
}

fn run_pulse_capture(
    device: PulseDevice,
    audio_tx: Sender<PcmAudio>,
    partial_tx: Option<Sender<PcmAudio>>,
) -> anyhow::Result<()> {
    let mut proplist = Proplist::new().unwrap();
    proplist
        .set_str(libpulse_binding::proplist::properties::APPLICATION_NAME, "mubbles")
        .unwrap();

    let mainloop = Rc::new(RefCell::new(
        Mainloop::new().ok_or_else(|| anyhow::anyhow!("Failed to create mainloop"))?,
    ));

    let context = Rc::new(RefCell::new(
        Context::new_with_proplist(mainloop.borrow().deref(), "mubbles-context", &proplist)
            .ok_or_else(|| anyhow::anyhow!("Failed to create context"))?,
    ));

    // Set up context state callback
    {
        let mainloop_ref = Rc::clone(&mainloop);
        context
            .borrow_mut()
            .set_state_callback(Some(Box::new(move || {
                mainloop_ref.borrow_mut().signal(false);
            })));
    }

    context.borrow_mut().connect(None, ContextFlagSet::NOFLAGS, None)?;

    // Wait for context to be ready
    loop {
        match mainloop.borrow_mut().iterate(false) {
            IterateResult::Quit(_) | IterateResult::Err(_) => {
                return Err(anyhow::anyhow!("Context connection failed"));
            }
            IterateResult::Success(_) => {}
        }
        match context.borrow().get_state() {
            libpulse_binding::context::State::Ready => break,
            libpulse_binding::context::State::Failed | libpulse_binding::context::State::Terminated => {
                return Err(anyhow::anyhow!("Context connection failed"));
            }
            _ => {}
        }
    }

    tracing::info!("PulseAudio context connected");

    // Create audio specification
    let spec = Spec {
        format: Format::FLOAT32LE,
        channels: 1, // Mono
        rate: 48000, // 48 kHz, will be resampled to 16kHz later
    };

    if !spec.is_valid() {
        return Err(anyhow::anyhow!("Invalid audio spec"));
    }

    let buffer_attr = BufferAttr {
        maxlength: std::u32::MAX,
        tlength: std::u32::MAX,
        prebuf: std::u32::MAX,
        minreq: std::u32::MAX,
        fragsize: 1024, // Small fragment size for low latency
    };

    let stream = Rc::new(RefCell::new(
        Stream::new(
            &mut context.borrow_mut(),
            "mubbles-capture",
            &spec,
            None,
        )
        .ok_or_else(|| anyhow::anyhow!("Failed to create stream"))?,
    ));

    // Set up stream callbacks
    {
        let mainloop_ref = Rc::clone(&mainloop);
        stream
            .borrow_mut()
            .set_state_callback(Some(Box::new(move || {
                mainloop_ref.borrow_mut().signal(false);
            })));
    }

    {
        let audio_tx_clone = audio_tx.clone();
        let partial_tx_clone = partial_tx.clone();
        let sample_rate = spec.rate;
        
        stream
            .borrow_mut()
            .set_read_callback(Some(Box::new(move |length| {
                let stream_ref = stream.borrow();
                if let Ok(data) = stream_ref.peek() {
                    if let Some(data_slice) = data {
                        // Convert bytes to f32 samples
                        let f32_samples: &[f32] = bytemuck::cast_slice(data_slice);
                        let samples_vec = f32_samples.to_vec();

                        let pcm_audio = PcmAudio {
                            data: samples_vec.clone(),
                            sample_rate: sample_rate as usize,
                        };

                        // Send to main audio processing
                        if let Err(_) = audio_tx_clone.send(pcm_audio) {
                            tracing::debug!("Audio channel closed");
                        }

                        // Send to partial processing if enabled
                        if let Some(ref partial_tx) = partial_tx_clone {
                            let partial_pcm = PcmAudio {
                                data: samples_vec,
                                sample_rate: sample_rate as usize,
                            };
                            if let Err(_) = partial_tx.send(partial_pcm) {
                                tracing::debug!("Partial channel closed");
                            }
                        }

                        // Discard the data we just read
                        let _ = stream_ref.discard();
                    }
                }
            })));
    }

    // Connect the stream
    let device_name = if device.name == "@DEFAULT_SOURCE@" {
        None
    } else {
        Some(device.name.as_str())
    };

    stream.borrow_mut().connect_record(
        device_name,
        Some(&buffer_attr),
        StreamFlagSet::INTERPOLATE_TIMING
            | StreamFlagSet::ADJUST_LATENCY
            | StreamFlagSet::AUTO_TIMING_UPDATE,
    )?;

    // Wait for stream to be ready
    loop {
        match mainloop.borrow_mut().iterate(false) {
            IterateResult::Quit(_) | IterateResult::Err(_) => {
                return Err(anyhow::anyhow!("Stream connection failed"));
            }
            IterateResult::Success(_) => {}
        }
        match stream.borrow().get_state() {
            State::Ready => break,
            State::Failed | State::Terminated => {
                return Err(anyhow::anyhow!("Stream connection failed"));
            }
            _ => {}
        }
    }

    tracing::info!("PulseAudio stream connected and recording");

    // Run the main loop
    loop {
        match mainloop.borrow_mut().iterate(false) {
            IterateResult::Quit(_) | IterateResult::Err(_) => break,
            IterateResult::Success(_) => {}
        }
        thread::sleep(Duration::from_millis(1));
    }

    Ok(())
}

// Legacy compatibility for existing code
pub type AppDevice = PulseDevice;