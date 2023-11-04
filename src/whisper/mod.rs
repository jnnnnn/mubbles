use std::{
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

mod audio;
pub use audio::{AppDevice, get_devices};

mod transcribe;

pub mod candle_example;

use cpal::traits::{DeviceTrait, StreamTrait};


pub enum WhisperUpdate {
    Recording(bool),
    Transcribing(bool),
    Transcript(String),
    Level(f32),
}

pub struct StreamState {
    // the app holds this handle to keep the stream open (and the whisper context / thread alive)
    #[allow(dead_code)]
    pub(crate) stream: cpal::Stream,
}

#[derive(Clone, Copy)]
pub struct WhisperParams {
    pub accuracy: usize, // 1 for greedy, more for beam search
    pub model: candle_example::WhichModel,
    pub quantized: bool,
}


// once the return value is dropped, listening stops
// and the sender is closed
pub fn start_listening(
    app: &Sender<WhisperUpdate>,
    app_device: &audio::AppDevice,
    params: WhisperParams,
) -> Option<StreamState> {
    tracing::info!(
        "Listening on device: {}",
        app_device.device.name().expect("device name")
    );

    let (audio_tx, audio_rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = mpsc::channel();

    let err_fn = move |err| tracing::error!("an error occurred on stream: {}", err);
    let data_callback =
        move |data: &[f32], _: &_| audio_tx.send(data.to_vec()).expect("Failed to send data");
    let stream = app_device
        .device
        .build_input_stream(
            &app_device.config.clone().into(),
            data_callback,
            err_fn,
            None,
        )
        .expect("Failed to build input stream");

    match stream.play() {
        Ok(_) => (),
        Err(err) => {
            tracing::error!("Failed to play stream: {}", err);
            return None;
        }
    }

    let app2 = app.clone();
    let (filtered_tx, filtered_rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = mpsc::channel();
    let config2 = app_device.config.clone();
    thread::spawn(move || {
        audio::filter_audio_loop(app2, audio_rx, filtered_tx, config2);
    });

    let app2 = app.clone();
    thread::spawn(move || {
        transcribe::whisper_loop(app2, filtered_rx, params);
    });

    Some(StreamState { stream })
}
