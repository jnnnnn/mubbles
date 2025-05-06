# Architectural Overview

This application records audio, converts it into text, and then can summarize the text.

## UI

The UI is built using egui and eframe. This is an immediate-mode GUI -- the whole window is redrawn from scratch every frame. It is based on graphics libraries. It does keep some state internally so that certain things are cached; but much state is managed by `MubblesApp` class.

## Audio

Audio is recorded via the `cpal` library. The various audio devices (microphones and speakers) are listed, and the user chooses one to transcribe from. The speakers are transcribed using Windows' audio loopback functionality.

## Transcribe

The transcription is performed using OpenAI's Whisper model, running locally. The particular implementation is heavily based on the HuggingFace model repository's `candle` library.

## Summarization

The transcribed text (or whatever is in the trancription window/textbox) can be summarized using a statistical summary (just picking out the least common words), or by passing the text to a Large Language Model that conforms to OpenAI's chatgpt API.

## Threading

The UI runs in the main thread. Audio transcription in `audio.rs`'s thread streams 100ms chunks into an mpsc channel. The app directs this into `whisper.rs`' thread. 

When a setting is changed (audio source or model), the old threads are dropped and new threads are created (and the model is loaded again).

Status updates and transcription results are sent back to the UI thread using another channel.