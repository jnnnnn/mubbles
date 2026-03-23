# Changelog

## 2.6 (2026-03-23)
 - Option to pause whisper and free GPU memory during AI summarization (for local Ollama)
 - Whisper thread drops and reloads model on pause/resume, buffering audio in the channel

## 2.5 (2026-03-20)
 - AI summarization: Ollama, OpenAI, or custom API endpoint
 - Streaming summary output (SSE / ndjson)
 - Incremental chunked summarization for long transcripts
 - Auto-detect Ollama models and context length
 - Abort actually cancels the HTTP stream and stops generation
 - Configurable system/user prompts, thinking budget, max tokens
 - Summary blocks separated by newlines

## 2.4 (2026-03-04)
 - Switch audio devices without stopping transcription

## 2.3 (2026-02-19)
 - Fix autotype
 - Split file transcription at silence boundaries instead of fixed 30s chunks

## 2.2 (2026-02-13)
 - File transcription: transcribe audio files (wav, mp3, flac, ogg, m4a, aac, wma)

## 2.1 (2026-01-14)
 - Monthly log files for transcription history
 - History tab shows accuracy of each word
 - Linux PipeWire audio backend for monitor source support
 - Dual audio channel support (two sound devices)
 - Partial transcription with incremental mel spectrogram visualization
 - Word alignment/timestamps display
 - Download progress indicator for models
 - CPU support (Intel MKL)
 - Log tab in UI
 - Status messages during transcription

## 2.0 (2025-04-28)
 - Major rewrite using candle (Rust ML framework) instead of whisper.cpp
 - Native Rust implementation of Whisper model
 - Model selection (tiny, base, small, medium, large, distil variants)

## 1.4 (2023-10-13)
 - Statistical summary shows five most unusual words for each ten lines of transcript

## 1.3 (2023-10-03)
 - Auto-scrolling each time transcription is added

## 1.2 (2023-08-29)
 - Log to file in case of accidental clear

## 1.1 (2023-08-07)
 - Always on Top checkbox keeps the window visible while Zoom is open
 - Open window faster (load model in thread)

## 1.0 (2023-05-25)
 - Initial release
 - Select input or output audio stream
 - Transcribes to textbox
 - Autotype uses OS to type transcript through keyboard to whatever app is foreground
 - Shows chart of input volume
 - Set beam size for speech recognition -- lower is faster but less polished output
 - Uses CUDA / whisper-rs / whisper.cpp for speech recognition