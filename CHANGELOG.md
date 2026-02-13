# Changelog

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