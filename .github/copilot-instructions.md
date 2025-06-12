This project is a blazingly-fast transcription built using Rust. It leverages the Whisper model for audio processing and transcription.

## Key Files
- whisper.rs Contains the main logic for Whisper model integration and transcription.
- audio.rs Handles audio input and processing.
- mel.rs Responsible for generating mel spectrograms.
- whisper_model.rs Defines the Whisper model and its decoder.
- app.rs Renders the UI using egui / eframe.

## Development Guidelines
1. **Error Handling**: Use `anyhow::Error` for error propagation and logging.
2. **Logging**: Use `tracing` for structured logging.
3. **Concurrency**: Use `std::sync::mpsc` for thread communication.
4. **Performance**: Optimize audio processing and model inference for real-time transcription.

Write unit tests for new functionality. Add unit tests in the `tests` directory for Rust code.

Use context7 to get lookup up huggingface/candle docs.
