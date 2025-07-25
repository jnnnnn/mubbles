# Silero VAD Integration To-Do List

## 1. Generalize HuggingFace Model Download
- [ ] Refactor the existing HuggingFace API download logic into a new module (e.g., `huggingface.rs`).
- [ ] Ensure the module supports downloading arbitrary files (not just Whisper models).
- [ ] Add error handling using `anyhow::Error` and logging with `tracing`.

## 2. Download Silero ONNX Model
- [ ] Use the new HuggingFace download module to fetch `model.onnx` from https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx.
- [ ] Cache the model locally after download.
- [ ] Add logic to check for and use the cached model if available.

## 3. Implement Silero VAD Logic
- [ ] Integrate ONNX runtime (e.g., `ort` or `candle-onnx`) to load and run inference on the downloaded model.
- [ ] Implement the `detect` function in `voice_detect.rs` to process audio and return speech detection results.
- [ ] Add error handling and logging for VAD operations.

## 4. Integrate VAD into Audio Pipeline
- [ ] Call `voice_detect::detect` from the main audio processing code.
- [ ] Use VAD results to segment or filter audio before transcription.

## 5. Add Unit Tests
- [ ] Write unit tests for the new HuggingFace download module.
- [ ] Write unit tests for Silero VAD detection using sample audio data.

## 6. Update Documentation
- [ ] Document the new VAD feature and HuggingFace download module in `README.md` and relevant docs.

---

Performance optimization and configuration options will be addressed in future steps.
