# Refactoring Plan: Dual Audio Streams for Transcription

## Overview
This refactor introduces a single audio stream feeding two Whisper transcription instances, with the following goals:

1. **Fast Real-Time Transcription Instance**:
   - Provides immediate feedback to the user.
   - Prioritizes speed over accuracy.
   - Sends partial transcription updates back to the app.

2. **Accurate Post-Sentence Transcription Instance**:
   - Processes audio after the end of each sentence.
   - Focuses on accuracy, correcting errors from the fast transcription.
   - Sends full transcription updates back to the app.

Additionally, the application will display the audio waveform in real-time, overlaying the partial and final transcriptions on the waveform.

## Goals
- Enhance user experience by providing immediate feedback while ensuring accurate transcriptions.
- Maintain a clear separation between the two transcription instances to avoid performance bottlenecks.
- Ensure the application remains responsive and stable during transcription.
- Visualize the audio waveform with overlaid transcriptions.

## Current Architecture
The current architecture involves the following components:

1. **`app.rs`**:
   - Triggers the transcription process.

2. **`whisper.rs`**:
   - Starts audio monitoring using the `cpal` library in one thread.
   - Runs the Whisper transcription model in another thread.
   - Links the two threads using `mpsc` channels for audio data transfer.

### Diagram: Current State
```plaintext
+----------------+       +----------------+       +----------------+
|                |       |                |       |                |
|    app.rs      +------>+   whisper.rs   +------>+   Transcribe   | ---> updates back to app
|                |       |                |       |                |
+----------------+       +----------------+       +----------------+
                            |                        ^
                            v                        |
                     +----------------+              |
                     |                |              |
                     |   cpal Audio   +--------------+
                     |   Monitoring   |
                     +----------------+
```

## Future Architecture
The future architecture will introduce two separate Whisper transcription instances fed by a single audio stream:

1. **Real-Time Transcription Instance**:
   - Provides immediate feedback to the user.
   - Sends partial updates to the app.

2. **Accurate Transcription Instance**:
   - Buffers audio data until the end of a sentence is detected.
   - Sends final updates to the app.

Additionally, the audio waveform will be displayed in real-time, with partial and final transcriptions overlaid.

## Refactoring Steps

1. Factor Out Audio Sampling

Move audio sampling logic from `whisper.rs` to a new `audio.rs` module.

Ensure the new module provides reusable functions for audio processing.

Update `whisper.rs` to use the new `audio.rs` module.

2. Move orchestration of audio out to app or separate module, so that the cpal stream is filtered for noise and segmented into chunks before it gets to the whisper processing.

3. Add a spot in the UI for partial updates to be displayed, overwritten each time a new one arrives.

4. Have the orchestrator create a second whisper instance that uses the tiny model and gets fed the whole of the current in-progress segment each time. 

There will need to be some sort of rate limiting as tiny transcribes take 0.4s but new samples arrive every 0.1s. 

5. Work on the UI to show things more dynamically. 

Maybe display the previous Mel and the final transcript, as well as the current mel and the partial transcript.

6. Have the transcripts with word-level timestamps, and line up the words with the mel.

If showing the mel doesn't work very well, just use an amplitude plot as is currently done.

## Changes to Audio Data Flow

In the current architecture, the audio data is buffered until a complete segment is available before being passed to the Whisper thread for transcription. This ensures that only finalized audio segments are processed, but it introduces latency since transcription cannot begin until the segment is complete.

In the proposed architecture, this will change as follows:

1. **Real-Time Transcription Instance**:
   - Audio data will be streamed continuously to the real-time transcription instance as it becomes available.
   - This instance will process partial segments of audio in near real-time, providing immediate feedback to the user.
   - The data flow will no longer wait for a complete segment, enabling faster updates.

2. **Accurate Transcription Instance**:
   - Audio data will still be buffered until a complete segment is available, similar to the current architecture.
   - Once a segment is finalized, it will be passed to the accurate transcription instance for processing.

This dual-stream approach introduces a continuous data flow for real-time transcription while maintaining the existing buffering mechanism for accurate transcription. The key difference is that audio data will now be split and sent to two different transcription instances, with the real-time instance prioritizing speed and the accurate instance focusing on correctness.

## Optimizing Mel Generation with Buffer Reuse

To further enhance performance, the mel spectrogram generation process can be optimized by reusing the same buffer for repeated operations. This approach offers several benefits:

1. **Reduced Memory Allocation Overhead**:
   - By reusing a pre-allocated buffer, the application avoids frequent memory allocation and deallocation, which can improve overall performance.

2. **Improved Cache Efficiency**:
   - Reusing the same buffer can lead to better CPU cache utilization, as the memory access patterns remain consistent.

3. **Simplified Synchronization**:
   - If the buffer is shared between the real-time and accurate transcription instances, it simplifies data flow and reduces duplication.

### Implementation Considerations
- **Thread Safety**:
  - Proper synchronization mechanisms must be in place to avoid race conditions when multiple threads access the shared buffer.

- **Buffer Size**:
  - The buffer should be large enough to handle the largest expected mel spectrogram, ensuring no data loss or overflow.

- **Performance Testing**:
  - Benchmark the mel generation process before and after implementing buffer reuse to quantify the performance gains.

This optimization is particularly useful if mel spectrogram generation is a frequent operation and contributes significantly to processing time. It aligns with the goal of maintaining a responsive and efficient transcription system.