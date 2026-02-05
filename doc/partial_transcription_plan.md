# Plan: Get Partial Transcription Working Efficiently

## Overview

The partial transcription system currently has the structure in place but is disabled/partially working. The goal is to:
1. **Efficiently compute incremental FFT** as new audio samples arrive
2. **Efficiently update the mel spectrogram texture** shown to the user  
3. **Show transcription results on the chart** (aligned words overlay)
4. **Remove dead code** that's no longer needed

## Design Decisions

1. **Unify mel1 and mel2** - single mel display system
2. **Remove DisplayMel.buffer VecDeque** - texture handles display directly
3. **Mel extends at the end** - when final transcription arrives, reset mel to empty
4. **Aligned words stay fixed** - words don't scroll, mel just extends rightward
5. **When partials disabled** - completely remove mel from UI (not just hidden)
6. **Pre-allocate texture** - fixed size (500 frames = 5s) to avoid reallocation
7. **Aligned words replace completely** - each `WhisperUpdate::Alignment` replaces previous
8. **Partial buffer lifecycle** - extends as audio arrives, repeatedly transcribed, cleared on final transcription
9. **Fixed width display with scrolling** - mel display has fixed UI width, spectrogram scrolls within it
10. **Word timings** - relative to start of current partial buffer (word.start=0 means first sample)
11. **Show full 500 frames** - no cropping, scroll to show most recent frames
12. **Implement MelProcessor** - cache FFT planner and Hanning window for efficiency

---

## Current State Analysis

### What exists:
- `partial.rs`: Has `partial_loop()` that accumulates audio and generates mel frames incrementally
- `mel.rs`: Has `pcm_to_mel_frame()` for incremental FFT processing
- `app.rs`: Has `DisplayMel` struct with buffer, texture, and image fields
- `WhisperUpdate::MelFrame` message type exists but handler is disabled (line 325-326)

### Problems identified:
1. **Mel frame handler disabled**: `update_mel_buffer` is marked `#[allow(dead_code)]` and not called
2. **Texture updates inefficient**: `draw_mel1` rebuilds entire texture every frame
3. **Two mel drawing functions**: `draw_mel1` (unused) and `draw_mel2` (uses Tensor from full transcription)
4. **Partial transcription calls full `pcm_to_mel`** instead of using incremental frames (line 54-55 in partial.rs)
5. **`perform_partial_transcription` is unused** - `perform2` is called instead which doesn't use incremental mel frames

---

## Implementation Plan

### Phase 1: Fix Incremental FFT Pipeline ✅ COMPLETE

#### Task 1.1: Enable incremental mel frame processing ✅
- Changed `partial_loop` to use `perform_partial_transcription` instead of `perform2`
- Removed the duplicate `pcm_to_mel` call

#### Task 1.2: Implement `MelProcessor` struct ✅
- Created `MelProcessor` in `mel.rs` that caches:
  - FFT planner (expensive to create, ~1ms)
  - Hanning window (computed once)
  - Mel filters reference
- Added test `mel_processor_matches_original` - passes

#### Task 1.3: Use MelProcessor in partial.rs ✅
- Created `MelProcessor` once at start of `partial_loop`
- Updated `generate_new_mel_frames` to use `MelProcessor`
- Removed unused `perform2` function
- Removed unused `PartialAudio` struct

---

### Phase 2: Efficient Texture Updates

#### Task 2.1: Use `set_partial` for incremental texture updates
The egui `TextureHandle::set_partial(pos, image, options)` API allows updating a subregion:
```rust
texture.set_partial(
    [x_offset, 0],  // position to update
    column_image,   // just the new column(s)
    TextureOptions::default()
);
```

#### Task 2.2: Implement pre-allocated texture strategy
- Create fixed-size texture on first frame (e.g., 500 frames × 80 bins)
- Track `frame_count` for how many frames are populated
- When final transcription arrives, reset `frame_count` to 0 and clear texture
- No reallocation during recording

#### Task 2.3: Simplify `DisplayMel` structure
```rust
const MAX_MEL_FRAMES: usize = 500;  // 5 seconds at 100 Hz

struct DisplayMel {
    texture: Option<egui::TextureHandle>,
    frame_count: usize,  // current number of frames populated
    min: f32,
    max: f32,
}
```
- Remove `buffer: VecDeque` (not needed)
- Remove `image: Option<ColorImage>` (not needed)

#### Task 2.4: Update frame handling
```rust
fn push_frame(&mut self, frame: &[f32], ctx: &egui::Context) {
    if self.frame_count >= MAX_MEL_FRAMES {
        return;  // Don't overflow pre-allocated texture
    }
    
    self.update_range(frame);
    let bytes = self.frame_to_bytes(frame);
    
    // Create column image for just this frame
    let column = ColorImage::from_gray([1, PARTIAL_MEL_BINS], &bytes);
    
    if let Some(tex) = &mut self.texture {
        // Update column in pre-allocated texture
        tex.set_partial([self.frame_count, 0], column, TextureOptions::default());
        self.frame_count += 1;
    } else {
        // Create pre-allocated texture (black/silent initially)
        let empty = ColorImage::new([MAX_MEL_FRAMES, PARTIAL_MEL_BINS], Color32::BLACK);
        let tex = ctx.load_texture("mel", empty, TextureOptions::default());
        tex.set_partial([0, 0], column, TextureOptions::default());
        self.texture = Some(tex);
        self.frame_count = 1;
    }
}

fn reset(&mut self) {
    // Clear texture to black, keep the allocation
    if let Some(tex) = &mut self.texture {
        let empty = ColorImage::new([MAX_MEL_FRAMES, PARTIAL_MEL_BINS], Color32::BLACK);
        tex.set(empty, TextureOptions::default());
    }
    self.frame_count = 0;
    self.min = -10.0;
    self.max = 0.0;
}
```

---

### Phase 3: Show Results on Chart

#### Task 3.1: Enable aligned words display for partial transcription
- `draw_aligned_words` already exists and works
- Aligned words stay fixed at their positions
- Mel extends rightward as new audio comes in

#### Task 3.2: Scale word positions correctly
- Partial transcription alignment times map to mel frames
- `mel_seconds = frame_count / MEL_UPDATE_HZ` (100 Hz = 10ms per frame)
- Words are positioned at `start_time * pixels_per_second`

#### Task 3.3: Reset words when mel resets
- When transcription buffer clears, also clear `aligned_words`
- New partial transcriptions start fresh

---

### Phase 4: Remove Dead Code ✅ COMPLETE

#### Task 4.1: Remove unused items in app.rs ✅
- [x] `draw_mel1` function (line 988-1022) - marked `#[allow(dead_code)]`
- [x] `update_mel_buffer` function (line 962-965) - marked `#[allow(dead_code)]`
- [x] `DisplayMel.image` field
- [x] `DisplayMel.buffer` field
- [x] `mel2: Tensor` field - unify with mel1

#### Task 4.2: Remove unused items in partial.rs ✅
- [x] `PartialAudio` struct (line 23-26) - defined but never used
- [x] `perform2` function once `perform_partial_transcription` is fixed

#### Task 4.3: Audit other dead_code markers ✅
- [x] `whisper_model.rs` line 58, 70, 510 - Actually used or intentionally kept
- [x] `voice_detect.rs` line 197 - No longer has dead_code marker
- [x] `audio.rs` line 162, 266 - Intentional (fields kept for drop semantics)

---

### Phase 5: Wire Everything Together

#### Task 5.1: Re-enable MelFrame handler in app.rs
```rust
WhisperUpdate::MelFrame(frame) => {
    self.mel.push_frame(&frame, ctx);
}
```
**Note**: Need to pass `ctx` to `process_whisper_updates` or restructure.

#### Task 5.2: Unify mel rendering
- Remove `mel2: Tensor` field
- Single `mel: DisplayMel` field  
- Single `draw_mel(&mut self.mel, ui)` function

#### Task 5.3: Handle reset on final transcription
```rust
WhisperUpdate::Transcription(t) => {
    // ... existing code ...
    self.mel.reset();  // Clear mel when final transcription arrives
    self.aligned_words.clear();  // Clear aligned words too
}
```

#### Task 5.4: Handle partials disabled
- When `self.partials == false`, don't render mel at all
- Skip mel row in UI or show placeholder

#### Task 5.5: Handle aligned words replacement
```rust
WhisperUpdate::Alignment(a) => {
    self.word_history.push(a.clone());
    self.aligned_words = a;  // Complete replacement (already correct)
}
```

---

## File Changes Summary

| File | Changes |
|------|---------|
| `src/partial.rs` | Remove `perform2`, fix `perform_partial_transcription`, remove `PartialAudio` |
| `src/mel.rs` | Add `MelProcessor` with cached FFT planner |
| `src/app.rs` | Unify mel1/mel2, enable MelFrame handler, use `set_partial`, remove dead code |

---

## Performance Considerations

1. **FFT Caching**: Creating `FftPlanner` is expensive (~1ms). Cache it in `MelProcessor`.
2. **Texture Updates**: `set_partial` only uploads changed pixels to GPU
3. **Pre-allocated texture**: Fixed 500×80 texture, no reallocation during recording
4. **Normalization**: Consider adaptive normalization that doesn't require scanning all values

---

## Testing Strategy

1. Manual test: Enable partials, start recording, verify mel appears and grows
2. Manual test: Verify aligned words appear at correct positions and replace on each update
3. Manual test: When final transcription arrives, verify mel and words reset
4. Manual test: Disable partials, verify no mel displayed
5. Performance test: Check frame processing time in logs

---

## Answered Questions

1. **When partials disabled**: Show nothing (remove mel display entirely)
2. **Pre-allocation**: Yes, pre-allocate 500 frames (5 seconds) to avoid reallocation
3. **When mel resets**: When `WhisperUpdate::Transcription` arrives (final transcription)
4. **Aligned words**: Complete replacement on each `WhisperUpdate::Alignment`
5. **Partial buffer lifecycle**: Extends as audio arrives → repeatedly transcribed → cleared on final
