# P4 — UI thread joins during decode / model load

**Status:** open · **Verified:** 7cd9e95

**Severity:** Medium — visible UI freezes on common interactions.
**Type:** Blocking joins on the egui thread.

## Symptom

Changing accuracy, toggling Partials, or pressing Stop can freeze the window
for as long as a Whisper decode or a model load takes. Dragging the accuracy
slider is the worst case: it calls the blocking path on every `changed()`.

## Evidence

`restart_whisper` (UI thread) calls `WhisperWorker::stop`, which joins the
partial thread and then the whisper thread:

```rust
// src/app.rs:729-758
fn restart_whisper(&mut self) {
    if let Some(whisper) = self.whisper_worker.take() {
        self.pending_filtered_rx = Some(whisper.stop());
    }
    ...
}
```

```rust
// src/workers.rs:140-161
pub(crate) fn stop(mut self) -> Receiver<PcmAudio> {
    self.whisper_stop.store(true, Ordering::Relaxed);
    drop(self.partial_tx.take());
    if let Some(pt) = self.partial_thread.take() { let _ = pt.join(); }   // may be loading Tiny
    if let Some(thread) = self.whisper_thread.take() { match thread.join() { .. } } // may be mid-decode
}
```

The `stop` flag is only observed at loop boundaries: top of loop
(`src/whisper.rs:332`), and on `recv_timeout` (50 ms) expiry
(`src/whisper.rs:362-365`). During `whisperize` (`src/whisper.rs:380`) the flag
is not checked, so the join waits for the whole decode.

The accuracy slider triggers this on every change:

```rust
// src/app.rs:807-812
if ui.add(egui::Slider::new(&mut self.accuracy, 1..=8).text("Accuracy")).changed() {
    self.restart_whisper();
}
```

`WhisperWorker::drop` (`src/workers.rs:164-175`) has the same joins, so
`toggle_recording` (`src/app.rs:663-725`) and app shutdown can also block.

## Impact

- Slider drag: repeated multi-second freezes, and each restart re-loads the
  model.
- Stop / window close: up to a full decode plus model load.

## Options

1. **Cancellation token checked inside the decode loop.** Thread a
   `CancellationToken`-style `Arc<AtomicBool>` into `whisperize`, checked
   between decoder steps. `stop` sets it and joins; decode aborts quickly.
   This is the proper fix but touches `whisperize` / decoder call sites.

2. **Don't join on the UI thread.** Detach the old worker and reap it in the
   background (or on a later frame via `check_panic`), starting the new
   whisper immediately with a fresh/recovered receiver. Needs P3's receiver
   recovery to be safe.

3. **Debounce the accuracy slider.** Even with 1/2 unaddressed, restart only
   on slider *release* (`response.drag_stopped()`), not every `changed()`.

Recommended: 1 (or 2) plus 3 as a cheap immediate mitigation.

## Files

- `src/app.rs` (`restart_whisper`, accuracy/partials controls, `toggle_recording`)
- `src/workers.rs` (`stop`, `Drop`)
- `src/whisper.rs` (`whisperize`, loop)
