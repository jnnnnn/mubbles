# P1 — Partial preview is completely dead

**Status:** done · **Verified:** 7cd9e95

**Severity:** High — a whole feature is silently non-functional.
**Type:** Regression from `3c9f28d "Decouple audio and whisper thread lifecycles"`.

## Symptom

With "Partials" enabled, the mel spectrogram / partial overlay never receives
live audio. No error is shown anywhere; the sends just fail silently.

## Evidence

`start_audio` creates a partial channel and immediately discards the receiver:

```rust
// src/workers.rs:195
let (partial_tx, _partial_rx) = mpsc::channel();
```

The sender from that dead channel is handed to the first device's capture
callback (`src/workers.rs:205-210`), where the send result is deliberately
swallowed:

```rust
// src/audio.rs:571-578 (cpal)
if let Some(partial_tx) = &partial_tx {
    partial_tx
        .send(PcmAudio { data, sample_rate })
        .unwrap_or_else(|_| { /* this is too noisy. */ });
}
```

Meanwhile `start_whisper` starts the partial thread on a **different** channel
that nothing ever feeds:

```rust
// src/workers.rs:251-254
let (partial_thread, partial_tx) = if params.partials {
    let (tx, rx) = mpsc::channel();
    let thread = partial::start_partial_thread(app.clone(), rx)?;
    (Some(thread), Some(tx))
} else {
    (None, None)
};
```

The stored `WhisperWorker.partial_tx` is never sent to; dropping it is used
only to signal shutdown (`src/workers.rs:143, 167`).

## Root cause / history

Before the decoupling refactor, `start_listening` created **one** partial
channel and cloned its sender into the first audio stream (old `src/app.rs`,
`fn start_listening`). Splitting `start_listening` into `start_audio` +
`start_whisper` broke the wiring: audio now owns one channel and whisper owns
another.

This is not pre-existing behavior — partials worked before `3c9f28d`.

## Impact

- Mel preview and partial transcription are dormant regardless of the
  "Partials" toggle.
- Work is still done: the partial thread loads the Tiny model and blocks on
  `recv()` for the app's lifetime. Wasted model load and VRAM.

## Options

### Option A — Minimal: thread the receiver through

- Create the partial channel in `start_audio`; store `partial_tx` on
  `AudioWorker` and use it in `AudioWorker::add_device` for the first device
  (the `add_device` path currently hardcodes `None`, `src/workers.rs:49`).
- Return `Option<Receiver<PcmAudio>>` from `start_audio` and accept it in
  `start_whisper`.
- Change `partial_loop` to return its `Receiver` on exit (mirroring the
  whisper thread's `JoinHandle<Receiver<PcmAudio>>`), so `restart_whisper` can
  recover it instead of rebuilding the channel.

Pros: restores behavior with least surface change.
Cons: reaches from audio into whisper's partial lifecycle; `restart_whisper`
must carry the receiver through.

### Option B — Decouple partials from whisper entirely

- `partial` becomes part of the audio lifecycle: started in `start_audio`,
  stopped in `AudioWorker::drop`.
- Toggling "Partials" triggers a full audio restart rather than
  `restart_whisper`.

Pros: clean ownership; no cross-lifecycle channel handoff.
Cons: toggling partials interrupts capture briefly.

## Files touched by either option

- `src/workers.rs` (`AudioWorker`, `start_audio`, `start_whisper`)
- `src/partial.rs` (`start_partial_thread` / `partial_loop` return type)
- `src/app.rs` (`toggle_recording`, `restart_whisper`)

## Note on `AudioWorker::add_device`

`add_device` passes `None` for `partial_tx` with the comment
"only first device gets it, handled at setup" (`src/workers.rs:49`). Once
partials are wired, this must also supply the sender for the first stream,
otherwise a device added by resync after startup won't feed partials.

## Resolution (Option B)

Partials are now owned by the **audio lifecycle**, not whisper:

- `start_audio` creates the partial channel and starts the partial thread
  (when `partials_enabled`), giving a clone of the sender to the first device.
  The receiver never leaves `AudioWorker`.
- `AudioWorker` holds `partial_tx` (for resync re-adds), `partial_device`
  (so only one device feeds the preview) and the `partial_thread` handle.
- `AudioWorker::drop` stops the streams first (dropping the capture callbacks'
  sender clones), then drops `partial_tx`, which closes the channel and lets the
  partial thread exit; it joins the thread so it can't outlive the worker.
- `WhisperWorker` and `WhisperParams` no longer know about partials, so a
  model/accuracy change no longer restarts the preview.
- Toggling "Partials" now calls `restart_recording()` (a brief capture restart)
  rather than `restart_whisper()`; this is the price of correct ownership — the
  sender is baked into the capture callback, so a runtime toggle cannot rewire
  it without recreating the stream.

Option A (threading the receiver out of `start_audio` and through
`WhisperWorker`) was rejected: supporting a runtime toggle there would require
parking the receiver while capture keeps sending into it, growing an unbounded,
never-drained channel.

### Also fixed while here

- `partial_loop` no longer `.expect()`s the model load; the error propagates and
  the thread exits cleanly (the panic half of [P5](05-partial-thread-panic.md)).
- `accumulate_audio` no longer `.unwrap()`s `convert_sample_rate` (part of
  [P10.2](10-minor-cleanups.md)).

### Tests

- `partial::tests::accumulate_audio_errors_when_channel_closes`
- `partial::tests::accumulate_audio_appends_received_samples`
