# P3 — Whisper thread exit loses `filtered_rx`; restart wires a dead channel

**Status:** open · **Verified:** 7cd9e95

**Severity:** High — a single whisper exit can permanently kill all transcription
and then all capture.
**Type:** Receiver ownership on the recovery path.

## Symptom

If the whisper thread ever exits on its own (panic, or a normal early return
such as a model-load failure), the app still believes whisper is running.
Restarting it produces a whisper thread that immediately stops, and audio
capture then dies device-by-device.

## Evidence

`check_panic` joins the thread and **discards** the receiver it returns:

```rust
// src/workers.rs:120-136
if thread.is_finished() {
    let handle = self.whisper_thread.take().unwrap();
    if let Err(e) = handle.join() { /* log panic */ }
    // Ok(rx) is dropped here — the receiver is gone
}
```

The whisper thread is designed to hand the receiver back on exit
(`fn whisper_loop -> Receiver<PcmAudio>`, `src/whisper.rs:301-313`), so this
discards the only way to reconnect audio.

`stop()` then fabricates a replacement, but the sender is dropped immediately,
so the "recovered" receiver is already disconnected:

```rust
// src/workers.rs:149-160
match thread.join() {
    Ok(rx) => rx,
    Err(_) => {
        let (_, rx) = mpsc::channel(); // sender dropped -> rx is disconnected
        rx
    }
} 
// and the same in the `else` branch at :158
```

So `restart_whisper` (`src/app.rs:729-758`) feeds the new whisper a dead
channel. The filter threads still hold clones of the original `filtered_tx`
(`src/workers.rs:30`, cloned in `start_audio_thread`), whose receiver was
dropped on unwind. At the next utterance end:

```rust
// src/audio.rs:180-183
filtered_tx.send(PcmAudio { .. })?; // Err -> filter thread exits
```

Each filter thread then exits with `Err`, `check_panic` removes it, and P2's
resync loop takes over.

## Options

1. **Never let the thread lose the receiver.** Inside the spawned closure,
   `catch_unwind(AssertUnwindSafe(..))` around `whisper_loop_inner`, always
   returning `filtered_rx`:

   ```rust
   .spawn(move || {
       let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
           whisper_loop_inner(&app, &filtered_rx, &params, &paused, &stop)
       }));
       if let Err(e) = r { tracing::error!("whisper panicked: {:?}", e); }
       filtered_rx
   });
   ```

   Then `stop()`'s `Ok(rx)` arm is effectively always taken. (Note:
   `catch_unwind` does not catch `panic = "abort"`.)

2. **Stash the recovered receiver.** Give `WhisperWorker` a
   `recovered_rx: Option<Receiver<PcmAudio>>` that `check_panic` fills when it
   joins, and have `stop()` prefer it. Also fixes the fabricated-empty-channel
   case.

3. **Don't fabricate a disconnected channel.** At minimum, make the panic/None
   fallback explicit — either mark the worker as dead and force a full
   audio+whisper restart, or recover the real receiver. Returning a silently
   disconnected channel is the worst option.

Recommended: 1 + 3.

## Files

- `src/whisper.rs` (thread body)
- `src/workers.rs` (`check_panic`, `stop`)
