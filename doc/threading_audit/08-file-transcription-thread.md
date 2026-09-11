# P8 — File transcription thread not joined or cancelled

**Status:** open · **Verified:** 7cd9e95

**Severity:** Low (process exit reclaims it) — but no graceful cleanup.
**Type:** Unmanaged thread lifetime + missing cancellation on shutdown.

## Symptom

Closing the app (or starting another file) while a file is transcribing leaves
the old thread detached. `file_cancel` is never set to `true` on exit, so the
decode loop is not asked to stop. No hang, because process exit kills detached
threads, but model/GPU resources aren't released and there is no orderly stop.

## Evidence

The handle is stored but there is no `on_exit`/`Drop` that cancels or joins:

```rust
// src/app.rs:888-897
let thread = std::thread::spawn(move || {
    let tx2 = tx.clone();
    if let Err(e) = file_transcription::transcribe_file(path, model, tx, cancel) {
        tracing::error!("File transcription failed: {}", e);
        tx2.send(FileUpdate::Error(e.to_string())).ok();
    }
});
self.file_transcription_thread = Some(thread);
```

`check_thread_error` only joins after the thread has finished, and leaves a
running one in place:

```rust
// src/app.rs:1601-1618
fn check_thread_error(join: &mut Option<JoinHandle<()>>) {
    if let Some(thread) = join.take() {
        if thread.is_finished() { let _ = thread.join(); }
        else { *join = Some(thread); }
    }
}
```

Starting a new file overwrites `file_transcription_thread`, dropping the old
`JoinHandle` (detaching it). The UI should prevent this; verify the button is
disabled while `file_transcription_running`.

## Options

1. Set `file_cancel.store(true)` and join the thread in `MubblesApp::drop` (or
   an explicit `on_exit`), with a bounded wait so shutdown can't hang.
2. Keep a small "cancelling" state so a second start is impossible while the
   first thread is winding down (in addition to the UI disable).
3. Make `transcribe_file` check `cancel` inside the decoding/chunk loop more
   often (it currently checks once per 30 s chunk, `src/file_transcription.rs:191-195`).

## Files

- `src/app.rs` (`start_file_transcription`, `check_thread_error`, shutdown)
- `src/file_transcription.rs` (`transcribe_file`)
