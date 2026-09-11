# P5 — Partial thread panics on model load; loads a second model

**Status:** open · **Verified:** 7cd9e95

**Severity:** Medium.
**Type:** Unhandled panic + duplicated model.

## Symptom

If the Tiny model fails to load (e.g. first run offline, corrupt cache), the
partial thread panics. The panic is swallowed by the join in
`WhisperWorker::stop`/`Drop`, so partials silently never run. Additionally,
when partials are enabled a **second** Whisper model is loaded purely for
preview.

## Evidence

```rust
// src/partial.rs:31-34
let mut whisper_context =
    load_whisper_model(WhichModel::Tiny, move |s| {
        app_status.send(WhisperUpdate::Status(s)).ok();
    }).expect("Failed to load whisper model"); // panics on failure
```

The panic propagates out of `partial_loop`; `start_partial_thread`'s closure
does not catch it (`src/partial.rs:14-20`), and the join result is ignored:

```rust
// src/workers.rs:145-147 and :168-170
if let Some(pt) = self.partial_thread.take() { let _ = pt.join(); }
```

The partial thread independently calls `load_whisper_model(WhichModel::Tiny, ..)`
even though the main whisper thread may already have a model resident, doubling
VRAM / GPU memory when partials are on.

## Impact

- Silent loss of partials; no UI signal.
- Extra model load and memory whenever partials are enabled.

## Options

1. Replace `.expect(..)` with `?`/logging and let `partial_loop` return `Err`
   (it already logs on error). `start_partial_thread` can report the failure
   up so `start_whisper` can disable partials and surface a status message.
2. Decide whether partials should reuse the main `WhisperContext` (Tiny vs the
   selected model) or keep a separate context deliberately. If separate,
   document the memory cost; if reuse, this needs the context to be shared and
   synchronized (or moved into the partial thread).
3. Consider whether `WhisperUpdate::ThreadPanicked`-style signalling is wanted
   so panics in any worker surface in the status line rather than only logs.

## Files

- `src/partial.rs`
- `src/workers.rs` (`start_whisper`, `stop`, `Drop`)
