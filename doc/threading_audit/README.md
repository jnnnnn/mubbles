# Threading & Teardown Audit

Audit of thread lifecycle, shutdown, device-loss handling, and error/cleanup
paths in mubbles. Written while working through the "fragile shutdown"
complaint and the hot-unplug fix.

> **Verified against `7cd9e95` (`stability`).** The `file:line` references in
> these docs are accurate as of that commit and will drift as the source
> changes. Prefer the symbol/function names when following a reference.

## Scope

Everything that owns or spawns a thread, plus the channels that connect them:

- `src/audio.rs` — capture stream + per-device filter thread (cpal and PipeWire).
- `src/audio_pipewire.rs` — PipeWire stream thread, enumeration thread.
- `src/workers.rs` — `AudioWorker`, `WhisperWorker`.
- `src/whisper.rs` — whisper transcription thread.
- `src/partial.rs` — partial-preview thread.
- `src/app.rs` — orchestration, resync, file-transcription thread.
- `src/summary.rs` — AI summary request thread.
- `src/main.rs` — tracing guard / eframe entry.

## How to track this

- **The findings table below is the single source of truth.** One row per
  finding; no separate tracker.
- **Statuses:** `open` → `doing` → `done`, or `deferred` / `wontfix`.
- **One commit per finding.** Subject references the ID, e.g.
  `Fix partial preview channel wiring (P1)`. Put the resulting sha in the
  `Commit` column *in that same commit* — the table edit ships with the fix,
  so it can't drift.
- **Each finding doc carries a status line** under its title
  (`**Status:** … · **Verified:** …`). Update it only when the status changes;
  the README stays the aggregate.
- **Use the decision log** at the bottom for the "why" that isn't in a diff
  (option chosen, scope deferred, etc.).

## Findings

| ID | Title | Severity | Status | Commit | Notes |
| --- | --- | --- | --- | --- | --- |
| [P1](01-partial-preview-wiring.md) | Partial preview is completely dead | High (regression) | open | | needs decision: Option A vs B |
| [P2](02-device-failure-restart-storm.md) | Failed/unplugged device re-added every frame | High | open | | |
| [P3](03-whisper-lost-receiver.md) | Whisper exit loses `filtered_rx`; restart wires a dead channel | High | open | | |
| [P4](04-ui-thread-join-blocks.md) | UI thread joins during decode / model load | Medium | open | | needs scope decision (cancellation token) |
| [P5](05-partial-thread-panic.md) | Partial thread panics on model load; loads a second model | Medium | open | | depends on P1 decision |
| [P6](06-ollama-blocking-ui.md) | Blocking Ollama HTTP on the UI thread, no timeout | Medium | open | | |
| [P7](07-ai-summary-thread.md) | AI summary thread fire-and-forget; abort/race issues | Medium | open | | |
| [P8](08-file-transcription-thread.md) | File transcription thread not joined or cancelled | Low | open | | |
| [P9](09-pipewire-enumeration-thread.md) | PipeWire enumeration thread fire-and-forget; can spin | Low | open | | |
| [P10](10-minor-cleanups.md) | Minor cleanup gaps | Low | open | | |

## What is correct today

These paths were verified and should not be disturbed without care:

- **cpal unplug**: the error callback drops the shared sender
  (`Arc<Mutex<Option<Sender>>>`), so the filter thread's `recv()` returns `Err`
  and it exits instead of blocking forever (`src/audio.rs:543-551`,
  `src/audio.rs:106-113`).
- **PipeWire unplug**: the `state_changed` callback quits the main loop, so
  `run()` returns, the user-data sender drops, and the filter thread unblocks
  (`src/audio_pipewire.rs:367-388`). `Paused` is intentionally not treated as
  fatal.
- **Whisper stop**: the `stop` flag plus `recv_timeout(50ms)` avoids the old
  join deadlock (`src/whisper.rs:360-371`).
- **Drop ordering**: `MubblesApp.audio_worker` drops before
  `whisper_worker`, so the `from_whisper` receiver outlives the filter threads
  (`src/app.rs:189` vs `:192`).
- `PwStreamState::drop` (send stop, then join) and `StreamState::stop` (drop
  capture source, then join filter) are correctly ordered on both platforms.

## Recommended order

1. **P1** and **P2** — small, high impact (restore partials; stop the
   restart storm and clear the chart).
2. **P3** — small, self-contained (prevents permanent capture death after a
   whisper exit).
3. **P4** — larger refactor; decide scope first.
4. **P6/P7/P8/P9** — independent robustness/cleanup fixes.
5. **P5/P10** — tidy-ups.

## Decision log

Newest first. One line per decision, with the finding ID.

- **2026-09-11** — Delivery order agreed: P1 + P2 first, then P3; P4 scoped
  after. P1 still needs a call between Option A (thread the receiver through)
  and Option B (partials owned by the audio lifecycle).
