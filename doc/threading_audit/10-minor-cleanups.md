# P10 — Minor cleanup gaps

**Status:** open · **Verified:** 7cd9e95

**Severity:** Low.
**Type:** Small leaks / latent panics / keying caveats.

## 10.1 `device_threshold_atoms` entries are never removed

`remove_device` and `check_panic` drop the stream but leave the threshold
`Arc<AtomicU32>` in `AudioWorker.device_threshold_atoms`
(`src/workers.rs:69-91`). Map entries accumulate for the process lifetime as
devices come and go. Fix: `self.device_threshold_atoms.remove(name)` in both
paths.

## 10.2 `convert_sample_rate` can panic

`convert_sample_rate` unwraps internally, and callers unwrap its result:

- `src/audio.rs:68` — `SincFixedIn::new(..).unwrap()`
- `src/audio.rs:71` — `resampler.process(..).unwrap()`
- `src/audio.rs:179` — `convert_sample_rate(..).unwrap()` (filter loop)
- `src/audio.rs:246` — `convert_sample_rate(..).unwrap()` (split path)
- `src/partial.rs:193` — `convert_sample_rate(..).unwrap()`

A device reporting `sample_rate == 0` (or an otherwise invalid rate) makes the
ratio `inf`/`NaN` and panics the filter thread. It is caught by `check_panic`,
which — before P2 is fixed — turns into the restart storm. Fix: return
`Result` all the way through and treat a bad rate as a stream error rather than
a panic; reject `sample_rate == 0` at the device boundary.

## 10.3 Streams are keyed by display name

`AudioWorker.audio_streams: HashMap<String, StreamState>` and
`device_threshold_atoms` are keyed by `device.name` (`src/workers.rs:29,34`).
Two devices with the same display name collide (second add is ignored;
removal affects both). Same-name replug is a related latent issue. Fix:
key by a stable identity (`node_id` on Linux / a device id on cpal) with the
name only for display.

## 10.4 `WhisperUpdate::Mel(Tensor)` is carried but unused

`WhisperUpdate::Mel` is never sent (`src/partial.rs:105` is commented out) and
is ignored by the UI (`src/app.rs:468-470`). The variant forces `WhisperUpdate`
to derive `Clone` over a `Tensor`. Consider removing both the variant and the
`Mel` send site to simplify the enum.

## 10.5 Duplicated panic-formatting helper

The `downcast_ref::<&'static str>() / <String>() / {:?}` block is repeated in
`join_filter_thread` (`src/audio.rs:256-267`),
`WhisperWorker::check_panic` (`src/workers.rs:124-132`), and
`check_thread_error` (`src/app.rs:1601-1618`). Extract a shared
`fn panic_message(e: Box<dyn Any + Send>) -> String`.

## Files

- `src/workers.rs`
- `src/audio.rs`
- `src/partial.rs`
- `src/app.rs`
