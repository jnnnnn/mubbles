# P2 — Failed/unplugged device is re-added every frame

**Status:** open · **Verified:** 7cd9e95

**Severity:** High — log spam, repeated expensive stream creation, stale chart.
**Type:** Interaction between `check_panic` and `resync_audio_devices`.

## Symptom

After a device errors or is unplugged, mubbles keeps trying to recreate its
stream roughly 10–60 times per second until the next device refresh (up to
30 s), logging an error each time. The device's chart keeps its old data until
the refresh.

## Evidence

Per frame, `update` runs `check_panic` and then `resync_audio_devices`
(`src/app.rs:1557-1585`).

`check_panic` removes a dead stream:

```rust
// src/workers.rs:77-91
let dead: Vec<String> = self.audio_streams.iter()
    .filter(|(_, s)| s.filter_thread.is_finished())
    .map(|(name, _)| name.clone())
    .collect();
for name in dead {
    if let Some(stream) = self.audio_streams.remove(&name) { /* stop, send Recording(false) */ }
}
```

`resync_audio_devices` then sees the device is still selected and still present
in the (stale) `self.devices` list, so `active` contains it and it is re-added:

```rust
// src/app.rs:565-593
let active: HashSet<String> = self.selected_device_names.iter()
    .filter(|name| self.devices.iter().any(|d| d.name.as_str() == name.as_str()))
    .cloned().collect();
...
for name in &active {
    if !audio.audio_streams.contains_key(name) {
        // find device, add_device -> start a new capture stream
    }
}
```

`self.devices` only updates on enumeration/refresh
(`src/app.rs:539-549`, `:1569-1577`); unplug detection is therefore up to
30 s (or a focus change) behind the stream death. The repaint interval is
100 ms, or 16 ms on the Devices tab (`src/app.rs:33`, `:1550-1555`).

## Impact

- Repeated `build_input_stream` / PipeWire stream creation against a dead
  device.
- Repeated `an error occurred on stream` / `Filter thread ... exited` /
  `Started audio stream for ...` log lines.
- Chart state for the device is retained (it's still in `active`), so the
  level plot shows frozen data instead of clearing — contrary to the intent of
  `resync_audio_devices` (`src/app.rs:607-612`).

## Options

1. **Failed-device backoff.** Have `AudioWorker::check_panic` (and a failed
   `add_device`) record the device name in a "quarantine" set. `resync` skips
   quarantined devices; `refresh_devices` / `check_device_enumeration` clears
   the set after re-confirming presence. Simple and local.

2. **Only add on refresh/selection change.** Restrict the *add* branch of
   `resync_audio_devices` to frames where the device list or selection
   actually changed, keeping the *remove* branch every frame. A dead stream is
   then only recreated when the user refreshes or re-selects.

3. **Clear chart on stream death.** Have `check_panic` also drop the
   `device_levels` / `device_vad_levels` / `device_muted` entries for the dead
   device, so the plot clears immediately.

Recommended: 1 + 3. Option 2 alone leaves the chart stale until refresh and
makes resync semantics more implicit.

## Note

The error callback fix (dropping the sender) is what makes this visible: the
filter thread now exits promptly, so `check_panic` fires each frame. The
"thread no longer blocks" fix needs this companion to avoid the thrash.
