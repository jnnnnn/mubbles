# P9 — PipeWire enumeration thread: fire-and-forget, can spin or panic silently

**Status:** open · **Verified:** 7cd9e95

**Severity:** Low.
**Type:** Unmanaged thread lifetime + fragile error handling.

## Symptom

Device enumeration runs on a detached thread. If PipeWire is misbehaving it can
spin forever at 100 ms intervals, or panic silently on a sync failure.
On shutdown the thread is simply abandoned (receiver dropped, sender send
fails).

## Evidence

No handle is kept:

```rust
// src/audio_pipewire.rs:71-80
pub fn start_device_enumeration() -> DeviceReceiver {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let devices = enumerate_devices_blocking();
        let _ = tx.send(devices);
    });
    rx
}
```

A failure to sync panics the thread:

```rust
// src/audio_pipewire.rs:201
core.sync(0).expect("Failed to sync with PipeWire core");
```

…and the completion wait spins until `done` is set, with no timeout:

```rust
// src/audio_pipewire.rs:206-210
while !done.load(std::sync::atomic::Ordering::SeqCst) {
    tracing::debug!("Iteration {:?}", count);
    count += 1;
    loop_.iterate(std::time::Duration::from_millis(100));
}
```

`check_device_enumeration` treats a disconnected channel as "thread finished
unexpectedly" and clears the receiver (`src/audio.rs:354-357`), which is the
only recovery — the panic itself is never surfaced.

## Impact

- A dead-hung enumeration thread burns a little CPU for the process lifetime
  and is never reaped.
- A panic silently drops to placeholder devices (`get_default_devices`),
  which is also what P2 interacts with.

## Options

1. Bound the wait: break after N iterations / a total deadline, log a warning,
   and return whatever was collected (or fall back to defaults).
2. Replace `.expect(...)` with a logged error and an early return so the
   thread exits cleanly instead of panicking.
3. Optionally store the `JoinHandle` alongside the receiver so it can be joined
   on shutdown (or at least logged if it is still alive).

## Files

- `src/audio_pipewire.rs` (`start_device_enumeration`,
  `enumerate_devices_blocking`)
- `src/audio.rs` (`get_devices`, `check_device_enumeration`)
