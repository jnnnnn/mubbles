# P7 — AI summary thread: fire-and-forget, ineffective abort, race

**Status:** open · **Verified:** 7cd9e95

**Severity:** Medium.
**Type:** Unmanaged thread lifetime + abort/concurrency correctness.

## Symptom

- A summary request thread is never joined; on shutdown it can outlive the UI
  (bounded only by a 600 s HTTP timeout).
- "Abort" does not actually cancel an in-flight request that is stuck before
  the first streamed line.
- Clicking request again after Abort can run two summarization threads at once,
  interleaving into the same `streaming_raw` state.

## Evidence

Fire-and-forget spawn with no stored handle:

```rust
// src/summary.rs:374-376
thread::spawn(move || {
    chat_completion_request(user_prompt, system_prompt, api_url, api_key, model,
                            needs_key, max_tokens, is_ollama, thinking_budget, sender, aborted);
});
```

600 s timeout:

```rust
// src/summary.rs:408-411
let client = Client::builder().timeout(Duration::from_secs(600)).build()...;
```

Abort is only observed between streamed lines, and only after the response
headers have arrived:

```rust
// src/summary.rs:482-487
for line in reader.lines() {
    if aborted.load(Ordering::Relaxed) { /* send StreamDone, return */ }
    ...
}
```

`ai_ui` re-enables the request button the instant Abort is pressed, and a new
request shares the same `aborted` flag — clearing it also un-aborts the old
thread:

```rust
// src/summary.rs:184
summary.aborted.store(false, Ordering::Relaxed); // on new request
// src/summary.rs:191-192
summary.aborted.store(true, Ordering::Relaxed);  // on abort
summary.in_progress.store(false, Ordering::Relaxed);
```

Both threads then send `StreamChunk`/`StreamDone`, and `poll_ai_updates`
(`src/summary.rs:232-293`) mutates the shared `streaming_start` /
`streaming_raw`, producing duplicated or overwritten text.

## Options

1. **Store the handle** on `SummaryState` (`Option<JoinHandle<()>>`) and join /
   cancel on drop or on a new request. Even a best-effort join at shutdown is
   better than none.
2. **Per-request abort token.** Give each request its own
   `Arc<AtomicBool>` (not a shared one) and hand it to the thread; a new request
   creates a fresh token, so it can't un-abort the previous one. Keep the
   shared flag only if a global "stop everything" is wanted.
3. **Real cancellation.** Use a `Client` whose connection can be dropped, or
   check `aborted` with a read timeout on the response body so Abort works
   before the first line. A shorter connect/read timeout helps too.
4. **Guard concurrency.** Don't clear `in_progress` on Abort until the old
   thread has actually stopped (e.g. set a "stopping" state), or key updates by
   a request id and ignore stale ones in `poll_ai_updates`.

Recommended: 2 + 3, then 1.

## Files

- `src/summary.rs` (`trigger_summarization_request`, `chat_completion_request`,
  `ai_ui`, `poll_ai_updates`, `SummaryState`)
- `src/app.rs` (shutdown path, if a join is added)
