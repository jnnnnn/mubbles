# P6 — Blocking Ollama HTTP on the UI thread, no timeout

**Status:** open · **Verified:** 7cd9e95

**Severity:** Medium — UI freeze when Ollama is unreachable/hung.
**Type:** Network I/O on the egui thread.

## Symptom

Opening/altering Ollama settings can freeze the window indefinitely if the
Ollama server is slow, hung, or a black hole. `reqwest::Client::new()` has no
default request timeout.

## Evidence

Both helpers build a default client and call `.send()` synchronously:

```rust
// src/summary.rs:577-597
pub fn fetch_ollama_models(api_url: &str) -> Vec<String> {
    ...
    let Ok(resp) = Client::new().get(&tags_url).send() else { ... return vec![]; };
    let Ok(json) = resp.json::<serde_json::Value>() else { ... return vec![]; };
    ...
}

// src/summary.rs:600-608
pub fn fetch_ollama_model_ctx(api_url: &str, model_name: &str) -> Option<usize> {
    ...
    let resp = Client::new().post(&show_url).json(&body).send().ok()?;
    ...
}
```

They are called directly from `render_text_editor` (UI thread):

- `src/app.rs:1009-1018` — on provider/refresh button.
- `src/app.rs:1048-1049` — model list refresh.
- `src/app.rs:1060-1065` — when the selected model changes.

## Impact

- The egui frame (and therefore the whole window) blocks for the duration of
  the TCP/HTTP exchange; with a black-holed host this can be minutes.
- Errors are only logged at `warn`, with no visible feedback.

## Options

1. Build the client with a short timeout
   (`Client::builder().timeout(Duration::from_secs(3)).build()`), which bounds
   the freeze.
2. Move the fetch to a background thread and deliver the result over a channel,
   like the summary request path. This is the correct fix and matches the
   existing `SummaryUpdate` pattern.
3. Repaint immediately after a successful fetch so the new model list shows
   without waiting for the next scheduled repaint.

Recommended: 2 (with 1 as the immediate mitigation if 2 is deferred).

## Files

- `src/summary.rs` (`fetch_ollama_models`, `fetch_ollama_model_ctx`)
- `src/app.rs` (`render_text_editor` Ollama block)
