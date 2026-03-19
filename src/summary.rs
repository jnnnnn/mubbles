use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

use reqwest::blocking::Client;
use serde_json::json;

#[derive(serde::Deserialize, serde::Serialize, PartialEq, Clone, Debug)]
pub enum ApiProvider {
    OpenAI,
    Ollama,
    Custom,
}

impl std::fmt::Display for ApiProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiProvider::OpenAI => write!(f, "OpenAI"),
            ApiProvider::Ollama => write!(f, "Ollama"),
            ApiProvider::Custom => write!(f, "Custom"),
        }
    }
}

impl ApiProvider {
    pub fn default_url(&self) -> &str {
        match self {
            ApiProvider::OpenAI => "https://api.openai.com/v1/chat/completions",
            ApiProvider::Ollama => "http://localhost:11434/v1/chat/completions",
            ApiProvider::Custom => "",
        }
    }

    pub fn default_model(&self) -> &str {
        match self {
            ApiProvider::OpenAI => "gpt-4o-mini",
            ApiProvider::Ollama => "llama3.2",
            ApiProvider::Custom => "",
        }
    }

    pub fn needs_key(&self) -> bool {
        !matches!(self, ApiProvider::Ollama)
    }
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct SummaryState {
    offset: usize, // everything before this character has already been summarized
    pub text: String,
    pub user_prompt: String,
    pub system_prompt: String,
    pub provider: ApiProvider,
    pub api_url: String,
    pub api_key: String,
    pub model: String,
    pub ai_input_chars: usize,
    pub summary_context_lines: usize,
    pub max_tokens: usize,
    pub thinking_budget: usize,
    output_words: usize,
    input_lines: usize,
    #[serde(skip)]
    pub ollama_models: Vec<String>,
    #[serde(skip)]
    pub ollama_model_ctx: Option<usize>,
    #[serde(skip)]
    pub in_progress: Arc<AtomicBool>,
    #[serde(skip)]
    aborted: Arc<AtomicBool>,
    #[serde(skip)]
    pub status: String,
    #[serde(skip)]
    streaming_start: usize,
    #[serde(skip)]
    streaming_raw: String,
    /// Thinking/reasoning tokens streamed separately from content.
    #[serde(skip)]
    pub streaming_thinking: String,
    /// Whether the thinking section is expanded in the UI.
    #[serde(skip)]
    pub thinking_expanded: bool,
    #[serde(skip)]
    tx: std::sync::mpsc::Sender<SummaryUpdate>,
    #[serde(skip)]
    rx: std::sync::mpsc::Receiver<SummaryUpdate>,
}

pub enum SummaryUpdate {
    StreamContent(String),
    StreamThinking(String),
    StreamDone,
    Status(String),
}

const DEFAULT_USER_PROMPT: &str = r#"Summary so far:
%SOFAR%

Additional raw meeting transcript:
%ADDITIONAL%"#;

const DEFAULT_SYSTEM_PROMPT: &str = r#"You are a skilled meeting secretary. You will receive recent summary context and a new chunk of raw transcript. Write ONLY new bullet points for the additional transcript. Do not repeat or rephrase the summary so far.
Focus on:
- Key decisions and action items
- Important discussion points
- Who said what (when speakers are identified)
- Concrete outcomes and next steps
Use bullet points. Be concise but don't omit important details. Output only the new bullet points, nothing else."#;

impl Default for SummaryState {
    fn default() -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<SummaryUpdate>();
        let provider = ApiProvider::OpenAI;
        let api_url = provider.default_url().to_string();
        let model = provider.default_model().to_string();
        Self {
            offset: 0,
            text: String::new(),
            user_prompt: DEFAULT_USER_PROMPT.to_string(),
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            provider,
            api_url,
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            model,
            ai_input_chars: 8000,
            summary_context_lines: 5,
            max_tokens: 4096,
            thinking_budget: 2048,
            output_words: 5,
            input_lines: 7,
            ollama_models: Vec::new(),
            ollama_model_ctx: None,
            in_progress: Arc::new(AtomicBool::new(false)),
            aborted: Arc::new(AtomicBool::new(false)),
            status: String::new(),
            streaming_start: 0,
            streaming_raw: String::new(),
            streaming_thinking: String::new(),
            thinking_expanded: false,
            tx,
            rx,
        }
    }
}

pub fn statistical_ui(summary: &mut SummaryState, ui: &mut egui::Ui, text: &mut String) {
    let changed = ui
        .add(
            egui::Slider::new(&mut summary.input_lines, 1..=20)
                .text("Input lines per summary line"),
        )
        .changed()
        || ui
            .add(
                egui::Slider::new(&mut summary.output_words, 1..=10)
                    .text("Output words for summary line"),
            )
            .changed();
    if changed {
        summarize(text, summary);
    }
}

pub fn ai_ui(summary: &mut SummaryState, ui: &mut egui::Ui, text: &mut String) {
    ui.horizontal(|ui| {
        let label = format!("Request {} summary", summary.provider);
        let busy = summary.in_progress.load(Ordering::Relaxed);
        ui.add_enabled_ui(!busy, |ui| {
            if ui.button(label).clicked() {
                summary.aborted.store(false, Ordering::Relaxed);
                trigger_summarization(summary, text);
            }
        });

        if busy {
            if ui.button("❌ Abort").clicked() {
                summary.aborted.store(true, Ordering::Relaxed);
                summary.in_progress.store(false, Ordering::Relaxed);
                summary.status = "Aborted".to_string();
            }
        }

        if ui.button("Clear summary").clicked() {
            summary.offset = 0;
            summary.text = String::new();
            summary.streaming_thinking.clear();
        }
    });

    if summary.in_progress.load(Ordering::Relaxed) {
        let total = text.len();
        let done = summary.offset;
        let fraction = if total > 0 {
            done as f32 / total as f32
        } else {
            0.0
        };
        ui.add(egui::ProgressBar::new(fraction).text(format!(
            "{} / {} chars ({}%)",
            done,
            total,
            (fraction * 100.0) as u32,
        )));
    }

    if !summary.status.is_empty() {
        ui.label(&summary.status);
    }

    // Show thinking in a collapsible, dimmed section
    if !summary.streaming_thinking.is_empty() {
        let thinking_chars = summary.streaming_thinking.len();
        let budget = summary.thinking_budget;
        let header = if thinking_chars >= budget {
            format!("💭 Thinking ({} chars — budget exhausted)", thinking_chars)
        } else {
            format!("💭 Thinking ({} / {} chars)", thinking_chars, budget)
        };
        egui::CollapsingHeader::new(
            egui::RichText::new(header)
                .italics()
                .color(ui.visuals().weak_text_color()),
        )
        .id_salt("thinking_section")
        .default_open(summary.thinking_expanded)
        .show(ui, |ui| {
            summary.thinking_expanded = true;
            egui::ScrollArea::vertical()
                .id_salt("thinking_scroll")
                .max_height(150.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new(&summary.streaming_thinking)
                            .italics()
                            .weak()
                            .size(11.0),
                    );
                });
        });
    }

    poll_ai_updates(summary, text);
}

/// Poll for AI summary responses from the background thread.
/// Call this from any UI path that needs to drive AI summary updates.
/// Returns the latest status message, if any.
pub fn poll_ai_updates(summary: &mut SummaryState, text: &mut String) -> Option<String> {
    let mut status = None;
    while let Ok(update) = summary.rx.try_recv() {
        match update {
            SummaryUpdate::StreamContent(chunk) => {
                if summary.aborted.load(Ordering::Relaxed) {
                    continue;
                }
                summary.text.push_str(&chunk);
                summary.streaming_raw.push_str(&chunk);
            }
            SummaryUpdate::StreamThinking(chunk) => {
                if summary.aborted.load(Ordering::Relaxed) {
                    continue;
                }
            }
            SummaryUpdate::StreamDone => {
                tracing::debug!(
                    "StreamDone — raw_len={}, thinking_len={}, start={}, text_len={}, aborted={}",
                    summary.streaming_raw.len(),
                    summary.streaming_thinking.len(),
                    summary.streaming_start,
                    summary.text.len(),
                    summary.aborted.load(Ordering::Relaxed),
                );
                if summary.aborted.load(Ordering::Relaxed) {
                    summary.streaming_raw.clear();
                    summary.streaming_thinking.clear();
                    summary.in_progress.store(false, Ordering::Relaxed);
                    continue;
                }
                if summary.streaming_raw.is_empty() && summary.streaming_thinking.is_empty() {
                    tracing::warn!("StreamDone with no content — stopping");
                    summary.in_progress.store(false, Ordering::Relaxed);
                    continue;
                }
                // Replace raw streamed text with XML-stripped version
                // (catches any <think> blocks models sneak into content)
                summary.text.truncate(summary.streaming_start);
                let filtered = strip_xml_blocks(&summary.streaming_raw);
                tracing::debug!(
                    "Filtered stream: {} raw → {} filtered chars, {} thinking chars",
                    summary.streaming_raw.len(),
                    filtered.len(),
                    summary.streaming_thinking.len(),
                );
                summary.text.push_str(&format!("\n{}", filtered));
                summary.streaming_raw.clear();
                // Keep streaming_thinking visible until next request clears it
                // Continue with next chunk (or stop if done)
                trigger_summarization(summary, text);
            }
            SummaryUpdate::Status(s) => {
                if !summary.aborted.load(Ordering::Relaxed) {
                    tracing::debug!("Summary status: {}", s);
                    summary.status = s.clone();
                    status = Some(s);
                }
            }
        }
    }
    status
}

/// Carve off the next chunk of transcript and spawn a background thread
/// to stream a chat-completion request.
fn trigger_summarization(summary: &mut SummaryState, raw: &str) {
    let additional: String = raw
        .chars()
        .skip(summary.offset)
        .take(summary.ai_input_chars)
        .collect();

    if additional.len() < 100 {
        let remaining = raw.len().saturating_sub(summary.offset);
        summary.in_progress.store(false, Ordering::Relaxed);
        let _ = summary.tx.send(SummaryUpdate::Status(format!(
            "Done — {} summary lines, {} remaining chars too short to summarize",
            summary.text.lines().count(),
            remaining,
        )));
        return;
    }

    tracing::info!(
        "requesting {} summary. Offset: {}, chars: {}",
        summary.provider,
        summary.offset,
        additional.len()
    );
    summary.offset += additional.len();
    summary.in_progress.store(true, Ordering::Relaxed);
    summary.streaming_start = summary.text.len();
    summary.streaming_raw.clear();
    summary.streaming_thinking.clear();
    summary.thinking_expanded = false;

    let remaining = raw.len().saturating_sub(summary.offset);
    let _ = summary.tx.send(SummaryUpdate::Status(format!(
        "Requesting {} summary — {} chars, {} remaining, {} lines so far",
        summary.provider,
        additional.len(),
        remaining,
        summary.text.lines().count(),
    )));

    let sofar: String = summary
        .text
        .lines()
        .rev()
        .take(summary.summary_context_lines)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n");

    let user_prompt = summary
        .user_prompt
        .replace("%SOFAR%", &sofar)
        .replace("%ADDITIONAL%", &additional);

    tracing::info!(
        "prompt: {} context lines, {} prompt chars",
        sofar.lines().count(),
        user_prompt.len(),
    );

    let tx = summary.tx.clone();
    let system_prompt = summary.system_prompt.clone();
    let api_url = summary.api_url.clone();
    let api_key = summary.api_key.clone();
    let model = summary.model.clone();
    let needs_key = summary.provider.needs_key();
    let is_ollama = summary.provider == ApiProvider::Ollama;
    let max_tokens = summary.max_tokens;
    let thinking_budget = summary.thinking_budget;

    thread::spawn(move || {
        chat_completion_stream(
            &user_prompt,
            &system_prompt,
            &api_url,
            &api_key,
            &model,
            needs_key,
            is_ollama,
            max_tokens,
            thinking_budget,
            &tx,
        );
    });
}

fn chat_completion_stream(
    user_prompt: &str,
    system_prompt: &str,
    api_url: &str,
    api_key: &str,
    model: &str,
    needs_key: bool,
    is_ollama: bool,
    max_tokens: usize,
    thinking_budget: usize,
    tx: &std::sync::mpsc::Sender<SummaryUpdate>,
) {
    // --- validation ----------------------------------------------------------
    if api_url.is_empty() {
        tracing::error!("API URL is not configured");
        let _ = tx.send(SummaryUpdate::Status(
            "AI summary error: API URL not configured".into(),
        ));
        let _ = tx.send(SummaryUpdate::StreamDone);
        return;
    }
    if needs_key && api_key.is_empty() {
        tracing::error!("API key is not configured");
        let _ = tx.send(SummaryUpdate::Status(
            "AI summary error: API key not configured".into(),
        ));
        let _ = tx.send(SummaryUpdate::StreamDone);
        return;
    }
    if model.is_empty() {
        tracing::error!("Model is not configured");
        let _ = tx.send(SummaryUpdate::Status(
            "AI summary error: model not configured".into(),
        ));
        let _ = tx.send(SummaryUpdate::StreamDone);
        return;
    }

    // --- request -------------------------------------------------------------
    let client = Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .unwrap_or_else(|_| Client::new());

    let mut body = json!({
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user",   "content": user_prompt }
        ],
        "model": model,
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "stream": true,
    });

    if is_ollama {
        // Ollama's OpenAI-compatible API supports `reasoning_effort`
        // with values "high", "medium", "low", "none".
        // When thinking_budget is 0 we disable reasoning entirely so
        // the model produces only content tokens, not just thinking.
        if thinking_budget == 0 {
            body["reasoning_effort"] = json!("none");
        } else {
            body["reasoning_effort"] = json!("low");
        }
    } else {
        // For OpenAI-compatible providers that support thinking budgets.
        if thinking_budget > 0 {
            body["thinking"] = json!({ "budget_tokens": thinking_budget });
        }
    }

    let mut req = client
        .post(api_url)
        .header("Content-Type", "application/json");
    if !api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {}", api_key));
    }

    let _ = tx.send(SummaryUpdate::Status(format!(
        "Streaming from {} — {} prompt chars, max_tokens {}",
        model,
        user_prompt.len(),
        max_tokens
    )));

    let response = match req.json(&body).send() {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("Request to {} failed: {}", api_url, e);
            let _ = tx.send(SummaryUpdate::Status(format!("AI summary error: {}", e)));
            let _ = tx.send(SummaryUpdate::StreamDone);
            return;
        }
    };

    let status = response.status();
    tracing::info!("SSE response status: {}", status);
    if !status.is_success() {
        use std::io::Read;
        let mut body = String::new();
        let _ = response.take(2000).read_to_string(&mut body);
        tracing::error!("API error {}: {}", status, body);
        let _ = tx.send(SummaryUpdate::Status(format!(
            "AI error {}: {}",
            status, body
        )));
        let _ = tx.send(SummaryUpdate::StreamDone);
        return;
    }

    // --- stream SSE ----------------------------------------------------------
    use std::io::{BufRead, BufReader};
    let reader = BufReader::new(response);
    let mut total_chars = 0usize;
    let mut line_count = 0usize;

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("SSE read error at line {}: {}", line_count, e);
                let _ = tx.send(SummaryUpdate::Status(format!("AI stream error: {}", e)));
                break;
            }
        };
        line_count += 1;

        if line.is_empty() {
            continue;
        }

        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            tracing::info!("SSE [DONE] at line {}", line_count);
            break;
        }

        let chunk: serde_json::Value = match serde_json::from_str(data) {
            Ok(j) => j,
            Err(e) => {
                tracing::warn!("SSE JSON parse error at line {}: {}", line_count, e);
                continue;
            }
        };

        // Extract text from whichever field the model uses.
        // Normal models → delta.content
        // Reasoning models (qwen3, deepseek-r1, …) → delta.reasoning / reasoning_content
        let delta = &chunk["choices"][0]["delta"];
        let content = delta["content"].as_str().unwrap_or("");
        let reasoning = delta["reasoning"]
            .as_str()
            .or_else(|| delta["reasoning_content"].as_str())
            .unwrap_or("");

        // Route reasoning and content to separate update channels so the
        // UI can display them independently.
        if !reasoning.is_empty() {
            total_chars += reasoning.len();
            let _ = tx.send(SummaryUpdate::StreamThinking(reasoning.to_string()));
        }

        if !content.is_empty() {
            // Content may still contain inline <think>…</think> blocks
            // from models that embed reasoning in the content field.
            // We detect an opening <think> and route everything until
            // the closing tag to StreamThinking instead.
            let _ = parse_inline_thinking(content, tx, &mut total_chars);
        }

        if content.is_empty() && reasoning.is_empty() {
            continue;
        }
    }

    tracing::info!(
        "SSE stream finished — {} lines, {} chars",
        line_count,
        total_chars
    );
    let _ = tx.send(SummaryUpdate::Status(format!(
        "AI stream complete — {} chars received",
        total_chars
    )));
    if let Err(e) = tx.send(SummaryUpdate::StreamDone) {
        tracing::error!("Failed to send StreamDone: {}", e);
    }
}

// ---------------------------------------------------------------------------
// Ollama helpers
// ---------------------------------------------------------------------------

/// State machine for splitting inline `<think>…</think>` from content.
/// Some models embed reasoning directly in the content field rather than using
/// a separate `reasoning` SSE field.  This function parses each chunk and sends
/// the appropriate `StreamContent` / `StreamThinking` updates.
///
/// We track whether we're inside a `<think>` block with a simple substring scan.
/// Because chunks may split tags across boundaries, we buffer partial tag matches.
fn parse_inline_thinking(
    content: &str,
    tx: &std::sync::mpsc::Sender<SummaryUpdate>,
    total_chars: &mut usize,
) -> Result<(), std::sync::mpsc::SendError<SummaryUpdate>> {
    // Fast path: no XML-ish content at all
    if !content.contains('<') && !content.contains('>') {
        *total_chars += content.len();
        tx.send(SummaryUpdate::StreamContent(content.to_string()))?;
        return Ok(());
    }

    let mut rest = content;
    while !rest.is_empty() {
        if let Some(open) = rest.find("<think>") {
            // Everything before the tag is content
            if open > 0 {
                let before = &rest[..open];
                *total_chars += before.len();
                tx.send(SummaryUpdate::StreamContent(before.to_string()))?;
            }
            let after_open = &rest[open + "<think>".len()..];
            if let Some(close) = after_open.find("</think>") {
                // Full think block in this chunk
                let thinking = &after_open[..close];
                *total_chars += thinking.len();
                tx.send(SummaryUpdate::StreamThinking(thinking.to_string()))?;
                rest = &after_open[close + "</think>".len()..];
            } else {
                // Unclosed <think> — treat the remainder as thinking
                let thinking = after_open;
                *total_chars += thinking.len();
                tx.send(SummaryUpdate::StreamThinking(thinking.to_string()))?;
                rest = "";
            }
        } else if let Some(close) = rest.find("</think>") {
            // We're seeing a close tag without an open — likely a continuation
            // from a previous chunk. Everything before it is thinking.
            if close > 0 {
                let thinking = &rest[..close];
                *total_chars += thinking.len();
                tx.send(SummaryUpdate::StreamThinking(thinking.to_string()))?;
            }
            rest = &rest[close + "</think>".len()..];
        } else {
            // No think tags — plain content
            *total_chars += rest.len();
            tx.send(SummaryUpdate::StreamContent(rest.to_string()))?;
            rest = "";
        }
    }
    Ok(())
}

fn ollama_base_url(api_url: &str) -> &str {
    api_url
        .trim_end_matches('/')
        .trim_end_matches("/v1/chat/completions")
        .trim_end_matches('/')
}

pub fn fetch_ollama_models(api_url: &str) -> Vec<String> {
    let base = ollama_base_url(api_url);
    let tags_url = format!("{}/api/tags", base);
    tracing::info!("Fetching Ollama models from {}", tags_url);
    let Ok(resp) = Client::new().get(&tags_url).send() else {
        tracing::warn!("Failed to reach Ollama at {}", tags_url);
        return vec![];
    };
    let Ok(json) = resp.json::<serde_json::Value>() else {
        tracing::warn!("Failed to parse Ollama model list");
        return vec![];
    };
    json["models"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m["name"].as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

/// Fetch the context length for a specific Ollama model via `/api/show`.
pub fn fetch_ollama_model_ctx(api_url: &str, model_name: &str) -> Option<usize> {
    let show_url = format!("{}/api/show", ollama_base_url(api_url));
    let body = json!({ "name": model_name });
    let resp = Client::new().post(&show_url).json(&body).send().ok()?;
    let json: serde_json::Value = resp.json().ok()?;
    json["model_info"]["general.context_length"]
        .as_u64()
        .map(|v| v as usize)
}

// ---------------------------------------------------------------------------
// XML / think-block stripping
// ---------------------------------------------------------------------------

/// Strip content inside XML-style tag blocks (e.g. `<think>...</think>`) from text.
fn strip_xml_blocks(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(open_start) = rest.find('<') {
        let after_open = &rest[open_start + 1..];
        let tag_end = after_open
            .find(|c: char| c == '>' || c.is_whitespace())
            .unwrap_or(after_open.len());
        let tag_name = &after_open[..tag_end];
        if tag_name.is_empty() || tag_name.starts_with('/') {
            result.push_str(&rest[..open_start + 1]);
            rest = &rest[open_start + 1..];
            continue;
        }
        let closing = format!("</{}>", tag_name);
        if let Some(close_pos) = rest[open_start..].find(&closing) {
            result.push_str(&rest[..open_start]);
            rest = &rest[open_start + close_pos + closing.len()..];
        } else {
            result.push_str(&rest[..open_start + 1]);
            rest = &rest[open_start + 1..];
        }
    }
    result.push_str(rest);
    result.trim().to_string()
}

// ---------------------------------------------------------------------------
// Statistical (offline) summary
// ---------------------------------------------------------------------------

struct WordInSummary {
    importance: f64,
    first_seen_index: usize,
}

fn statistical_summary(
    state: &mut SummaryState,
    raw: &str,
    lines_to_consume: usize,
    words_to_produce: usize,
) {
    let mut linecount = 0;
    let additional: String = raw
        .chars()
        .skip(state.offset)
        .take_while(|c| {
            if *c == '\n' {
                linecount += 1;
            }
            linecount < lines_to_consume
        })
        .collect();
    state.offset += additional.len();

    let ignored = get_ignore_words();
    let mut word_counts = HashMap::new();
    let wordchar = |c: char| !c.is_alphabetic() && c != '\'' && c != '-';
    for (index, word) in additional.split(wordchar).enumerate() {
        if word.trim().len() <= 3 || word.contains('\'') {
            continue;
        }
        let word: String = word
            .chars()
            .filter(|c| c.is_alphabetic())
            .collect::<String>()
            .to_lowercase();
        if ignored.contains(word.as_str()) {
            continue;
        }
        let count = word_counts.entry(word).or_insert(WordInSummary {
            importance: 0f64,
            first_seen_index: index,
        });
        count.importance += 1f64;
    }

    let word_freq_table = get_word_freq_table();
    for (word, count) in word_counts.iter_mut() {
        let word_frequency = word_freq_table.get(word).unwrap_or(&1f64);
        count.importance /= word_frequency;
    }

    let mut sorted: Vec<_> = word_counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.importance.partial_cmp(&a.1.importance).unwrap());
    let mut important_words: Vec<_> = sorted.into_iter().take(words_to_produce).collect();
    important_words.sort_by(|a, b| {
        a.1.first_seen_index
            .partial_cmp(&b.1.first_seen_index)
            .unwrap()
    });

    let words: Vec<_> = important_words
        .iter()
        .map(|(word, _)| word.to_owned())
        .collect();
    let summary = format!("\n- {}", words.join(" "));
    state.text.push_str(&summary);
}

use std::sync::OnceLock;
static WORD_FREQ: OnceLock<HashMap<String, f64>> = OnceLock::new();

fn get_word_freq_table() -> &'static HashMap<String, f64> {
    WORD_FREQ.get_or_init(|| {
        let mut map = HashMap::new();
        let file = include_str!("../assets/count_1w.txt");
        for (index, line) in file.lines().enumerate() {
            let mut parts = line.split('\t');
            let word = parts.next().unwrap();
            let freq = parts.next().unwrap().parse::<usize>().unwrap();
            map.insert(word.to_string(), freq as f64 / index as f64);
        }
        map
    })
}

static IGNORED_WORDS: OnceLock<HashSet<&str>> = OnceLock::new();
fn get_ignore_words() -> &'static HashSet<&'static str> {
    IGNORED_WORDS.get_or_init(|| {
        [
            "um", "uh", "ah", "like", "so", "yeah", "anyway", "right", "okay",
        ]
        .into()
    })
}

pub(crate) fn summarize(raw: &str, summary: &mut SummaryState) {
    summary.offset = 0;
    summary.text = String::new();

    tracing::info!("summarizing from offset {}", summary.offset);
    while summary.offset < raw.len() {
        let prev_offset = summary.offset;
        statistical_summary(summary, raw, summary.input_lines, summary.output_words);
        if summary.offset <= prev_offset {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistical_summary() {
        let mut state = SummaryState::default();
        let raw = "Anyway, so yeah, thank you everybody.
        These sessions are more regular now
        and intended to keep us all updated
        and on the same page in terms of project progress
        from a technology perspective and from product
        and everybody else who's involved.
        We had a bit of a catch up whenever it was,
        I can't remember now, but we spoke to kind of Gaia
        and said it would be good to get the team together
        and really progress through and talk about
        how we're progressing to plan,
        what are some of the risks that are coming through
        so that we need to be aware of as a concern. asdifnkoasnidf";
        statistical_summary(&mut state, raw, 10, 5);
        assert_eq!(state.text, "\n- everybody catch whenever spoke gaia");
        assert_eq!(state.offset, 502);
        statistical_summary(&mut state, raw, 10, 5);
        let last_line = state.text.lines().last().unwrap();
        assert_eq!(
            last_line,
            "- progressing risks aware concern asdifnkoasnidf"
        );
        assert_eq!(state.offset, 668);
        assert_eq!(state.offset, raw.len());
    }

    #[test]
    fn test_strip_xml_blocks() {
        assert_eq!(
            strip_xml_blocks("<think>reasoning here</think>The answer"),
            "The answer"
        );
        assert_eq!(
            strip_xml_blocks("Before <reasoning>internal</reasoning> after"),
            "Before  after"
        );
        assert_eq!(
            strip_xml_blocks("<think>one</think>middle<reflect>two</reflect>end"),
            "middleend"
        );
        assert_eq!(strip_xml_blocks("no tags here"), "no tags here");
        assert_eq!(strip_xml_blocks("a < b and c > d"), "a < b and c > d");
        assert_eq!(
            strip_xml_blocks("<think>\nlong\nblock\n</think>\n- bullet point"),
            "- bullet point"
        );
    }

    #[test]
    fn test_parse_inline_thinking() {
        let (tx, rx) = std::sync::mpsc::channel::<SummaryUpdate>();
        let mut total = 0usize;

        // Plain content — no tags
        parse_inline_thinking("hello world", &tx, &mut total).unwrap();
        assert_eq!(total, 11);
        match rx.try_recv().unwrap() {
            SummaryUpdate::StreamContent(s) => assert_eq!(s, "hello world"),
            other => panic!(
                "expected StreamContent, got {:?}",
                std::mem::discriminant(&other)
            ),
        }

        // Full <think> block in one chunk
        total = 0;
        parse_inline_thinking("<think>reasoning</think>answer", &tx, &mut total).unwrap();
        assert_eq!(total, "reasoning".len() + "answer".len());
        match rx.try_recv().unwrap() {
            SummaryUpdate::StreamThinking(s) => assert_eq!(s, "reasoning"),
            other => panic!(
                "expected StreamThinking, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
        match rx.try_recv().unwrap() {
            SummaryUpdate::StreamContent(s) => assert_eq!(s, "answer"),
            other => panic!(
                "expected StreamContent, got {:?}",
                std::mem::discriminant(&other)
            ),
        }

        // Unclosed <think> — remainder treated as thinking
        total = 0;
        parse_inline_thinking("prefix<think>still thinking", &tx, &mut total).unwrap();
        match rx.try_recv().unwrap() {
            SummaryUpdate::StreamContent(s) => assert_eq!(s, "prefix"),
            other => panic!(
                "expected StreamContent, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
        match rx.try_recv().unwrap() {
            SummaryUpdate::StreamThinking(s) => assert_eq!(s, "still thinking"),
            other => panic!(
                "expected StreamThinking, got {:?}",
                std::mem::discriminant(&other)
            ),
        }

        // Close tag without open — continuation from a previous chunk
        total = 0;
        parse_inline_thinking("continued thinking</think>real content", &tx, &mut total).unwrap();
        match rx.try_recv().unwrap() {
            SummaryUpdate::StreamThinking(s) => assert_eq!(s, "continued thinking"),
            other => panic!(
                "expected StreamThinking, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
        match rx.try_recv().unwrap() {
            SummaryUpdate::StreamContent(s) => assert_eq!(s, "real content"),
            other => panic!(
                "expected StreamContent, got {:?}",
                std::mem::discriminant(&other)
            ),
        }

        // Nothing left on the channel
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_thinking_budget() {
        let mut state = SummaryState::default();
        state.thinking_budget = 20;

        // Simulate receiving thinking chunks that exceed the budget
        let _ = state
            .tx
            .send(SummaryUpdate::StreamThinking("abcdefghij".to_string())); // 10 chars
        let _ = state
            .tx
            .send(SummaryUpdate::StreamThinking("klmnopqrst".to_string())); // 10 chars
        let _ = state
            .tx
            .send(SummaryUpdate::StreamThinking("uvwxyz".to_string())); // 6 chars — should be dropped

        let mut dummy_text = String::new();
        poll_ai_updates(&mut state, &mut dummy_text);

        // Budget is 20 — first two chunks fill it exactly, third is dropped
        assert_eq!(state.streaming_thinking.len(), 20);
        assert_eq!(state.streaming_thinking, "abcdefghijklmnopqrst");

        // Partial capping: budget 15, first chunk 10, second chunk 10 → 5 kept
        let mut state2 = SummaryState::default();
        state2.thinking_budget = 15;
        let _ = state2
            .tx
            .send(SummaryUpdate::StreamThinking("0123456789".to_string()));
        let _ = state2
            .tx
            .send(SummaryUpdate::StreamThinking("abcdefghij".to_string()));
        poll_ai_updates(&mut state2, &mut dummy_text);
        assert_eq!(state2.streaming_thinking.len(), 15);
        assert_eq!(state2.streaming_thinking, "0123456789abcde");
    }

    /// Helper: create a SummaryState configured for local Ollama.
    fn ollama_test_state() -> SummaryState {
        let mut state = SummaryState::default();
        state.provider = ApiProvider::Ollama;
        state.api_url = ApiProvider::Ollama.default_url().to_string();
        state.api_key = String::new();
        state.max_tokens = 512;

        let models = fetch_ollama_models(&state.api_url);
        if !models.is_empty() {
            state.model = models[0].clone();
        } else {
            state.model = ApiProvider::Ollama.default_model().to_string();
        }
        state
    }

    /// Verify we can reach local Ollama and list models.
    #[test]
    fn test_ollama_reachable() {
        let api_url = ApiProvider::Ollama.default_url();
        let models = fetch_ollama_models(api_url);
        assert!(
            !models.is_empty(),
            "Ollama must be running locally with at least one model. Got none from {}",
            api_url,
        );
        println!("Ollama models: {:?}", models);
    }

    /// End-to-end: feed a transcript through trigger_summarization + poll,
    /// verify streaming works and produces a non-empty summary.
    #[test]
    fn test_ollama_end_to_end_summarization() {
        let mut state = ollama_test_state();
        state.ai_input_chars = 4000;
        state.summary_context_lines = 3;

        let raw_transcript = r#"Alice: Good morning everyone, let's get started with the sprint review.
Bob: Sure. So this week I finished the authentication module. We now support OAuth2 and SAML.
Alice: Great work Bob. Any blockers?
Bob: The only issue is that the SAML integration tests are flaky on CI. I've opened a ticket for DevOps to look into it.
Carol: I can help with that. I had a similar issue last month with the certificate rotation.
Alice: Perfect, Carol please sync with Bob after this meeting.
Dave: On my end, I've been working on the new dashboard. The charting library we chose has some performance issues with large datasets.
Alice: How large are we talking?
Dave: Anything over 10,000 data points causes noticeable lag. I'm looking into virtualization or switching to canvas rendering.
Alice: OK, let's timebox that investigation to two days. If we can't fix it, we'll evaluate alternatives.
Carol: For the backend, I've migrated three more endpoints to the new API versioning scheme. Should be done with all of them by Wednesday.
Alice: Nice. Any risks?
Carol: The payments endpoint is tricky because of backward compatibility. I might need an extra day for that one.
Alice: That's fine. Let's plan for Thursday as the deadline for payments.
Bob: One more thing — the security audit report came back. No critical issues, but there are two medium-severity findings we should address this sprint.
Alice: Can you create tickets for those and we'll prioritize them in the next planning session?
Bob: Already done.
Alice: Excellent. Anything else? OK, let's wrap up. Action items: Carol syncs with Bob on SAML CI, Dave timeboxes dashboard perf to two days, Bob created security audit tickets. Next meeting same time Thursday."#;

        println!("Using model: {}", state.model);

        trigger_summarization(&mut state, raw_transcript);

        assert!(
            state.in_progress.load(Ordering::Relaxed),
            "Expected in_progress after trigger",
        );

        let timeout = Duration::from_secs(120);
        let start = std::time::Instant::now();
        let mut raw_text = raw_transcript.to_string();
        let mut idle_polls = 0;

        loop {
            if start.elapsed() > timeout {
                panic!(
                    "Timed out. Status: '{}', summary so far:\n{}",
                    state.status, state.text,
                );
            }

            if let Some(s) = poll_ai_updates(&mut state, &mut raw_text) {
                println!("[status] {}", s);
            }

            if !state.in_progress.load(Ordering::Relaxed) {
                idle_polls += 1;
                if idle_polls > 5 {
                    break;
                }
            }

            thread::sleep(Duration::from_millis(100));
        }

        println!("Final status: {}", state.status);
        println!("Summary text:\n{}", state.text);

        assert!(
            !state.in_progress.load(Ordering::Relaxed),
            "Summarization should have completed",
        );
        assert!(
            !state.text.trim().is_empty(),
            "Summary text should not be empty. Status: '{}'",
            state.status,
        );

        let bullet_count = state
            .text
            .lines()
            .filter(|l| {
                let t = l.trim();
                t.starts_with('-') || t.starts_with('*') || t.starts_with('•')
            })
            .count();
        println!("Bullet points found: {}", bullet_count);
        assert!(
            bullet_count > 0,
            "Expected bullet points in summary, got:\n{}",
            state.text,
        );

        assert!(
            !state.status.to_lowercase().contains("error"),
            "Status indicates an error: {}",
            state.status,
        );
    }
}
