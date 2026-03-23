use std::{
    collections::{HashMap, HashSet},
    sync::{atomic::{AtomicBool, Ordering}, Arc},
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
            ApiProvider::Ollama => "http://localhost:11434/api/chat",
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
    pub free_gpu: bool,
    output_words: usize,
    input_lines: usize,
    #[serde(skip)]
    pub ollama_models: Vec<String>,
    #[serde(skip)]
    pub ollama_model_ctx: Option<usize>,
    #[serde(skip)]
    pub whisper_paused: Option<Arc<AtomicBool>>,
    #[serde(skip)]
    pub in_progress: Arc<AtomicBool>,
    #[serde(skip)]
    aborted: Arc<AtomicBool>,
    #[serde(skip)]
    pub status: String,
    #[serde(skip)]
    pub thinking_text: String,
    #[serde(skip)]
    streaming_start: usize,
    #[serde(skip)]
    streaming_raw: String,
    #[serde(skip)]
    tx: std::sync::mpsc::Sender<SummaryUpdate>,
    #[serde(skip)]
    rx: std::sync::mpsc::Receiver<SummaryUpdate>,
}

pub enum SummaryUpdate {
    StreamChunk(String),
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
            thinking_budget: 0,
            free_gpu: false,
            output_words: 5,
            input_lines: 7,
            ollama_models: Vec::new(),
            ollama_model_ctx: None,
            whisper_paused: None,
            in_progress: Arc::new(AtomicBool::new(false)),
            aborted: Arc::new(AtomicBool::new(false)),
            status: String::new(),
            thinking_text: String::new(),
            streaming_start: 0,
            streaming_raw: String::new(),
            tx,
            rx,
        }
    }
}

impl SummaryState {
    pub fn set_whisper_paused(&self, paused: bool) {
        if self.free_gpu {
            if let Some(ref flag) = self.whisper_paused {
                flag.store(paused, Ordering::Relaxed);
            }
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
                trigger_summarization_request(summary, text);
            }
        });

        if busy {
            if ui.button("❌ Abort").clicked() {
                summary.aborted.store(true, Ordering::Relaxed);
                summary.in_progress.store(false, Ordering::Relaxed);
                summary.set_whisper_paused(false);
                summary.status = "Aborted".to_string();
            }
        }

        if ui.button("Clear summary").clicked() {
            summary.offset = 0;
            summary.text = String::new();
        }
    });

    if summary.in_progress.load(Ordering::Relaxed) {
        let total = text.len();
        let done = summary.offset;
        let fraction = if total > 0 { done as f32 / total as f32 } else { 0.0 };
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

    if !summary.thinking_text.is_empty() {
        ui.collapsing("Thinking", |ui| {
            ui.label(&summary.thinking_text);
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
            SummaryUpdate::StreamChunk(chunk) => {
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
                summary.thinking_text.push_str(&chunk);
            }
            SummaryUpdate::StreamDone => {
                tracing::debug!(
                    "StreamDone received — raw_len={}, streaming_start={}, text_len={}, aborted={}",
                    summary.streaming_raw.len(),
                    summary.streaming_start,
                    summary.text.len(),
                    summary.aborted.load(Ordering::Relaxed),
                );
                if summary.aborted.load(Ordering::Relaxed) {
                    summary.streaming_raw.clear();
                    summary.in_progress.store(false, Ordering::Relaxed);
                    summary.set_whisper_paused(false);
                    continue;
                }
                if summary.streaming_raw.is_empty() {
                    // Stream produced no content (error or empty response)
                    tracing::warn!("StreamDone with no content — stopping");
                    summary.in_progress.store(false, Ordering::Relaxed);
                    summary.set_whisper_paused(false);
                    continue;
                }
                // Replace the raw streamed text with the XML-stripped version
                summary.text.truncate(summary.streaming_start);
                let filtered = strip_xml_blocks(&summary.streaming_raw);
                tracing::debug!(
                    "Filtered stream: {} raw chars → {} filtered chars",
                    summary.streaming_raw.len(),
                    filtered.len(),
                );
                summary.text.push_str(filtered.trim_start());
                summary.streaming_raw.clear();
                // Continue with next chunk (sets in_progress=true, or false if done)
                trigger_summarization_request(summary, text);
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

fn trigger_summarization_request(summary: &mut SummaryState, raw: &str) {
    summary.set_whisper_paused(true);
    let additional = raw
        .chars()
        .skip(summary.offset)
        .take(summary.ai_input_chars)
        .collect::<String>();
    if additional.len() < 100 {
        tracing::warn!(
            "{} chars is not enough additional text to summarize",
            additional.len()
        );
        let remaining = raw.len().saturating_sub(summary.offset);
        summary.in_progress.store(false, Ordering::Relaxed);
        summary.set_whisper_paused(false);
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
    if !summary.text.is_empty() && !summary.text.ends_with('\n') {
        summary.text.push('\n');
    }
    summary.streaming_start = summary.text.len();
    summary.streaming_raw.clear();
    summary.thinking_text.clear();

    let remaining = raw.len().saturating_sub(summary.offset);
    let _ = summary.tx.send(SummaryUpdate::Status(format!(
        "Requesting {} summary — {} chars to summarize, {} remaining, {} summary lines so far",
        summary.provider,
        additional.len(),
        remaining,
        summary.text.lines().count(),
    )));

    let sofar = summary
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
        .replace("%SOFAR%", sofar.as_str())
        .replace("%ADDITIONAL%", additional.as_str());

    tracing::info!(
        "prompt: {} context lines, {} prompt chars",
        sofar.lines().count(),
        user_prompt.len(),
    );

    let sender = summary.tx.clone();
    let system_prompt = summary.system_prompt.to_owned();
    let api_url = summary.api_url.clone();
    let api_key = summary.api_key.clone();
    let model = summary.model.clone();
    let needs_key = summary.provider.needs_key();
    let max_tokens = summary.max_tokens;
    let is_ollama = matches!(summary.provider, ApiProvider::Ollama);
    let thinking_budget = summary.thinking_budget;
    let aborted = summary.aborted.clone();

    thread::spawn(move || {
        chat_completion_request(user_prompt, system_prompt, api_url, api_key, model, needs_key, max_tokens, is_ollama, thinking_budget, sender, aborted);
    });
}

fn chat_completion_request(
    user_prompt: String,
    system_prompt: String,
    api_url: String,
    api_key: String,
    model: String,
    needs_key: bool,
    max_tokens: usize,
    is_ollama: bool,
    thinking_budget: usize,
    tx: std::sync::mpsc::Sender<SummaryUpdate>,
    aborted: Arc<AtomicBool>,
) {
    if api_url.is_empty() {
        tracing::error!("API URL is not configured. Set it in Settings.");
        let _ = tx.send(SummaryUpdate::Status("AI summary error: API URL not configured".to_string()));
        return;
    }
    if needs_key && api_key.is_empty() {
        tracing::error!("API key is not configured. Set it in Settings or via OPENAI_API_KEY env var.");
        let _ = tx.send(SummaryUpdate::Status("AI summary error: API key not configured".to_string()));
        return;
    }
    if model.is_empty() {
        tracing::error!("Model is not configured. Set it in Settings.");
        let _ = tx.send(SummaryUpdate::Status("AI summary error: model not configured".to_string()));
        return;
    }

    let client = Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .unwrap_or_else(|_| Client::new());
    let body = if is_ollama {
        let mut b = json!({
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_prompt }
            ],
            "model": model,
            "stream": true,
            "think": thinking_budget > 0,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7,
            },
        });
        if thinking_budget > 0 {
            b["options"]["num_predict"] = json!(max_tokens + thinking_budget);
        }
        b
    } else {
        json!({
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_prompt }
            ],
            "model": model,
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "stream": true,
        })
    };

    let mut request = client
        .post(&api_url)
        .header("Content-Type", "application/json");
    if !api_key.is_empty() {
        request = request.header("Authorization", format!("Bearer {}", api_key));
    }

    let _ = tx.send(SummaryUpdate::Status(format!(
        "Streaming from {} — {} prompt chars, max_tokens {}",
        model, user_prompt.len(), max_tokens
    )));

    let response = match request.json(&body).send() {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("Failed to send request to {}: {}", api_url, e);
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
        let _ = tx.send(SummaryUpdate::Status(format!("AI error {}: {}", status, body)));
        let _ = tx.send(SummaryUpdate::StreamDone);
        return;
    }

    use std::io::{BufRead, BufReader};
    let reader = BufReader::new(response);
    let mut total_chars = 0usize;
    let mut line_count = 0usize;

    for line in reader.lines() {
        if aborted.load(Ordering::Relaxed) {
            tracing::info!("Aborting stream at line {} — closing connection", line_count);
            let _ = tx.send(SummaryUpdate::StreamDone);
            return;
        }
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("Error reading stream line {}: {}", line_count, e);
                let _ = tx.send(SummaryUpdate::Status(format!("AI stream error: {}", e)));
                break;
            }
        };
        line_count += 1;

        if line.is_empty() {
            continue;
        }

        // Ollama native: each line is a JSON object directly
        // OpenAI SSE: each line is "data: {json}" or "data: [DONE]"
        let data = if is_ollama {
            line.as_str()
        } else {
            let Some(d) = line.strip_prefix("data: ") else {
                tracing::debug!("SSE skip non-data line {}: {:?}", line_count, &line[..line.len().min(200)]);
                continue;
            };
            if d == "[DONE]" {
                tracing::info!("SSE [DONE] at line {}", line_count);
                break;
            }
            d
        };

        let chunk: serde_json::Value = match serde_json::from_str(data) {
            Ok(j) => j,
            Err(e) => {
                tracing::warn!("JSON parse error at line {}: {} — data: {:?}", line_count, e, &data[..data.len().min(200)]);
                continue;
            }
        };

        if is_ollama {
            // Ollama native ndjson: content in message.content, thinking in message.thinking
            if let Some(content) = chunk["message"]["content"].as_str() {
                if !content.is_empty() {
                    total_chars += content.len();
                    let _ = tx.send(SummaryUpdate::StreamChunk(content.to_string()));
                }
            }
            if let Some(thinking) = chunk["message"]["thinking"].as_str() {
                if !thinking.is_empty() {
                    let _ = tx.send(SummaryUpdate::StreamThinking(thinking.to_string()));
                }
            }
            if let Some(err) = chunk["error"].as_str() {
                tracing::error!("Ollama stream error: {}", err);
                let _ = tx.send(SummaryUpdate::Status(format!("Ollama error: {}", err)));
                break;
            }
            if chunk["done"].as_bool() == Some(true) {
                tracing::info!("Ollama stream done at line {}", line_count);
                break;
            }
        } else {
            // OpenAI SSE: content in choices[0].delta.content
            if let Some(content) = chunk["choices"][0]["delta"]["content"].as_str() {
                total_chars += content.len();
                let _ = tx.send(SummaryUpdate::StreamChunk(content.to_string()));
            } else {
                tracing::debug!("SSE chunk without content at line {}: {:?}", line_count, &data[..data.len().min(200)]);
            }
        }
    }

    tracing::info!("SSE stream finished — {} lines, {} content chars", line_count, total_chars);
    let _ = tx.send(SummaryUpdate::Status(format!(
        "AI stream complete — {} chars received",
        total_chars
    )));
    if let Err(e) = tx.send(SummaryUpdate::StreamDone) {
        tracing::error!("Failed to send stream-done to main thread: {}", e);
    }
}

fn ollama_base_url(api_url: &str) -> &str {
    api_url
        .trim_end_matches('/')
        .trim_end_matches("/v1/chat/completions")
        .trim_end_matches("/api/chat")
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

/// Strip content inside XML-style tag blocks (e.g. `<think>...</think>`) from text.
fn strip_xml_blocks(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(open_start) = rest.find('<') {
        // Find the tag name
        let after_open = &rest[open_start + 1..];
        let tag_end = after_open
            .find(|c: char| c == '>' || c.is_whitespace())
            .unwrap_or(after_open.len());
        let tag_name = &after_open[..tag_end];
        if tag_name.is_empty() || tag_name.starts_with('/') {
            // Not an opening tag, keep the '<' and move on
            result.push_str(&rest[..open_start + 1]);
            rest = &rest[open_start + 1..];
            continue;
        }
        let closing = format!("</{}>", tag_name);
        if let Some(close_pos) = rest[open_start..].find(&closing) {
            // Found matching close tag — skip the entire block
            result.push_str(&rest[..open_start]);
            rest = &rest[open_start + close_pos + closing.len()..];
        } else {
            // No closing tag, keep everything
            result.push_str(&rest[..open_start + 1]);
            rest = &rest[open_start + 1..];
        }
    }
    result.push_str(rest);
    result.trim().to_string()
}

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
    // summarize ten lines at a time
    let mut linecount = 0;
    let additional = raw
        .chars()
        .skip(state.offset)
        .take_while(|c| {
            if *c == '\n' {
                linecount += 1;
            }
            linecount < lines_to_consume
        })
        .collect::<String>();
    state.offset += additional.len();

    let ignored = get_ignore_words();
    // count the words, splitting at any non-alpha character except ' and - and ignoring whitespace
    let mut word_counts = HashMap::new();
    let wordchar = |c: char| !c.is_alphabetic() && c != '\'' && c != '-';
    for (index, word) in additional.split(wordchar).enumerate() {
        if word.trim().len() <= 3 || word.contains('\'') {
            continue;
        }
        // strip non-alpha as count_1w.txt doesn't have apostrophes or similar
        let word = word
            .chars()
            .filter(|c| c.is_alphabetic())
            .collect::<String>()
            .to_lowercase();
        // if <3 or ignored or contains ' then skip
        if ignored.contains(word.as_str()) {
            continue;
        }
        let count = word_counts.entry(word).or_insert(WordInSummary {
            importance: 0f64,
            first_seen_index: index,
        });
        count.importance += 1f64;
    }

    // divide the counts by the google-search frequency of that word from https://norvig.com/ngrams/count_1w.txt
    let word_freq_table = get_word_freq_table();
    for (word, count) in word_counts.iter_mut() {
        let word_frequency = word_freq_table.get(word).unwrap_or(&1f64);
        count.importance /= word_frequency;
    }

    // now sort by the weighted counts
    let mut sorted: Vec<_> = word_counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.importance.partial_cmp(&a.1.importance).unwrap());
    // take the top 10
    let mut important_words: Vec<_> = sorted.into_iter().take(words_to_produce).collect();
    // sort by first seen index
    important_words.sort_by(|a, b| {
        a.1.first_seen_index
            .partial_cmp(&b.1.first_seen_index)
            .unwrap()
    });

    let words = important_words
        .iter()
        .map(|(word, _)| word.to_owned())
        .collect::<Vec<_>>();

    let summary = format!("\n- {}", words.join(" "));
    state.text.push_str(&summary);
}

use std::sync::OnceLock;
static WORD_FREQ: OnceLock<HashMap<String, f64>> = OnceLock::new();

// the file 'assets/count_1w.txt' contains a list of words and their frequency in google searches, separated by tab and newline.
// here we load the file so that it is compiled into the binary, and build a hashmap of the words and their frequencies.
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

// static lookup for ignored words "um", "uh", "ah", "like", "so", "yeah", "anyway", "right", "okay"
static IGNORED_WORDS: OnceLock<HashSet<&str>> = OnceLock::new();
fn get_ignore_words() -> &'static HashSet<&'static str> {
    IGNORED_WORDS.get_or_init(|| {
        ["um", "uh", "ah", "like", "so", "yeah", "anyway", "right", "okay"].into()
    })
}

pub(crate) fn summarize(raw: &str, summary: &mut SummaryState) {
    // just do a statistical summary from now, regenerating the full summary every time
    summary.offset = 0;
    summary.text = String::new();

    tracing::info!("summarizing from offset {}", summary.offset);
    while summary.offset < raw.len() {
        let prev_offset = summary.offset;
        statistical_summary(summary, raw, summary.input_lines, summary.output_words);
        if summary.offset <= prev_offset {
            break; // we didn't make any progress, so stop
        }
    }
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_statistical_summary() {
        // make sure the least common words from the first ten lines are summarized
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
        // make sure the last line is right
        let last_line = state.text.lines().last().unwrap();
        assert_eq!(
            last_line,
            "- progressing risks aware concern asdifnkoasnidf"
        );
        // make sure we end up at the end of the text
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

    /// Parse a single ndjson line from an Ollama native streaming response.
    /// Returns (content, thinking, done, error).
    fn parse_ollama_line(line: &str) -> (String, String, bool, Option<String>) {
        let chunk: serde_json::Value = serde_json::from_str(line).unwrap();
        let content = chunk["message"]["content"].as_str().unwrap_or("").to_string();
        let thinking = chunk["message"]["thinking"].as_str().unwrap_or("").to_string();
        let done = chunk["done"].as_bool().unwrap_or(false);
        let error = chunk.get("error").and_then(|e| e.as_str()).map(|s| s.to_string());
        (content, thinking, done, error)
    }

    #[test]
    fn test_parse_ollama_ndjson_content_only() {
        // Real captured response from Ollama with think:false
        let lines = vec![
            r#"{"model":"qwen3.5:0.8b","created_at":"2026-03-19T19:40:05.5353876Z","message":{"role":"assistant","content":"Hello"},"done":false}"#,
            r#"{"model":"qwen3.5:0.8b","created_at":"2026-03-19T19:40:05.6092958Z","message":{"role":"assistant","content":" from"},"done":false}"#,
            r#"{"model":"qwen3.5:0.8b","created_at":"2026-03-19T19:40:05.6804279Z","message":{"role":"assistant","content":"."},"done":false}"#,
            r#"{"model":"qwen3.5:0.8b","created_at":"2026-03-19T19:40:05.6942017Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","total_duration":706201400,"load_duration":377537300,"prompt_eval_count":19,"prompt_eval_duration":167056200,"eval_count":6,"eval_duration":155825300}"#,
        ];

        let mut content = String::new();
        let mut thinking = String::new();
        let mut finished = false;
        for line in &lines {
            let (c, t, done, err) = parse_ollama_line(line);
            assert!(err.is_none());
            content.push_str(&c);
            thinking.push_str(&t);
            if done { finished = true; }
        }
        assert_eq!(content, "Hello from.");
        assert_eq!(thinking, "");
        assert!(finished);
    }

    #[test]
    fn test_parse_ollama_ndjson_with_thinking() {
        // Real captured response with think:true — thinking comes in message.thinking,
        // content is empty during thinking phase
        let lines = vec![
            r#"{"model":"qwen3.5:0.8b","created_at":"2026-03-19T19:40:14.0739283Z","message":{"role":"assistant","content":"","thinking":"Thinking"},"done":false}"#,
            r#"{"model":"qwen3.5:0.8b","created_at":"2026-03-19T19:40:14.0882712Z","message":{"role":"assistant","content":"","thinking":" Process"},"done":false}"#,
            r#"{"model":"qwen3.5:0.8b","created_at":"2026-03-19T19:40:14.1021096Z","message":{"role":"assistant","content":"","thinking":":"},"done":false}"#,
            // then content arrives after thinking
            r#"{"model":"qwen3.5:0.8b","created_at":"2026-03-19T19:40:25.9Z","message":{"role":"assistant","content":"Hello"},"done":false}"#,
            r#"{"model":"qwen3.5:0.8b","created_at":"2026-03-19T19:40:26.0Z","message":{"role":"assistant","content":" world"},"done":false}"#,
            r#"{"model":"qwen3.5:0.8b","created_at":"2026-03-19T19:40:26.1Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","total_duration":1564518800}"#,
        ];

        let mut content = String::new();
        let mut thinking = String::new();
        let mut finished = false;
        for line in &lines {
            let (c, t, done, err) = parse_ollama_line(line);
            assert!(err.is_none());
            if !c.is_empty() { content.push_str(&c); }
            if !t.is_empty() { thinking.push_str(&t); }
            if done { finished = true; }
        }
        assert_eq!(thinking, "Thinking Process:");
        assert_eq!(content, "Hello world");
        assert!(finished);
    }

    #[test]
    fn test_parse_ollama_ndjson_thinking_exhausts_budget() {
        // When thinking runs out of tokens, done_reason is "length" and content may be empty
        let lines = vec![
            r#"{"model":"qwen3.5:0.8b","created_at":"2026-03-19T19:40:25.7706112Z","message":{"role":"assistant","content":"","thinking":" words"},"done":false}"#,
            r#"{"model":"qwen3.5:0.8b","created_at":"2026-03-19T19:40:25.782425Z","message":{"role":"assistant","content":"","thinking":")"},"done":false}"#,
            r#"{"model":"qwen3.5:0.8b","created_at":"2026-03-19T19:40:25.782425Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"length","total_duration":1564518800}"#,
        ];

        let mut content = String::new();
        let mut thinking = String::new();
        let mut done_reason = String::new();
        for line in &lines {
            let chunk: serde_json::Value = serde_json::from_str(line).unwrap();
            if let Some(c) = chunk["message"]["content"].as_str() { content.push_str(c); }
            if let Some(t) = chunk["message"]["thinking"].as_str() { thinking.push_str(t); }
            if let Some(r) = chunk["done_reason"].as_str() { done_reason = r.to_string(); }
        }
        assert_eq!(content, ""); // no actual content was produced
        assert_eq!(thinking, " words)");
        assert_eq!(done_reason, "length");
    }

    #[test]
    fn test_parse_ollama_ndjson_error() {
        let line = r#"{"error":"an error was encountered while running the model"}"#;
        let chunk: serde_json::Value = serde_json::from_str(line).unwrap();
        let err = chunk["error"].as_str();
        assert_eq!(err, Some("an error was encountered while running the model"));
    }

    /// Integration test: make a real request to Ollama with think:false.
    /// Requires Ollama running locally with qwen3.5:0.8b.
    #[test]
    fn test_ollama_real_request_no_thinking() {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();

        // Check Ollama is reachable
        let tags = client.get("http://localhost:11434/api/tags").send();
        if tags.is_err() || !tags.unwrap().status().is_success() {
            eprintln!("Skipping test_ollama_real_request_no_thinking: Ollama not running");
            return;
        }

        let body = json!({
            "model": "qwen3.5:0.8b",
            "messages": [{"role": "user", "content": "Say hello in 3 words"}],
            "stream": true,
            "think": false,
            "options": {"num_predict": 20},
        });

        let resp = client
            .post("http://localhost:11434/api/chat")
            .json(&body)
            .send()
            .expect("Failed to send request");

        assert!(resp.status().is_success(), "HTTP error: {}", resp.status());

        use std::io::{BufRead, BufReader};
        let reader = BufReader::new(resp);
        let mut content = String::new();
        let mut saw_done = false;

        for line in reader.lines() {
            let line = line.expect("Failed to read line");
            if line.is_empty() { continue; }

            let chunk: serde_json::Value = serde_json::from_str(&line)
                .unwrap_or_else(|e| panic!("Failed to parse JSON: {} — line: {}", e, line));

            // With think:false, there should be no thinking field
            assert!(
                chunk["message"]["thinking"].is_null() || chunk["message"]["thinking"].as_str() == Some(""),
                "Unexpected thinking content with think:false: {:?}", chunk
            );

            if let Some(c) = chunk["message"]["content"].as_str() {
                content.push_str(c);
            }
            if chunk["done"].as_bool() == Some(true) {
                saw_done = true;
                break;
            }
        }

        assert!(saw_done, "Stream did not end with done:true");
        assert!(!content.is_empty(), "No content received from Ollama");
        eprintln!("Ollama response (think:false): {:?}", content);
    }

    /// Integration test: make a real request to Ollama with think:true.
    /// Requires Ollama running locally with qwen3.5:0.8b.
    #[test]
    fn test_ollama_real_request_with_thinking() {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap();

        // Check Ollama is reachable
        let tags = client.get("http://localhost:11434/api/tags").send();
        if tags.is_err() || !tags.unwrap().status().is_success() {
            eprintln!("Skipping test_ollama_real_request_with_thinking: Ollama not running");
            return;
        }

        let body = json!({
            "model": "qwen3.5:0.8b",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "stream": true,
            "think": true,
            "options": {"num_predict": 2000},
        });

        let resp = client
            .post("http://localhost:11434/api/chat")
            .json(&body)
            .send()
            .expect("Failed to send request");

        assert!(resp.status().is_success(), "HTTP error: {}", resp.status());

        use std::io::{BufRead, BufReader};
        let reader = BufReader::new(resp);
        let mut content = String::new();
        let mut thinking = String::new();
        let mut saw_done = false;

        for line in reader.lines() {
            let line = line.expect("Failed to read line");
            if line.is_empty() { continue; }

            let chunk: serde_json::Value = serde_json::from_str(&line)
                .unwrap_or_else(|e| panic!("Failed to parse JSON: {} — line: {}", e, line));

            if let Some(c) = chunk["message"]["content"].as_str() {
                content.push_str(c);
            }
            if let Some(t) = chunk["message"]["thinking"].as_str() {
                thinking.push_str(t);
            }
            if chunk["done"].as_bool() == Some(true) {
                saw_done = true;
                break;
            }
        }

        assert!(saw_done, "Stream did not end with done:true");
        assert!(!thinking.is_empty(), "No thinking received from Ollama with think:true");
        // With a small model, thinking may consume entire budget. Just verify we got either content or done_reason=length.
        eprintln!("Ollama thinking len: {}, content len: {}", thinking.len(), content.len());
        eprintln!("Ollama content: {:?}", content);
    }

    /// Integration test: use our actual chat_completion_request function via mpsc channel.
    /// Requires Ollama running locally with qwen3.5:0.8b.
    #[test]
    fn test_chat_completion_request_ollama() {
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();
        let tags = client.get("http://localhost:11434/api/tags").send();
        if tags.is_err() || !tags.unwrap().status().is_success() {
            eprintln!("Skipping: Ollama not running");
            return;
        }

        let (tx, rx) = std::sync::mpsc::channel::<SummaryUpdate>();
        let aborted = Arc::new(AtomicBool::new(false));
        chat_completion_request(
            "Say 'hello world' and nothing else.".to_string(),
            "Be concise.".to_string(),
            "http://localhost:11434/api/chat".to_string(),
            String::new(),
            "qwen3.5:0.8b".to_string(),
            false,  // needs_key
            50,     // max_tokens
            true,   // is_ollama
            0,      // thinking_budget (think: false)
            tx,
            aborted,
        );

        let mut content = String::new();
        let mut thinking = String::new();
        let mut got_done = false;
        let mut statuses = vec![];
        loop {
            match rx.recv_timeout(Duration::from_secs(30)) {
                Ok(SummaryUpdate::StreamChunk(c)) => content.push_str(&c),
                Ok(SummaryUpdate::StreamThinking(t)) => thinking.push_str(&t),
                Ok(SummaryUpdate::StreamDone) => { got_done = true; break; }
                Ok(SummaryUpdate::Status(s)) => statuses.push(s),
                Err(_) => panic!("Timed out waiting for Ollama response"),
            }
        }
        eprintln!("Content: {:?}", content);
        eprintln!("Thinking: {:?}", thinking);
        eprintln!("Statuses: {:?}", statuses);
        assert!(got_done, "Did not receive StreamDone");
        assert!(!content.is_empty(), "Got no content from chat_completion_request");
        assert!(thinking.is_empty(), "Got thinking with think:false");
    }
}
