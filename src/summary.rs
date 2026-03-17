use std::{
    collections::{HashMap, HashSet},
    thread,
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
    output_words: usize,
    input_lines: usize,
    #[serde(skip)]
    tx: std::sync::mpsc::Sender<SummaryUpdate>,
    #[serde(skip)]
    rx: std::sync::mpsc::Receiver<SummaryUpdate>,
}

pub enum SummaryUpdate {
    Additional(String),
    Status(String),
}

const DEFAULT_USER_PROMPT: &str = r#"Summary so far:
%SOFAR%

Additional raw meeting transcript:
%ADDITIONAL%"#;

const DEFAULT_SYSTEM_PROMPT: &str = r#"You are a skilled meeting secretary. Write concise, well-structured minutes for the additional transcript, continuing from the summary so far. Focus on:
- Key decisions and action items
- Important discussion points
- Who said what (when speakers are identified)
- Concrete outcomes and next steps
Use bullet points. Be concise but don't omit important details."#;

impl Default for SummaryState {
    fn default() -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<SummaryUpdate>();
        let provider = ApiProvider::OpenAI;
        Self {
            offset: 0,
            text: String::new(),
            user_prompt: DEFAULT_USER_PROMPT.to_string(),
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            provider: provider.clone(),
            api_url: provider.default_url().to_string(),
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            model: provider.default_model().to_string(),
            ai_input_chars: 8000,
            output_words: 5,
            input_lines: 7,
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
        if ui.button(label).clicked() {
            trigger_summarization_request(summary, text);
        }

        if ui.button("Clear summary").clicked() {
            summary.offset = 0;
            summary.text = String::new();
        }
    });

    poll_ai_updates(summary, text);
}

/// Poll for AI summary responses from the background thread.
/// Call this from any UI path that needs to drive AI summary updates.
/// Returns the latest status message, if any.
pub fn poll_ai_updates(summary: &mut SummaryState, text: &mut String) -> Option<String> {
    let mut status = None;
    while let Ok(update) = summary.rx.try_recv() {
        match update {
            SummaryUpdate::Additional(additional) => {
                summary.text.push_str(format!("\n{}", additional).as_str());
                trigger_summarization_request(summary, text);
            }
            SummaryUpdate::Status(s) => {
                status = Some(s);
            }
        }
    }
    status
}

fn trigger_summarization_request(summary: &mut SummaryState, raw: &str) {
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
        return;
    }
    tracing::info!(
        "requesting {} summary. Offset: {}, chars: {}",
        summary.provider,
        summary.offset,
        additional.len()
    );
    summary.offset += additional.len();

    let _ = summary.tx.send(SummaryUpdate::Status(format!(
        "Requesting {} summary ({} chars)...",
        summary.provider,
        additional.len()
    )));

    let sofar = summary
        .text
        .lines()
        .rev()
        .take(10)
        .collect::<Vec<_>>()
        .join("\n");

    let user_prompt = summary
        .user_prompt
        .replace("%SOFAR%", sofar.as_str())
        .replace("%ADDITIONAL%", additional.as_str());

    let sender = summary.tx.clone();
    let system_prompt = summary.system_prompt.to_owned();
    let api_url = summary.api_url.clone();
    let api_key = summary.api_key.clone();
    let model = summary.model.clone();
    let needs_key = summary.provider.needs_key();

    thread::spawn(move || {
        chat_completion_request(user_prompt, system_prompt, api_url, api_key, model, needs_key, sender);
    });
}

/// Synchronously send a chat completion request to any OpenAI-compatible API
/// (OpenAI, Ollama, or any custom endpoint) and send the result back via the channel.
fn chat_completion_request(
    user_prompt: String,
    system_prompt: String,
    api_url: String,
    api_key: String,
    model: String,
    needs_key: bool,
    tx: std::sync::mpsc::Sender<SummaryUpdate>,
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

    let client = Client::new();
    let body = json!({
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_prompt }
        ],
        "model": model,
        "temperature": 0.7,
        "max_tokens": 1024,
    });

    let mut request = client
        .post(&api_url)
        .header("Content-Type", "application/json");
    if !api_key.is_empty() {
        request = request.header("Authorization", format!("Bearer {}", api_key));
    }

    let response = match request.json(&body).send() {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("Failed to send request to {}: {}", api_url, e);
            let _ = tx.send(SummaryUpdate::Status(format!("AI summary error: {}", e)));
            return;
        }
    };

    let response_json: serde_json::Value = match response.json() {
        Ok(j) => j,
        Err(e) => {
            tracing::error!("Failed to parse response: {}", e);
            let _ = tx.send(SummaryUpdate::Status(format!("AI summary parse error: {}", e)));
            return;
        }
    };
    tracing::info!("response: {:?}", response_json);

    let raw_summary = response_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_else(|| {
            tracing::error!("Unexpected response format: {:?}", response_json);
            ""
        });
    let summary = strip_xml_blocks(raw_summary);
    let _ = tx.send(SummaryUpdate::Status("AI summary received".to_string()));
    if let Err(e) = tx.send(SummaryUpdate::Additional(summary)) {
        tracing::error!("Failed to send summary to main thread: {}", e);
    }
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
        let mut set = HashSet::new();
        let words = [
            "um", "uh", "ah", "like", "so", "yeah", "anyway", "right", "okay",
        ];
        for word in words.iter() {
            set.insert(*word);
        }
        set
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
}
