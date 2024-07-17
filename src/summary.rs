use std::{
    collections::{HashMap, HashSet},
    thread,
};

use reqwest::blocking::Client;
use serde_json::json;

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct SummaryState {
    offset: usize, // everything before this character has already been summarized
    pub text: String,
    pub user_prompt: String,
    pub system_prompt: String,
    output_words: usize,
    input_lines: usize,
    #[serde(skip)]
    tx: std::sync::mpsc::Sender<SummaryUpdate>,
    #[serde(skip)]
    rx: std::sync::mpsc::Receiver<SummaryUpdate>,
}

pub enum SummaryUpdate {
    Additional(String),
}

const DEFAULT_USER_PROMPT: &str = r#"
    Summary so far: 
    %SOFAR%

    Additional raw meeting transcript:
    %ADDITIONAL%

    "#;
const DEFAULT_SYSTEM_PROMPT: &str = r#"Act as a meeting secretary and write minutes for the additional transcript, following on from the summary so far."#;

impl Default for SummaryState {
    fn default() -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<SummaryUpdate>();
        Self {
            offset: 0,
            text: String::new(),
            user_prompt: DEFAULT_USER_PROMPT.to_string(),
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
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
    // this button triggers an OpenAI summary request
    if ui.button("Request OpenAI summary").clicked() {
        trigger_summarization_request(summary, text);
    }

    // this is the only way for the user to clear the offset for generating new AI summary
    if ui.button("Clear summary").clicked() {
        summary.offset = 0;
        summary.text = String::new();
    }

    // Since we're on the main thread here, we can see if there's any responses
    // that have been returned by a summary thread through the mpsc channel
    while let Ok(update) = summary.rx.try_recv() {
        match update {
            SummaryUpdate::Additional(additional) => {
                summary.text.push_str(format!("\n{}", additional).as_str());
                trigger_summarization_request(summary, text);
            }
        }
    }
}

fn trigger_summarization_request(summary: &mut SummaryState, raw: &str) {
    // 8000 chars is ~1500 tokens. Allowing for 10 lines of response (100
    // tokens) and 10 lines of context (100 tokens), we have a total size of
    // 1700 tokens. gpt-3.5-turbo-16k has a max_tokens of 16k; -turbo has a
    // max_tokens of 8k. So use whichever model makes sense (the smaller model
    // is half the price). So we can safely take 14k tokens of transcript, which
    // is 14k * 8000 / 1500 = 75k chars.
    //
    // change of plan
    //
    // after some quick tests, I find that this skips too much. Each request
    // seems to return about ten lines of summary, so taking 8000 chars (~160
    // lines) gives us one line of summary for every 16 lines of transcript. So
    // we'll take 8000 chars of transcript at a time.
    let additional = raw
        .chars()
        .skip(summary.offset)
        .take(8000)
        .collect::<String>();
    if additional.len() < 100 {
        tracing::warn!(
            "{} chars is not enough additional text to summarize",
            additional.len()
        );
        return;
    }
    // call openai to generate summary of additional text.
    tracing::info!(
        "requesting summary. Offset: {}, chars: {}",
        summary.offset,
        additional.len()
    );
    summary.offset += additional.len();

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
    thread::spawn(move || {
        openai_request(user_prompt, system_prompt, sender);
    });
}

/// Synchronously request a summary from openai, and send it back to the main thread.
fn openai_request(
    user_prompt: String,
    system_prompt: String,
    tx: std::sync::mpsc::Sender<SummaryUpdate>,
) {
    let client = Client::new();
    // if user + system > 35k chars, use -16k model
    let model = if user_prompt.len() + system_prompt.len() > 35000 {
        "gpt-3.5-turbo-16k"
    } else {
        "gpt-3.5-turbo"
    };
    let body = json!({
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_prompt }
        ],
        "model": model,
        "temperature": 0.7,
        "max_tokens": 256,
        "stop": ["\n\n", " Human:", " AI:"]
    });
    let apikey = if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        key
    } else {
        tracing::error!("OPENAI_API_KEY not set. Create an account and then a key at https://platform.openai.com/account/usage .");
        return;
    };
    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", apikey))
        .json(&body)
        .send()
        .expect("failed to send request");
    let response_json: serde_json::Value = response.json().expect("failed to parse response");
    tracing::info!("response: {:?}", response_json);
    let summary = response_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_else(|| {
            tracing::error!("failed to parse response: {:?}", response_json);
            ""
        })
        .to_string();
    tx.send(SummaryUpdate::Additional(summary))
        .expect("failed to send summary");
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
}
