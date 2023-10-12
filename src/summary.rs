use std::{
    collections::{HashMap, HashSet},
    thread,
};

use reqwest::blocking::Client;
use serde_json::json;

pub struct SummaryState {
    pub offset: usize, // everything before this character has already been summarized
    pub text: String,
    pub output_words: usize,
    pub input_lines: usize,
    tx: std::sync::mpsc::Sender<SummaryUpdate>,
    pub rx: std::sync::mpsc::Receiver<SummaryUpdate>,
}

pub enum SummaryUpdate {
    Additional(String),
}

impl Default for SummaryState {
    fn default() -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<SummaryUpdate>();
        Self {
            offset: 0,
            text: String::new(),
            output_words: 5,
            input_lines: 7,
            tx,
            rx,
        }
    }
}

fn trigger_summarization_request(state: &SummaryState, raw: &str) {
    let additional = raw
        .chars()
        .skip(state.offset)
        .take(8000)
        .collect::<String>();
    // call openai to generate summary of additional text.
    tracing::info!(
        "requesting summary. Offset: {}, chars: {}",
        state.offset,
        additional.len()
    );
    let user_prompt = format!(
        "
    Summary so far: 
    {}

    Additional raw transcript:
    {}

    ",
        state.text, additional
    );
    let system_prompt = format!(
        "
    You are a concise and helpful secretary. 
    Given the additional raw text, 
    fix up the summary so far 
    by adding additional points. 
    Each point can be either an action item 
    or an important fact. 
    Each point begins with a heading, followed by a colon.
    "
    );

    let sender = state.tx.clone();

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
    let body = json!({
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_prompt }
        ],
        "model": "gpt-3.5-turbo-16k",
        "temperature": 0.7,
        "max_tokens": 256,
        "stop": ["\n\n", " Human:", " AI:"]
    });
    let apikey = if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        key
    } else {
        tracing::error!("OPENAI_API_KEY not set");
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
    let summary = response_json["choices"][0]["text"]
        .as_str()
        .unwrap()
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
    // count the words, splitting at any non-alpha character and ignoring whitespace
    let mut word_counts = HashMap::new();
    for (index, word) in additional.split(|c: char| !c.is_alphabetic()).enumerate() {
        // strip non-alpha
        let word = word
            .chars()
            .filter(|c| c.is_alphabetic())
            .collect::<String>()
            .to_lowercase();
        if word.trim().len() <= 3 || ignored.contains(word.as_str()) {
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
        assert_eq!(state.text, "\n- gaia everybody spoke whenever catch");
        assert_eq!(state.offset, 502);
        statistical_summary(&mut state, raw, 10, 5);
        // make sure the last line is right
        let last_line = state.text.lines().last().unwrap();
        assert_eq!(
            last_line,
            "- asdifnkoasnidf progressing risks concern aware"
        );
        // make sure we end up at the end of the text
        assert_eq!(state.offset, 668);
        assert_eq!(state.offset, raw.len());
    }
}
