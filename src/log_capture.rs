//! In-memory log capture for UI display
//!
//! Provides a tracing layer that captures log events in a circular buffer
//! for real-time display in the application UI.

use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use tracing::{Event, Subscriber};
use tracing::span::Attributes;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

/// Maximum number of log entries to keep in memory
const MAX_LOG_ENTRIES: usize = 1000;

/// Convert log level index to tracing::Level
pub fn index_to_tracing_level(index: usize) -> tracing::Level {
    match index {
        0 => tracing::Level::TRACE,
        1 => tracing::Level::DEBUG,
        2 => tracing::Level::INFO,
        3 => tracing::Level::WARN,
        4 => tracing::Level::ERROR,
        _ => tracing::Level::INFO, // Default to INFO
    }
}

/// A single log entry
#[derive(Clone, Debug)]
pub struct LogEntry {
    pub timestamp: chrono::DateTime<chrono::Local>,
    pub level: tracing::Level,
    pub target: String,
    pub message: String,
}

impl LogEntry {
    /// Format the log entry for display
    pub fn format(&self) -> String {
        format!(
            "[{}] {:5} {}: {}",
            self.timestamp.format("%H:%M:%S%.3f"),
            self.level.to_string(),
            self.target,
            self.message
        )
    }
}

/// Shared log level control
#[derive(Clone)]
pub struct LogLevelControl {
    level: Arc<Mutex<tracing::Level>>,
}

impl LogLevelControl {
    pub fn new(initial_level: tracing::Level) -> Self {
        Self {
            level: Arc::new(Mutex::new(initial_level)),
        }
    }

    pub fn set_level(&self, level: tracing::Level) {
        *self.level.lock().unwrap() = level;
    }

    pub fn get_level(&self) -> tracing::Level {
        *self.level.lock().unwrap()
    }
}

/// Thread-safe log buffer
#[derive(Clone)]
pub struct LogBuffer {
    entries: Arc<Mutex<VecDeque<LogEntry>>>,
    level_control: LogLevelControl,
}

impl LogBuffer {
    /// Create a new log buffer
    pub fn new() -> Self {
        Self {
            entries: Arc::new(Mutex::new(VecDeque::with_capacity(MAX_LOG_ENTRIES))),
            level_control: LogLevelControl::new(tracing::Level::INFO),
        }
    }

    /// Get the level control for this buffer
    pub fn level_control(&self) -> &LogLevelControl {
        &self.level_control
    }

    /// Add a log entry to the buffer
    fn push(&self, entry: LogEntry) {
        let mut entries = self.entries.lock().unwrap();
        if entries.len() >= MAX_LOG_ENTRIES {
            entries.pop_front();
        }
        entries.push_back(entry);
    }

    /// Get all log entries (already filtered at capture time)
    pub fn get_all(&self) -> Vec<LogEntry> {
        self.entries.lock().unwrap().iter().cloned().collect()
    }

    /// Clear all log entries
    pub fn clear(&self) {
        self.entries.lock().unwrap().clear()
    }
}

impl Default for LogBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Tracing layer that captures events to memory
pub struct LogCaptureLayer {
    buffer: LogBuffer,
}

impl LogCaptureLayer {
    /// Create a new log capture layer
    pub fn new(buffer: LogBuffer) -> Self {
        Self { buffer }
    }
}

impl<S> Layer<S> for LogCaptureLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let metadata = event.metadata();
        
        // Filter out messages below the minimum level
        let min_level = self.buffer.level_control().get_level();
        if *metadata.level() > min_level {
            return;
        }
        
        // Create a visitor to extract the message
        struct MessageVisitor(String);
        
        impl tracing::field::Visit for MessageVisitor {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0 = format!("{:?}", value);
                    // Remove surrounding quotes from debug format
                    if self.0.starts_with('"') && self.0.ends_with('"') {
                        self.0 = self.0[1..self.0.len()-1].to_string();
                    }
                } else if !self.0.is_empty() {
                    self.0.push_str(&format!(", {}={:?}", field.name(), value));
                } else {
                    self.0 = format!("{}={:?}", field.name(), value);
                }
            }
        }
        
        let mut visitor = MessageVisitor(String::new());
        event.record(&mut visitor);
        
        let entry = LogEntry {
            timestamp: chrono::Local::now(),
            level: *metadata.level(),
            target: metadata.target().to_string(),
            message: visitor.0,
        };
        
        self.buffer.push(entry);
    }

    fn on_new_span(&self, _attrs: &Attributes<'_>, _id: &tracing::span::Id, _ctx: Context<'_, S>) {
        // We don't need to track spans for the log display
    }
}

/// Logger for appending transcriptions to monthly files
#[derive(Clone)]
pub struct TranscriptionLogger {
    last_written_minute: Arc<Mutex<Option<chrono::DateTime<chrono::Local>>>>,
}

impl TranscriptionLogger {
    pub fn new() -> Self {
        Self {
            last_written_minute: Arc::new(Mutex::new(None)),
        }
    }

    /// Append transcription text to the monthly file in the given folder.
    /// If the minute has changed since the last write, prepends an ISO timestamp.
    pub fn append(&self, folder: &str, text: &str) {
        if folder.is_empty() {
            return;
        }

        let folder = shellexpand::tilde(folder);
        let folder_path = std::path::Path::new(folder.as_ref());

        // Create folder if it doesn't exist
        if let Err(e) = std::fs::create_dir_all(folder_path) {
            tracing::error!("Failed to create transcription folder: {}", e);
            return;
        }

        let now = chrono::Local::now();
        let filename = format!("{}.txt", now.format("%Y-%m"));
        let file_path = folder_path.join(&filename);

        // Check if we need to write a timestamp (minute changed)
        let mut last_minute = self.last_written_minute.lock().unwrap();
        let need_timestamp = match *last_minute {
            Some(last) => now.format("%Y-%m-%d %H:%M").to_string() 
                != last.format("%Y-%m-%d %H:%M").to_string(),
            None => true,
        };

        let mut content = String::new();
        if need_timestamp {
            content.push_str(&format!("\n[{}]\n", now.format("%Y-%m-%dT%H:%M:%S%:z")));
        }
        content.push_str(text);
        content.push('\n');

        // Append to file
        use std::io::Write;
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
        {
            Ok(mut file) => {
                if let Err(e) = file.write_all(content.as_bytes()) {
                    tracing::error!("Failed to write to transcription file: {}", e);
                } else {
                    *last_minute = Some(now);
                    tracing::debug!("Appended transcription to {}", file_path.display());
                }
            }
            Err(e) => {
                tracing::error!("Failed to open transcription file {}: {}", file_path.display(), e);
            }
        }
    }
}

impl Default for TranscriptionLogger {
    fn default() -> Self {
        Self::new()
    }
}
