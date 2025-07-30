use std::sync::mpsc::Sender;
use crate::app::WhisperUpdate;
use anyhow::{Error as E, Result};

struct MyProgress {
    sender: Sender<WhisperUpdate>,
    total_size: usize,
    current: usize,
    fname: String,
}
impl MyProgress {
    fn new(sender: Sender<WhisperUpdate>) -> Self {
        MyProgress {
            sender,
            total_size: 0,
            current: 0,
            fname: String::new(),
        }
    }
}
impl hf_hub::api::Progress for MyProgress {
    fn init(&mut self, size: usize, filename: &str) {
        self.total_size = size;
        self.fname = filename.to_string();
    }
    fn update(&mut self, size: usize) {
        self.current += size;
        let percentage = (self.current as f64 / self.total_size as f64) * 100.0;

        self.sender
            .send(WhisperUpdate::Status(format!(
                "Downloading {}: {} MiB ({percentage:.1}%)",
                self.fname,
                self.current / (1024 * 1024)
            )))
            .unwrap_or_default();
    }
    fn finish(&mut self) {}
}

pub(crate) fn get_with_progress(
    repo: hf_hub::Repo,
    app: Sender<WhisperUpdate>,
    filename: &str,
) -> Result<std::path::PathBuf, E> {
    // unfortunately, api::download_with_progress does not check cache first, so we do it manually
    // this is made harder by ApiRepo not exposing the cache, so we have to use the Cache directly
    let cache = hf_hub::Cache::default();
    let cached = cache.repo(repo.clone()).get(filename);
    let path = if let Some(path) = cached {
        tracing::info!("Using cached file: {}", path.display());
        path
    } else {
        tracing::info!("Downloading file: {}", filename);
        hf_hub::api::sync::Api::new()?
            .repo(repo)
            .download_with_progress(filename, MyProgress::new(app))?
    };
    Ok(path)
}