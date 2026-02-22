/// Simulate keyboard input to type text into the foreground application.
///
/// Platform support:
/// - **Windows**: Uses `winput::send_str()` (Win32 SendInput)
/// - **Linux/Wayland**: Uses `wtype` (must be installed)
/// - **Linux/X11**: Uses `xdotool type` (must be installed)

/// Type the given text into the currently focused window, appending a trailing space.
pub fn type_text(text: &str) {
    let text = format!("{} ", text.trim());
    if text.trim().is_empty() {
        return;
    }
    if let Err(e) = type_text_platform(&text) {
        tracing::warn!("Autotype failed: {e}");
    }
}

#[cfg(windows)]
fn type_text_platform(text: &str) -> Result<(), String> {
    let failed = winput::send_str(text);
    if failed > 0 {
        Err(format!("{failed} inputs failed to send"))
    } else {
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn type_text_platform(text: &str) -> Result<(), String> {
    // Detect session type from environment
    let session_type = std::env::var("XDG_SESSION_TYPE").unwrap_or_default();

    let (cmd, args) = if session_type.contains("wayland") {
        ("wtype", vec!["--", text])
    } else {
        // X11 or unknown — try xdotool
        ("xdotool", vec!["type", "--clearmodifiers", "--", text])
    };

    match std::process::Command::new(cmd).args(&args).output() {
        Ok(output) if output.status.success() => Ok(()),
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("{cmd} failed: {stderr}"))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(format!(
            "{cmd} not found. Install it: {}",
            if cmd == "wtype" {
                "sudo apt install wtype (or equivalent)"
            } else {
                "sudo apt install xdotool (or equivalent)"
            }
        )),
        Err(e) => Err(format!("{cmd} error: {e}")),
    }
}

// Unsupported platforms compile but warn at runtime
#[cfg(not(any(windows, target_os = "linux")))]
fn type_text_platform(_text: &str) -> Result<(), String> {
    Err("Autotype is not supported on this platform".to_string())
}
