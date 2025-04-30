/*
Original Whisper implementation:


# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = N_SAMPLES,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 and 128 are supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (n_mels, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
*/

const SAMPLE_RATE: usize = 16000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const CHUNK_LENGTH: usize = 30; // seconds
const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE; // 480000 samples in a 30-second chunk
const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH; // 3000 frames in a mel spectrogram input
const N_SAMPLES_PER_TOKEN: usize = HOP_LENGTH * 2; // the initial convolutions has stride 2
const FRAMES_PER_SECOND: usize = SAMPLE_RATE / HOP_LENGTH; // 10ms per audio frame
const TOKENS_PER_SECOND: usize = SAMPLE_RATE / N_SAMPLES_PER_TOKEN; // 20ms per audio token

const N_MELS: usize = 80; // Number of Mel-frequency filters, only 80 and 128 are supported

const PADDING: usize = N_SAMPLES; // Number of zero samples to pad to the right

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size as f32 - 1.0)).cos()))
        .collect()
}

fn pad_audio(audio: &[f32], padding: usize) -> Vec<f32> {
    let mut padded_audio = Vec::with_capacity(padding);
    padded_audio.extend_from_slice(audio);
    padded_audio.resize(padding, 0.0);
    padded_audio
}

fn compute_fft(audio: &[f32], hann: &[f32], fft_size: usize) -> Vec<f32> {
    // Ensure the Hann window size matches the FFT size
    assert_eq!(hann.len(), fft_size, "Hann window size must match FFT size");

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft(fft_size, rustfft::FftDirection::Forward);

    let mut magnitudes = Vec::new();

    for chunk in audio.chunks(fft_size) {
        let mut fft_input: Vec<Complex<f32>> = chunk
            .iter()
            .zip(hann.iter().cycle()) // Cycle the Hann window if the chunk is smaller
            .map(|(&sample, &window)| Complex::new(sample * window, 0.0))
            .collect();

        fft_input.resize(fft_size, Complex::new(0.0, 0.0));
        fft.process(&mut fft_input);

        magnitudes.extend(fft_input.iter().take(fft_size / 2 + 1).map(|c| c.norm_sqr()));
    }

    magnitudes
}

fn apply_mel_filters(magnitudes: &[f32], mel_filters: &[f32], n_fft: usize) -> Vec<f32> {
    mel_filters
        .chunks(n_fft)
        .map(|filter| {
            magnitudes
                .iter()
                .zip(filter)
                .map(|(magnitude, &filter_value)| magnitude * filter_value)
                .sum()
        })
        .collect()
}

fn normalize_log_spec(log_spec: &[f32]) -> Vec<f32> {
    let max_log_spec = log_spec.iter().cloned().fold(f32::MIN, f32::max) - 8.0;
    log_spec
        .iter()
        .map(|&v| ((v.max(max_log_spec) + 4.0) / 4.0))
        .collect()
}

pub fn log_mel_spectrogram(
    audio: &[f32],
    num_mel_bins: usize,
) -> Vec<f32> {
    // Adjust padding to align with pcm_to_mel
    let fft_step = HOP_LENGTH;
    let n_len = (audio.len() + fft_step - 1) / fft_step;
    let padded_audio_len = n_len * fft_step;
    let mut padded_audio = audio.to_vec();
    padded_audio.resize(padded_audio_len, 0.0);

    let hann = hann_window(N_FFT);
    let magnitudes = compute_fft(&padded_audio, &hann, N_FFT);
    let mel_filters = load_mel_filters(num_mel_bins);
    let mel_spec = apply_mel_filters(&magnitudes, &mel_filters, N_FFT / 2 + 1);

    // Normalize the Mel spectrogram to match pcm_to_mel behavior
    let max_mel_spec = mel_spec.iter().cloned().fold(f32::MIN, f32::max) - 8.0;
    let log_spec: Vec<f32> = mel_spec
        .iter()
        .map(|&m| (m.max(1e-10).log10()).max(max_mel_spec))
        .collect();

    let normalized_log_spec = normalize_log_spec(&log_spec);

    normalized_log_spec
}

fn load_mel_filters (num_mel_bins: usize) -> Vec<f32> {
    const EMPTY_SLICE: &[u8; 0] = &[0u8; 0];
    let mel_bytes = match num_mel_bins {
        80 => include_bytes!("melfilters.bytes").as_slice(),
        128 => include_bytes!("melfilters128.bytes").as_slice(),
        _nmel => EMPTY_SLICE,
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);
    mel_filters
}

#[test]
fn test_log_mel_spectrogram_comparison() {
    use crate::mel::log_mel_spectrogram as mel_log_mel_spectrogram;
    use candle_transformers::models::whisper::audio::pcm_to_mel as audio_log_mel_spectrogram;

    // Example audio data (sine wave for simplicity)
    let sample_rate = SAMPLE_RATE as f32;
    let duration = 1.0; // 1 second
    let frequency = 440.0; // A4 note
    let num_samples = (sample_rate * duration) as usize;
    let audio: Vec<f32> = (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * frequency * i as f32 / sample_rate).sin())
        .collect();

    // Number of Mel bins
    let num_mel_bins = 80;

    // Generate Mel filters using the same logic as in `load_mel_filters`
    let filters = load_mel_filters(num_mel_bins);

    // Compute the log Mel spectrogram using both implementations
    let mel_result = mel_log_mel_spectrogram(&audio, num_mel_bins);
    let cfg = candle_transformers::models::whisper::Config {
        num_mel_bins,
        max_source_positions: 3000, // Matches N_FRAMES
        d_model: 512, // Example value, ensure it aligns with the implementation
        encoder_attention_heads: 8, // Example value
        encoder_layers: 6, // Example value
        vocab_size: 10000, // Example value
        max_target_positions: 512, // Example value
        decoder_attention_heads: 8, // Example value
        decoder_layers: 6, // Example value
        suppress_tokens: vec![],
    };
    let audio_result = audio_log_mel_spectrogram(&cfg, &audio, &filters);

    // Compare the results
    assert_eq!(mel_result.len(), audio_result.len());
    for (mel_value, audio_value) in mel_result.iter().zip(audio_result.iter()) {
        assert!((mel_value - audio_value).abs() < 1e-5, "Values differ: {} vs {}", mel_value, audio_value);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_pad_audio() {
        let audio = vec![1.0, 2.0, 3.0];
        let padded = pad_audio(&audio, 5);
        assert_eq!(padded, vec![1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn test_compute_fft() {
        let audio = vec![1.0, 0.0, -1.0, 0.0];
        let hann = hann_window(4);
        let magnitudes = compute_fft(&audio, &hann, 4);
        assert_eq!(magnitudes.len(), 3); // N_FFT / 2 + 1
        
    }

    #[test]
    fn test_apply_mel_filters() {
        let magnitudes = vec![1.0, 2.0, 3.0];
        let mel_filters = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let mel_spec = apply_mel_filters(&magnitudes, &mel_filters, 3);
        assert_eq!(mel_spec, vec![3.0, 3.0]);
    }

    #[test]
    fn test_normalize_log_spec() {
        let log_spec = vec![0.0, -1.0, -2.0];
        let normalized = normalize_log_spec(&log_spec);
        assert!(normalized.iter().all(|&v| v >= 0.0 && v <= 1.0));
    }
}

