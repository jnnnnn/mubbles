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

pub fn log_mel_spectrogram(
    audio: &[f32],
    num_mel_bins: usize,
) -> Vec<f32> {
    let mut padded_audio = Vec::with_capacity(PADDING);
    padded_audio.extend_from_slice(audio);
    padded_audio.resize(PADDING, 0.0);

    let hann = hann_window(N_FFT);

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft(N_FFT, rustfft::FftDirection::Forward);
    let mut buffer: Vec<Complex<f32>> = padded_audio.iter().enumerate().map(|(i, &x)| Complex::new(x * hann[i % N_FFT], 0.0)).collect();
    buffer.resize(N_FFT, Complex::new(0.0, 0.0));
    fft.process(&mut buffer);

    let magnitudes: Vec<f32> = buffer.iter().map(|c| c.norm_sqr()).collect();
    let mel_filters = load_mel_filters(num_mel_bins);
    let mut mel_spec = vec![0.0; num_mel_bins];
    for (i, mel_filter) in mel_filters.chunks(N_FFT / 2).enumerate() {
        mel_spec[i] = mel_filter.iter().zip(&magnitudes).map(|(f, m)| f * m).sum();
    }

    let log_spec: Vec<f32> = mel_spec
        .iter()
        .map(|&m| (m.max(1e-10).log10()).max(mel_spec.iter().cloned().fold(f32::MIN, f32::max) - 8.0))
        .map(|v| (v + 4.0) / 4.0)
        .collect();

    log_spec
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

    // Generate Mel filters (mocked for simplicity)
    let filters = vec![1.0; num_mel_bins * (N_FFT / 2 + 1)];

    // Compute the log Mel spectrogram using both implementations
    let mel_result = mel_log_mel_spectrogram(&audio, num_mel_bins);
    let cfg = candle_transformers::models::whisper::Config {
        num_mel_bins,
        max_source_positions: 0,
        d_model: 0,
        encoder_attention_heads: 0,
        encoder_layers: 0,
        vocab_size: 0,
        max_target_positions: 0,
        decoder_attention_heads: 0,
        decoder_layers: 0,
        suppress_tokens: vec![],
    };
    let audio_result = audio_log_mel_spectrogram(&cfg, &audio, &filters);

    // Compare the results
    assert_eq!(mel_result.len(), audio_result.len());
    for (mel_value, audio_value) in mel_result.iter().zip(audio_result.iter()) {
        assert!((mel_value - audio_value).abs() < 1e-5, "Values differ: {} vs {}", mel_value, audio_value);
    }
}

