use rustfft::{num_complex::Complex, FftPlanner};

const FFT_SIZE: usize = 400; // 200 real + 200 imaginary
const FFT_STEP: usize = 160; // 10ms at 16kHz

#[allow(clippy::too_many_arguments)]
fn log_mel_spectrogram_w(
    samples: &[f32],
    filters: &[f32],
    n_len: usize, // number of frames (usually 3000 for a 30s audio clip)
    n_mel: usize, // number of mel bins (80 or 128)
) -> Vec<f32> {
    let hann = hanning_window();
    let n_fft = 1 + FFT_SIZE / 2; // 201

    let zero = 0.0f32;
    let mut fft_in = vec![zero; FFT_SIZE];
    let mut mel = vec![zero; n_len * n_mel];
    let n_samples = samples.len();
    let end = std::cmp::min(n_samples / FFT_STEP + 1, n_len);

    for frame_index in 0..end {
        let offset = frame_index * FFT_STEP;

        for j in 0..std::cmp::min(FFT_SIZE, n_samples - offset) {
            fft_in[j] = hann[j] * samples[offset + j];
        }

        if n_samples - offset < FFT_SIZE {
            fft_in[n_samples - offset..].fill(zero);
        }

        let mut fft_out: Vec<Complex<f32>> = fft(&fft_in);

        // We only need the magnitude of the FFT output
        for j in 0..FFT_SIZE {
            fft_out[j] = Complex::new(
                fft_out[j].re * fft_out[j].re + fft_out[j].im * fft_out[j].im,
                0.0f32,
            );
        }
        for j in 1..FFT_SIZE / 2 {
            let v = fft_out[FFT_SIZE - j].re;
            fft_out[j].re += v;
        }

        // mel spectrogram
        for bin in 0..n_mel {
            let mut sum = zero;
            let mut fbin = 0;
            // Unroll loop
            while fbin < n_fft.saturating_sub(3) {
                sum += fft_out[fbin].re * filters[bin * n_fft + fbin]
                    + fft_out[fbin + 1].re * filters[bin * n_fft + fbin + 1]
                    + fft_out[fbin + 2].re * filters[bin * n_fft + fbin + 2]
                    + fft_out[fbin + 3].re * filters[bin * n_fft + fbin + 3];
                fbin += 4;
            }
            // Handle remainder
            while fbin < n_fft {
                sum += fft_out[fbin].re * filters[bin * n_fft + fbin];
                fbin += 1;
            }
            mel[bin * n_len + frame_index] = f32::max(sum, 1e-10f32).log10();
        }
    }
    mel
}

fn pad(samples: &[f32]) -> (usize, Vec<f32>) {
    // pad audio with at least one extra chunk of zeros
    const CHUNK_LENGTH: usize = 30;
    let pad = 100 * CHUNK_LENGTH / 2; // 1500
    let n_len = samples.len() / FFT_STEP;
    let n_len = if n_len % pad != 0 {
        (n_len / pad + 1) * pad
    } else {
        n_len
    };
    let n_len = n_len + pad;
    let samples = {
        let mut samples_padded = samples.to_vec();
        let to_add = n_len * FFT_STEP - samples.len();
        samples_padded.extend(std::iter::repeat_n(0f32, to_add));
        samples_padded
    };
    (n_len, samples)
}

fn hanning_window() -> Vec<f32> {
    let one = 1.0f32;
    let half = 0.5f32;
    let two_pi = std::f32::consts::PI + std::f32::consts::PI;
    let hann: Vec<f32> = (0..FFT_SIZE)
        .map(|i| half * (one - ((two_pi * i as f32) / FFT_SIZE as f32).cos()))
        .collect();
    hann
}

// spread the range from 0 to 1. Anything more than 8dB below the maximum is set to 0.
pub fn normalize(mel: &mut Vec<f32>) {
    let threshold = mel
        .iter()
        .max_by(|&u, &v| u.partial_cmp(v).unwrap_or(std::cmp::Ordering::Greater))
        .copied()
        .unwrap_or(0f32)
        - 8f32;
    for m in mel.iter_mut() {
        let v = f32::max(*m, threshold);
        *m = v / 4f32 + 1f32
    }
}

fn fft(inp: &[f32]) -> Vec<Complex<f32>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(inp.len());
    let mut buffer: Vec<Complex<f32>> = inp.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut buffer);
    buffer
}

/// Converts PCM audio to a mel spectrogram using the provided mel filters.
/// The mel filters are used to convert the FFT output to the mel scale.
/// Each mel frame contains n_mel frequency bins (80 or 128), representing low to high frequencies.
/// Each mel frame covers a time window of 160 samples (10ms at 16kHz).
pub(crate) fn pcm_to_mel(n_mel: usize, resampled: &[f32], mel_filters: &[f32]) -> Vec<f32> {
    let (n_len, samples) = pad(resampled);
    let mut mel = log_mel_spectrogram_w(&samples, &mel_filters, n_len, n_mel);
    normalize(&mut mel);
    mel
}

// this is for incremental processing. Provide 400 samples of audio, and it will
// return a mel frame. each time you get another 160 samples, call this function
// again (after discarding the oldest 160 samples) to get the next frame.
pub(crate) fn pcm_to_mel_frame(
    n_mel: usize,
    resampled: &[f32],
    mel_filters: &[f32],
) -> Vec<[f32; 80]> {
    let n_frames = (resampled.len() + FFT_STEP - FFT_SIZE) / FFT_STEP; // number of frames
    let mel = log_mel_spectrogram_w(&resampled, &mel_filters, n_frames, n_mel);

    // quickly calculate a rough histogram of the mel values for debugging. ten bins.
    let mut mel2 = mel.clone();
    mel2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut hist = [0f32; 10];
    for (i, chunk) in mel2.chunks(mel2.len() / 10).enumerate() {
        let min = chunk.first().copied().unwrap_or(0f32);
        hist[i] = min;
    }
    // silence is -10, most values -8ish, quite a strong frequency is -5.
    tracing::debug!("Mel {} histogram: {:?} ", mel.len(), hist);

    let mut mel_frames = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let mut frame = [-10.0f32; 80];
        for bin in 0..n_mel {
            frame[bin] = mel[bin * n_frames + f];
        }
        mel_frames.push(frame);
    }
    mel_frames
}
