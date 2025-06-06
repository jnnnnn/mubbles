use rustfft::{num_complex::Complex, FftPlanner};

#[allow(clippy::too_many_arguments)]
fn log_mel_spectrogram_w(
    hann: &[f32],
    samples: &[f32],
    filters: &[f32],
    fft_size: usize, // 400: 200 real + 200 im
    fft_step: usize, // 160
    n_len: usize,    // number of frames (usually 3000 for a 30s audio clip)
    n_mel: usize,    // number of mel bins (80 or 128)
) -> Vec<f32> {
    let n_fft = 1 + fft_size / 2; // 201

    let zero = 0.0f32;
    let mut fft_in = vec![zero; fft_size];
    let mut mel = vec![zero; n_len * n_mel];
    let n_samples = samples.len();
    let end = std::cmp::min(n_samples / fft_step + 1, n_len);

    for frame_index in 0..end {
        let offset = frame_index * fft_step;

        // apply Hanning window
        for j in 0..std::cmp::min(fft_size, n_samples - offset) {
            fft_in[j] = hann[j] * samples[offset + j];
        }

        // fill the rest with zeros
        if n_samples - offset < fft_size {
            fft_in[n_samples - offset..].fill(zero);
        }

        // FFT
        let mut fft_out: Vec<Complex<f32>> = fft(&fft_in);

        // Calculate modulus^2 of complex numbers
        for j in 0..fft_size {
            fft_out[j] = Complex::new(
                fft_out[j].re * fft_out[j].re + fft_out[j].im * fft_out[j].im,
                0.0f32,
            );
        }
        for j in 1..fft_size / 2 {
            let v = fft_out[fft_size - j].re;
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

pub fn log_mel_spectrogram_(
    samples: &[f32],
    filters: &[f32],
    fft_size: usize,
    fft_step: usize,
    n_mel: usize,
) -> Vec<f32> {
    let hann = hanning_window(fft_size);

    // pad audio with at least one extra chunk of zeros
    const CHUNK_LENGTH: usize = 30;
    let pad = 100 * CHUNK_LENGTH / 2;
    let n_len = samples.len() / fft_step;
    let n_len = if n_len % pad != 0 {
        (n_len / pad + 1) * pad
    } else {
        n_len
    };
    let n_len = n_len + pad;
    let samples = {
        let mut samples_padded = samples.to_vec();
        let to_add = n_len * fft_step - samples.len();
        samples_padded.extend(std::iter::repeat_n(0f32, to_add));
        samples_padded
    };

    log_mel_spectrogram_w(&hann, &samples, &filters, fft_size, fft_step, n_len, n_mel)
}

fn hanning_window(fft_size: usize) -> Vec<f32> {
    let one = 1.0f32;
    let half = 0.5f32;
    let two_pi = std::f32::consts::PI + std::f32::consts::PI;
    let fft_size_t = fft_size as f32;
    let hann: Vec<f32> = (0..fft_size)
        .map(|i| half * (one - ((two_pi * i as f32) / fft_size_t).cos()))
        .collect();
    hann
}

fn normalize(mel: &mut Vec<f32>) {
    let mmax = mel
        .iter()
        .max_by(|&u, &v| u.partial_cmp(v).unwrap_or(std::cmp::Ordering::Greater))
        .copied()
        .unwrap_or(0f32)
        - 8f32;
    for m in mel.iter_mut() {
        let v = f32::max(*m, mmax);
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
    let mut mel = log_mel_spectrogram_(&resampled, &mel_filters, 400, 160, n_mel);
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
) -> Vec<[f32; 128]> {
    let mut mel = log_mel_spectrogram_(&resampled, &mel_filters, 400, 160, n_mel);
    normalize(&mut mel);
    let n_frames = (resampled.len() - 240) / 160;
    let mut mel_frames = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let mut frame = [0f32; 128];
        for j in 0..n_mel {
            frame[j] = mel[i * n_mel + j];
        }
        mel_frames.push(frame);
    }
    mel_frames
}
