use rustfft::{FftPlanner, num_complex::Complex};

#[allow(clippy::too_many_arguments)]
// https://github.com/ggerganov/whisper.cpp/blob/4774d2feb01a772a15de81ffc34b34a1f294f020/whisper.cpp#L2414
fn log_mel_spectrogram_w(
    hann: &[f32],
    samples: &[f32],
    filters: &[f32],
    fft_size: usize,
    fft_step: usize,
    speed_up: bool,
    n_len: usize,
    n_mel: usize,
) -> Vec<f32> {
    let n_fft = if speed_up {
        1 + fft_size / 4
    } else {
        1 + fft_size / 2
    };

    let zero = 0.0f32;
    let half = 0.5f32;
    let mut fft_in = vec![zero; fft_size];
    let mut mel = vec![zero; n_len * n_mel];
    let n_samples = samples.len();
    let end = std::cmp::min(n_samples / fft_step + 1, n_len);

    for i in 0..end {
        let offset = i * fft_step;

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

        if speed_up {
            // scale down in the frequency domain results in a speed up in the time domain
            for j in 0..n_fft {
                fft_out[j].re = half * (fft_out[2 * j].re + fft_out[2 * j + 1].re);
            }
        }

        // mel spectrogram
        for j in 0..n_mel {
            let mut sum = zero;
            let mut k = 0;
            // Unroll loop
            while k < n_fft.saturating_sub(3) {
                sum += fft_out[k].re * filters[j * n_fft + k]
                    + fft_out[k + 1].re * filters[j * n_fft + k + 1]
                    + fft_out[k + 2].re * filters[j * n_fft + k + 2]
                    + fft_out[k + 3].re * filters[j * n_fft + k + 3];
                k += 4;
            }
            // Handle remainder
            while k < n_fft {
                sum += fft_out[k].re * filters[j * n_fft + k];
                k += 1;
            }
            mel[j * n_len + i] = f32::max(sum, 1e-10f32).log10();
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
    speed_up: bool,
) -> Vec<f32> {
    const CHUNK_LENGTH: usize = 30;
    let zero = 0.0f32;
    let two_pi = std::f32::consts::PI + std::f32::consts::PI;
    let half = 0.5f32;
    let one = 1.0f32;
    let four = 4.0f32;
    let fft_size_t = fft_size as f32;

    let hann: Vec<f32> = (0..fft_size)
        .map(|i| half * (one - ((two_pi * i as f32) / fft_size_t).cos()))
        .collect();
    let n_len = samples.len() / fft_step;

    // pad audio with at least one extra chunk of zeros
    let pad = 100 * CHUNK_LENGTH / 2;
    let n_len = if n_len % pad != 0 {
        (n_len / pad + 1) * pad
    } else {
        n_len
    };
    let n_len = n_len + pad;
    let samples = {
        let mut samples_padded = samples.to_vec();
        let to_add = n_len * fft_step - samples.len();
        samples_padded.extend(std::iter::repeat_n(zero, to_add));
        samples_padded
    };


    // use scope to allow for non static references to be passed to the threads
    // and directly collect the results into a single vector
    let mut mel = log_mel_spectrogram_w(
        &hann, &samples, &filters, fft_size, fft_step, speed_up, n_len, n_mel,
    );

    let mmax = mel
        .iter()
        .max_by(|&u, &v| u.partial_cmp(v).unwrap_or(std::cmp::Ordering::Greater))
        .copied()
        .unwrap_or(zero)
        - 8.0f32;
    for m in mel.iter_mut() {
        let v = f32::max(*m, mmax);
        *m = v / four + one
    }
    mel
}

pub trait Float: num_traits::Float + num_traits::FloatConst + num_traits::NumAssign + num_traits::FromPrimitive + num_traits::Signed + std::fmt::Debug + Send + Sync + 'static {}

impl Float for f32 {}

// https://github.com/ggerganov/whisper.cpp/blob/4774d2feb01a772a15de81ffc34b34a1f294f020/whisper.cpp#L2357
fn fft(inp: &[f32]) -> Vec<Complex<f32>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(inp.len());

    // Convert input to Complex numbers
    let mut buffer: Vec<Complex<f32>> = inp.iter().map(|&x| Complex::new(x, 0.0)).collect();

    // Perform FFT
    fft.process(&mut buffer);

    buffer
}