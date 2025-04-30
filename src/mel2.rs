use hound::{WavReader, WavSpec};
use rubato::{Resampler, SincInterpolationParameters, SincInterpolator};
use rustfft::{num_complex::Complex, FftPlanner};
use ndarray::{Array2, Axis};

const SAMPLE_RATE: u32 = 16000;
const N_FFT: usize = 512;
const HOP_LENGTH: usize = 160;
const WIN_LENGTH: usize = 400;
const N_MELS: usize = 80;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load audio file
    let mut reader = WavReader::open("input.wav")?;
    let spec = reader.spec();
    
    // Convert to mono and f32 samples
    let samples: Vec<f32> = reader.samples()
        .collect::<Result<Vec<i16>, _>>()?
        .iter()
        .map(|s| *s as f32 / i16::MAX as f32)
        .collect();

    // 2. Resample to 16kHz if needed
    let resampled = if spec.sample_rate != SAMPLE_RATE {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            oversampling_factor: 512,
            window: rubato::WindowFunction::BlackmanHarris2,
        };
        
        let mut resampler = SincInterpolator::new(
            spec.sample_rate,
            SAMPLE_RATE,
            params,
            WIN_LENGTH,
            1
        );
        
        resampler.process(&[samples], None)?.into_iter().flatten().collect()
    } else {
        samples
    };

    // 3. Compute STFT
    let stft = compute_stft(&resampled);
    
    // 4. Create Mel filterbank
    let mel_filters = create_mel_filterbank();
    
    // 5. Apply Mel filterbank
    let mel_spec = apply_mel_filters(&stft, &mel_filters);
    
    // 6. Log compression
    let log_mel_spec = mel_spec.mapv(|x| (x + 1e-6).ln());
    
    // (Optional) Normalization would go here
    Ok(())
}

fn compute_stft(audio: &[f32]) -> Array2<f32> {
    let fft = FftPlanner::new().plan_fft_forward(N_FFT);
    let mut window = vec![0.0; WIN_LENGTH];
    apodize::hanning_iter(&mut window);
    
    let n_frames = 1 + (audio.len() - WIN_LENGTH) / HOP_LENGTH;
    let mut stft = Array2::zeros((N_FFT / 2 + 1, n_frames));
    
    for (i, frame) in audio.windows(WIN_LENGTH).step_by(HOP_LENGTH).enumerate() {
        let mut buffer: Vec<Complex<f32>> = frame.iter()
            .zip(&window)
            .map(|(s, w)| Complex::new(s * w, 0.0))
            .collect();
        
        // Zero-pad if needed
        buffer.resize(N_FFT, Complex::new(0.0, 0.0));
        
        // Compute FFT
        fft.process(&mut buffer);
        
        // Compute magnitude squared
        for (bin, complex) in buffer[..N_FFT/2+1].iter().enumerate() {
            stft[[bin, i]] = complex.norm_sqr();
        }
    }
    
    stft
}

fn create_mel_filterbank() -> Array2<f32> {
    let f_min = 0.0;
    let f_max = 8000.0;
    let n_fft = N_FFT as f32;
    
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let mel_points = linspace(mel_min, mel_max, N_MELS + 2);
    
    let hz_points = mel_points.iter().map(|&m| mel_to_hz(m)).collect::<Vec<_>>();
    let bin_freqs = (0..N_FFT/2 + 1).map(|i| i as f32 * SAMPLE_RATE as f32 / n_fft).collect::<Vec<_>>();
    
    let mut filters = Array2::zeros((N_MELS, N_FFT/2 + 1));
    
    for i in 0..N_MELS {
        let left = hz_points[i];
        let center = hz_points[i + 1];
        let right = hz_points[i + 2];
        
        for (j, &freq) in bin_freqs.iter().enumerate() {
            let weight = if freq < left || freq > right {
                0.0
            } else if freq <= center {
                (freq - left) / (center - left)
            } else {
                (right - freq) / (right - center)
            };
            
            filters[[i, j]] = weight;
        }
    }
    
    // Normalize filters
    let sum = filters.sum_axis(Axis(1));
    for i in 0..N_MELS {
        for j in 0..N_FFT/2 + 1 {
            filters[[i, j]] /= sum[i];
        }
    }
    
    filters
}

fn apply_mel_filters(stft: &Array2<f32>, filters: &Array2<f32>) -> Array2<f32> {
    filters.dot(stft)
}

// Helper functions
fn hz_to_mel(f: f32) -> f32 {
    2595.0 * (1.0 + f / 700.0).log10()
}

fn mel_to_hz(m: f32) -> f32 {
    700.0 * (10.0.powf(m / 2595.0) - 1.0)
}

fn linspace(start: f32, end: f32, num: usize) -> Vec<f32> {
    let step = (end - start) / (num - 1) as f32;
    (0..num).map(|i| start + i as f32 * step).collect()
}
