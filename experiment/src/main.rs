use std::{sync::Arc, thread};

use hound::WavReader;
use image::GrayImage;
use ndarray::{Array2, Axis};
use rubato::Resampler;
use rustfft::{num_complex::Complex, FftPlanner};

use candle_transformers::models::whisper::{audio, Config};

const SAMPLE_RATE: u32 = 16000;
const N_FFT: usize = 512;
const HOP_LENGTH: usize = 160;
const WIN_LENGTH: usize = 400;
const N_MELS: usize = 80;

// type alias for ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start1 = std::time::Instant::now();
    let mel1 = custom()?;
    let elapsed1 = start1.elapsed();
    println!("Custom mel spectrogram took: {:?}", elapsed1);

    let start2 = std::time::Instant::now();
    let mel2 = candle_audio()?;
    let elapsed2 = start2.elapsed();
    println!("Candle mel spectrogram took: {:?}", elapsed2);

    let start3 = std::time::Instant::now();
    let mel3 = candle_rufft_audio()?;
    let elapsed3 = start3.elapsed();
    println!("Candle rufft mel spectrogram took: {:?}", elapsed3);

    Ok(())
}

fn candle_audio() -> Result<(), Box<dyn std::error::Error>> {
    let resampled = get_samples()?;
    // tinyen config: INFO mubbles::whisper: Config: Config { num_mel_bins: 80, max_source_positions: 1500, d_model: 384, encoder_attention_heads: 6, encoder_layers: 4, vocab_size: 51864, max_target_positions: 448, decoder_attention_heads: 6, decoder_layers: 4, suppress_tokens: [1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279, 1303, 1343, 1377, 1391, 1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211, 4600, 4808, 5299, 5855, 6329, 7203, 9609, 9959, 10563, 10786, 11420, 11709, 11907, 13163, 13697, 13700, 14808, 15306, 16410, 16791, 17992, 19203, 19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409, 34949, 40283, 40493, 40549, 47282, 49146, 50257, 50357, 50358, 50359, 50360, 50361] }
    let config = Config {
        num_mel_bins: N_MELS,
        max_source_positions: 1500,
        d_model: 384,
        encoder_attention_heads: 6,
        encoder_layers: 4,
        vocab_size: 51864,
        max_target_positions: 448,
        decoder_attention_heads: 6,
        decoder_layers: 4,
        suppress_tokens: vec![
            1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90,
        ],
    };

    let mel_filters = get_mel_filters();

    let mel = audio::pcm_to_mel(&config, &resampled, &mel_filters);

    let mel_array = Array2::from_shape_vec((N_MELS, mel.len() / N_MELS), mel)
        .map_err(|_| "Failed to reshape mel vector into Array2")?;

    save_mel_as_image(&mel_array, "mel_spectrogram-candle.png")?;

    Ok(())
}

fn get_mel_filters() -> Vec<f32> {
    let mel_bytes = include_bytes!("../../src/melfilters.bytes").as_slice();
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);
    mel_filters
}

fn custom() -> Result<(), Box<dyn std::error::Error>> {
    let mut resampled = get_samples()?;

    let target_length = SAMPLE_RATE as usize * 30;
    resampled.resize(target_length, 0.0);

    // 3. Compute STFT
    let stft = compute_stft(&resampled);

    // 4. Use the same Mel filterbank as candle_audio
    let mel_filters = create_mel_filterbank();
    //let filters = &Array2::from_shape_vec((N_MELS, mel_filters.len() / N_MELS), mel_filters)?;

    // 5. Apply Mel filterbank
    let mel_spec = apply_mel_filters(&stft, &mel_filters);

    // 6. Log compression
    let log_mel_spec = mel_spec.mapv(|x| (x + 1e-6).ln());

    // 7. Normalize and save as image
    save_mel_as_image(&log_mel_spec, "mel_spectrogram-custom.png")?;

    Ok(())
}

fn get_samples() -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = WavReader::open(r"C:\Users\J\Desktop\jfk.wav")?;
    let spec = reader.spec();
    let samples: Vec<f32> = reader
        .samples()
        .collect::<Result<Vec<i16>, _>>()?
        .iter()
        .map(|s| *s as f32 / i16::MAX as f32)
        .collect();
    let resampled = if spec.sample_rate != SAMPLE_RATE {
        let mut resampler = rubato::FftFixedInOut::new(
            // Replaced SincInterpolator with FftFixedInOut
            spec.sample_rate as usize,
            SAMPLE_RATE as usize,
            WIN_LENGTH,
            1,
        )?;

        resampler
            .process(&[samples], None)?
            .into_iter()
            .flatten()
            .collect() // Added missing second argument
    } else {
        samples
    };
    Ok(resampled)
}

fn compute_stft(audio: &[f32]) -> Array2<f32> {
    let fft = FftPlanner::new().plan_fft_forward(N_FFT);
    let window: Vec<f32> = apodize::hanning_iter(WIN_LENGTH)
        .map(|x| x as f32)
        .collect();

    let n_frames = 1 + (audio.len() - WIN_LENGTH) / HOP_LENGTH;
    let mut stft = Array2::zeros((N_FFT / 2 + 1, n_frames));

    for (i, frame) in audio.windows(WIN_LENGTH).step_by(HOP_LENGTH).enumerate() {
        let mut buffer: Vec<Complex<f32>> = frame
            .iter()
            .zip(&window)
            .map(|(s, w)| Complex::new(s * w, 0.0))
            .collect();

        // Zero-pad if needed
        buffer.resize(N_FFT, Complex::new(0.0, 0.0));

        // Compute FFT using rustfft
        fft.process(&mut buffer);

        // Compute magnitude squared
        for (bin, complex) in buffer[..N_FFT / 2 + 1].iter().enumerate() {
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
    let bin_freqs = (0..N_FFT / 2 + 1)
        .map(|i| i as f32 * SAMPLE_RATE as f32 / n_fft)
        .collect::<Vec<_>>();

    let mut filters = Array2::zeros((N_MELS, N_FFT / 2 + 1));

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
        for j in 0..N_FFT / 2 + 1 {
            filters[[i, j]] /= sum[i];
        }
    }

    filters
}

fn apply_mel_filters(stft: &Array2<f32>, filters: &Array2<f32>) -> Array2<f32> {
    let result = filters.dot(stft);
    // log shapes, including output shape
    println!("stft shape: {:?}", stft.shape());
    println!("filters shape: {:?}", filters.shape());
    println!("result shape: {:?}", result.shape());
    result
}

// Helper functions
fn hz_to_mel(f: f32) -> f32 {
    2595.0 * (1.0 + f / 700.0).log10()
}

fn mel_to_hz(m: f32) -> f32 {
    700.0 * (10.0_f32.powf(m / 2595.0) - 1.0) // Specified type for powf
}

fn linspace(start: f32, end: f32, num: usize) -> Vec<f32> {
    let step = (end - start) / (num - 1) as f32;
    (0..num).map(|i| start + i as f32 * step).collect()
}

fn save_mel_as_image(
    mel_spec: &Array2<f32>,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let min_val = mel_spec.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = mel_spec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let normalized: Vec<u8> = mel_spec
        .iter()
        .map(|&x| ((x - min_val) / (max_val - min_val) * 255.0) as u8)
        .collect();

    let width = mel_spec.shape()[1] as u32;
    let height = mel_spec.shape()[0] as u32;

    let img = GrayImage::from_raw(width, height, normalized)
        .ok_or("Failed to create image from raw data")?;

    img.save(filename)?;
    Ok(())
}

mod mel3;
use mel3::log_mel_spectrogram_;

fn candle_rufft_audio() -> Result<(), Box<dyn std::error::Error>> {
    let resampled = get_samples()?;

    let mel_filters = get_mel_filters();

    let mel = log_mel_spectrogram_(&resampled, &mel_filters, 400, 160, N_MELS, false);

    let mel_array = Array2::from_shape_vec((N_MELS, mel.len() / N_MELS), mel)
        .map_err(|_| "Failed to reshape mel vector into Array2")?;

    save_mel_as_image(&mel_array, "mel_spectrogram-candle-rufft.png")?;

    Ok(())
}
