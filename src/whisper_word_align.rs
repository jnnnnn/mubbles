use candle_core::{IndexOp, Result, Tensor};

use candle_transformers::models::whisper as m;

const HOP_LENGTH: usize = 160;
const SAMPLE_RATE: usize = 16000;
// Time per audio frame fed to the decoder, after encoder's downsampling by 2
const TIME_PER_AUDIO_FRAME: f64 = (2.0 * HOP_LENGTH as f64) / SAMPLE_RATE as f64;

#[derive(Debug, Clone)]
pub struct WordTiming {
    pub word: String,
    pub tokens: Vec<u32>,
    pub start: f64,
    pub end: f64,
    pub probability: f64,
}

// Helper function for DTW backtracing.
// trace is an (N+1)x(M+1) matrix where N is number of rows in original cost matrix (text tokens)
// and M is number of columns (audio frames).
// Returns a path as Vec<(text_idx, time_idx)>, where indices are 0-based.
fn backtrace(trace: &Vec<Vec<i8>>, n_rows: usize, n_cols: usize) -> Vec<(usize, usize)> {
    let mut path = Vec::new();
    // 1-based for trace, corresponds to 0-based n_rows-1 in x
    let mut row = n_rows;
    let mut col = n_cols;

    while row > 0 || col > 0 {
        path.push((row.saturating_sub(1), col.saturating_sub(1)));
        if row == 0 {
            col = col.saturating_sub(1);
        } else if col == 0 {
            row = row.saturating_sub(1);
        } else {
            match trace[row][col] {
                0 => {
                    row = row.saturating_sub(1);
                    col = col.saturating_sub(1);
                }
                1 => {
                    row = row.saturating_sub(1);
                }
                2 => {
                    col = col.saturating_sub(1);
                }
                _ => {
                    if row > 0 && col > 0 {
                        row -= 1;
                        col -= 1;
                    } else if row > 0 {
                        row -= 1;
                    } else {
                        col -= 1;
                    }
                }
            }
        }
    }
    path.reverse();
    path
}

/// Calculates the optimal path through a cost matrix using Dynamic Time Warping.
/// x_tensor is the input cost matrix (e.g., -log_probabilities),
/// where rows correspond to text tokens and columns to audio frames.
pub(crate) fn dtw_path_from_matrix(x_tensor: &Tensor) -> Result<Vec<(usize, usize)>> {
    let x_dims = x_tensor.dims();
    if x_dims.len() != 2 {
        return Err(candle_core::Error::Msg(format!(
            "Cost matrix for DTW must be 2D, got shape {:?}",
            x_dims
        )));
    }
    let n_rows = x_dims[0]; // Number of text tokens
    let n_cols = x_dims[1]; // Number of audio frames

    if n_rows == 0 || n_cols == 0 {
        return Ok(Vec::new()); // Return an empty path if either dimension is zero
    }

    let x_data: Vec<Vec<f32>> = x_tensor.to_vec2()?;

    // Dimensions are (n_rows+1) x (n_cols+1) to handle boundary conditions.
    let mut cost = vec![vec![f32::INFINITY; n_cols + 1]; n_rows + 1];
    let mut trace = vec![vec![-1i8; n_cols + 1]; n_rows + 1]; // -1 indicates unvisited/error

    cost[0][0] = 0.0; // Cost of aligning empty sequences is 0.

    for i in 1..=n_rows {
        for j in 1..=n_cols {
            let c_diag = cost[i - 1][j - 1];
            let c_up = cost[i - 1][j];
            let c_left = cost[i][j - 1];

            let (min_prev_cost, move_idx) = if c_diag <= c_up && c_diag <= c_left {
                (c_diag, 0i8) // 0 for diagonal
            } else if c_up <= c_diag && c_up <= c_left {
                (c_up, 1i8) // 1 for up
            } else {
                (c_left, 2i8) // 2 for left
            };

            // Current cell's cost in x_data is x_data[i-1][j-1] due to 0-indexing.
            cost[i][j] = x_data[i - 1][j - 1] + min_prev_cost;
            trace[i][j] = move_idx;
        }
    }

    let path = backtrace(&trace, n_rows, n_cols);
    Ok(path)
}

pub(crate) fn median_filter(tensor: &Tensor, filter_width: usize, dim: usize) -> Result<Tensor> {
    if filter_width % 2 == 0 {
        return Err(candle_core::Error::Msg(
            "Filter width must be an odd number".to_string(),
        ));
    }
    if dim >= tensor.dims().len() {
        return Err(candle_core::Error::Msg(format!(
            "Dimension {} is out of bounds for tensor with shape {:?}",
            dim,
            tensor.dims()
        )));
    }
    
    // todo: finish this implementation. not sure how important it is.
    
    Ok(tensor.clone())
}

pub fn set_attention_hooks(
    decoder: &mut m::model::TextDecoder,
) -> Result<std::sync::mpsc::Receiver<Tensor>> {
    let (tx, rx) = std::sync::mpsc::channel();
    for i in 0..decoder.n_blocks() {
        let tx_clone = tx.clone();
        decoder.set_attention_hook(
            i,
            Some(Box::new(move |qk: &Tensor| {
                let qk_slice = qk.i((0, .., .., ..))?; // Extract the relevant slice
                tx_clone
                    .send(qk_slice)
                    .map_err(|e| candle_core::Error::Msg(format!("Failed to send tensor: {}", e)))
            })),
        );
    }

    Ok(rx)
}

pub fn clear_attention_hooks(decoder: &mut m::model::TextDecoder) {
    for i in 0..decoder.n_blocks() {
        decoder.set_attention_hook(i, None);
    }
}

