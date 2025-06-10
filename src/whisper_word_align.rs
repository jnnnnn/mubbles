use candle_core::{IndexOp, Result, Tensor};

use candle_transformers::models::whisper as m;

use crate::whisper_model::AlignmentHead;

use tokenizers::Tokenizer;

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

#[derive(Debug, Clone)]
pub struct AlignedWord {
    pub word: String,
    pub start: f64,
    pub end: f64,
    pub probability: f32,
}

struct Word {
    pub word: String,
    pub tokens: Vec<u32>,
}

pub fn align(
    query_key_tensors: &Vec<Tensor>,
    alignment_heads: &Vec<AlignmentHead>,
    text_tokens: &[u32],
    text_token_probs: &[f32],
    prefix_len: usize,
    tokenizer: &Tokenizer,
) -> Result<Vec<AlignedWord>> {
    const TIME_PER_AUDIO_FRAME: f64 = 0.02; // 20ms per audio token (2 Mel spectrogram frames)

    const MAX_TEXT_TOKENS: usize = 50; // more than this means the model has gone crazy repeating; it's slow, so skip.
    if text_tokens.len() > MAX_TEXT_TOKENS {
        tracing::warn!(
            "Too many text tokens ({}) for alignment, skipping alignment.",
            text_tokens.len()
        );
        return Ok(Vec::new());
    }

    // each text token starts at a particular audio frame:
    let text_token_audio_frames =
        align_text_token_to_audio(query_key_tensors, alignment_heads, prefix_len)?;
    // use the tokenizer to go back from u32 text tokens to unicode strings
    let word_token_groups = decode_to_unicode(text_tokens, tokenizer)?;

    let mut aligned_words = Vec::new();
    for (text_idx, audio_idx) in text_token_audio_frames.iter().enumerate() {
        if text_idx < prefix_len || text_idx >= text_tokens.len() {
            continue; // Skip prefix tokens or out-of-bounds indices
        }
        // if text_idx has incremented, we're on a new token, so end the previous word

        let word = &word_token_groups[text_idx - prefix_len];
        let start_time = *audio_idx as f64 * TIME_PER_AUDIO_FRAME;
        let probability = text_token_probs.get(text_idx).cloned().unwrap_or(0.0);

        aligned_words.push(AlignedWord {
            word: word.word.clone(),
            start: start_time,
            end: 0.0, // will be set later
            probability,
        });
    }
    // end time is the start time of the next word, so we need to adjust it
    for i in 0..aligned_words.len() - 1 {
        aligned_words[i].end = aligned_words[i + 1].start;
    }
    // the last word's end time is the end of the audio
    if let Some(last_word) = aligned_words.last_mut() {
        let audio_duration = text_token_audio_frames.len() as f64 * TIME_PER_AUDIO_FRAME;
        last_word.end = audio_duration as f64;
    }
    Ok(aligned_words)
}

/// Splits the text tokens into word tokens based on the tokenizer.
/// It works by starting with one token and then adding tokens until the tokenizer decodes a full word.
fn decode_to_unicode(text_tokens: &[u32], tokenizer: &Tokenizer) -> Result<Vec<Word>> {
    let replacement_char = '\u{FFFD}'; // Unicode replacement character

    let mut result = Vec::<Word>::new();

    let mut current_tokens = Vec::<u32>::new();

    for &token in text_tokens {
        current_tokens.push(token);
        let subword = tokenizer
            .decode(&current_tokens, false)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenizer decode error: {e}")))?;

        if !subword.contains(char::from(replacement_char)) {
            // If the decoded string does not contain the replacement character,
            // it means we have valid unicode.
            result.push(Word {
                word: subword,
                tokens: current_tokens.clone(),
            });
            current_tokens.clear();
        }
    }
    Ok(result)
}

fn align_text_token_to_audio(
    query_key_tensors: &Vec<Tensor>,
    alignment_heads: &Vec<AlignmentHead>,
    prefix_len: usize,
) -> Result<Vec<usize>> {
    // What is qk? Each element `qk[h, i, j]` represents the attention score
    // between the `i`-th query token (from the text token sequence (so far))
    // and the `j`-th key token (from the key (audio token) sequence) for the
    // `h`-th attention head in the `b`-th batch. openai uses batching as an
    // extra dim0 but candle whisper currently doesn't. qk_receiver gets a qk
    // for each block (layer), so query_key_tensors is [layer][head,
    // text_query_token, audio_key_token]. For example, [6][8, 14, 1500] for
    // base.en with 6 layers, 8 heads per layer, 14 tokens generated so far, and
    // 1500 audio tokens representing 30s of audio.

    // The slice will panic if there aren't enough text tokens compared to prefix_len, so check.
    const TIME_PER_AUDIO_TOKEN: f32 = 0.02; // 20ms per audio token (2 Mel spectrogram frames)

    let dims = query_key_tensors[0].dims();
    let (max_layer, max_head) = alignment_heads
        .iter()
        .fold((0, 0), |(max_layer, max_head), head| {
            (max_layer.max(head.layer), max_head.max(head.head))
        });
    if dims.len() != 3
        || query_key_tensors.len() <= max_layer
        || dims[0] <= max_head
        || dims[1] <= prefix_len
        || dims[2] < 10
    // model really doesn't work at all with less than 200ms of audio
    {
        return Err(candle_core::Error::Msg(format!(
            "query_key_tensors shape too small, got shape [{}]{dims:?}, needed [l][h, i, j] at least [{max_layer}][{max_head}, {prefix_len}, 10]",
            query_key_tensors.len(),
        )));
    }

    // we now squash all this together into a single tensor of only the bits we care about: [usefulhead, i, j]
    // audio_tokens has already pruned the mel that represents the padding out to 30s
    // py: # heads * tokens * frames
    // py: weights = torch.stack([QKs[_l][_h] for _l, _h in model.alignment_heads.indices().T])
    let useful_slices: Vec<Tensor> = alignment_heads
        .iter()
        .map(|head| -> Result<Tensor> {
            let layer = head.layer;
            let head = head.head;
            if layer >= query_key_tensors.len() || head >= query_key_tensors[layer].dims()[0] {
                return Err(candle_core::Error::Msg(format!(
                    "Invalid layer {layer} or head {head} for query_key_tensors with shape [layers:{}][heads, texttokens, audiotokens:{:?}]",
                    query_key_tensors.len(), query_key_tensors[layer].dims()
                )));
            }
            tracing::debug!(
                "processing alignment head: layer {layer}, head {head} with dims {:?}",
                query_key_tensors[layer].dims()
            );
            // error here: DEBUG mubbles::whisper_word_align: processing alignment head: layer 1, head 0 with dims [20, 17, 750]
            // ERROR mubbles::whisper: Whisper thread failed: narrow invalid args start + len > dim_len: [17, 750], dim: 1, start: 0, len:1414
            // remove ..real_audio_tokens across j ? 1414 seems like about double 750...
            // audio tokens is already trimmed I think, no need to slice j dimension
            Ok(query_key_tensors[layer].i((head, prefix_len.., ../*real_audio_tokens*/))?)
        })
        .collect::<Result<_>>()?;
    // the python shape is now [alignmenthead][i, j]
    // we stack it to [alignmenthead, i, j] shape as candlenn wants to softmax the last dim
    // softmax takes in real numbers and outputs a probability distribution
    let weights = Tensor::stack(&useful_slices, 0)?;
    // py: weights = weights[:, :, : num_frames // 2]
    // py: weights = (weights * qk_scale).softmax(dim=-1)
    // qk_scale always 1
    let weights = candle_nn::ops::softmax_last_dim(&weights)?;
    // py: std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
    // this is a mean across dim `i`, the text query tokens
    let text_query_mean = weights.mean_keepdim(1)?;
    let text_query_std = weights.var_keepdim(1)?.sqrt()?;
    // py: weights = (weights - mean) / std
    let weights = (weights.broadcast_sub(&text_query_mean))?.broadcast_div(&text_query_std)?;
    // py: weights = median_filter(weights, medfilt_width)
    // pytorch median filters along the last dimension (j), and medfilt_width is 7
    // candle doesn't have a median filter, so we do it ourselves
    // this just smooths the attention weights a bit, removing outliers
    let weights = median_filter(&weights, 7, 1)?;
    // py: matrix = weights.mean(axis=0)
    // we now average over all the attention heads to get an [i, j] shape,
    // where i is the text query token and j is the audio key token.
    let matrix = weights.mean(0)?.neg()?;
    // py: matrix = matrix[len(tokenizer.sot_sequence) : -1]
    // we did this earlier
    // py: text_indices, time_indices = dtw(-matrix)
    // find a path through the cost matrix (top left to bottom right) using "dynamic time warping"
    let text_token_timesteps = dtw_path_from_matrix(&matrix)?;

    // we now have a path through the cost matrix, which is a sequence of (text_token_index, audio_frame_index) pairs.

    // iterate along the path, adding start and end times to each word, finding the start and end of each text token
    // based on monotonic duplicated text_idx.
    // for example, if text_audio_path is 0-0, 0-1, 0-2, 1-3, 1-4, 1-5, 1-6, 2-7, 2-8
    // then we have text tokens at 0..2, 1..6, 7..8.
    // Build a list of where each text token starts: 0, 1, 7.
    let mut text_idx_to_audio_idx: Vec<usize> = Vec::new();
    let mut last_text_idx = None;
    for (text_idx, audio_idx) in &text_token_timesteps {
        if let Some(last) = last_text_idx {
            if last != *text_idx {
                text_idx_to_audio_idx.push(*audio_idx);
            }
        } else {
            text_idx_to_audio_idx.push(*audio_idx);
        }
        last_text_idx = Some(*text_idx);
    }

    tracing::debug!(
        "audio steps where text_tokens changes: {:?}",
        text_idx_to_audio_idx,
    );

    return Ok(text_idx_to_audio_idx);
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_dtw_path_from_matrix_basic() {
        let device = Device::Cpu;
        // Cost matrix: text_tokens (rows) x audio_frames (cols)
        let n_rows = 3; // text tokens
        let n_cols = 4; // audio frames

        // Create a simple cost matrix tensor
        // Lower cost along a diagonal path
        let cost_matrix_data = vec![
            0.1f32, 5.0, 5.0, 5.0, // token 0
            5.0, 0.2, 5.0, 5.0, // token 1
            5.0, 5.0, 0.3, 5.0, // token 2
        ];
        let cost_tensor = Tensor::from_vec(cost_matrix_data, (n_rows, n_cols), &device).unwrap();

        let result = dtw_path_from_matrix(&cost_tensor);
        assert!(
            result.is_ok(),
            "dtw_path_from_matrix returned an error: {:?}",
            result.err()
        );
        let path = result.unwrap();

        assert_eq!(
            path,
            vec![(0, 0), (1, 1), (2, 2), (2, 3)],
            "Path should follow the diagonal",
        );

        // Test with empty matrix
        let empty_tensor = Tensor::from_slice(&[0f32; 0], (0, n_cols), &device).unwrap();
        let empty_path_result = dtw_path_from_matrix(&empty_tensor);
        assert!(empty_path_result.is_ok());
        assert!(
            empty_path_result.unwrap().is_empty(),
            "Path should be empty for zero rows."
        );

        let empty_tensor_cols = Tensor::from_slice(&[0f32; 0], (n_rows, 0), &device).unwrap();
        let empty_path_result_cols = dtw_path_from_matrix(&empty_tensor_cols);
        assert!(empty_path_result_cols.is_ok());
        assert!(
            empty_path_result_cols.unwrap().is_empty(),
            "Path should be empty for zero columns."
        );
    }

    #[test]
    fn test_align_basic() {
        let device = Device::Cpu;

        let num_text_tokens = 3; // i
        let num_audio_frames = 10; // j
        let num_alignment_heads_to_use = 2; // h (interpreted as number of heads to select)
        let prefix_len = 1;

        // Create alignment_heads
        // These will select specific heads from specific layers in query_key_tensors
        let alignment_heads_vec: Vec<AlignmentHead> = vec![
            AlignmentHead { layer: 0, head: 0 },
            AlignmentHead { layer: 1, head: 1 },
        ];

        // Create query_key_tensors
        // query_key_tensors is Vec<Tensor>. Each Tensor is for a layer.
        // Shape of Tensor for a layer: (num_actual_heads_in_layer, num_text_tokens, num_audio_frames)
        // For this test, layer 0 must have at least `num_alignment_heads_to_use` heads.
        let shape = (
            num_alignment_heads_to_use, // number of heads
            num_text_tokens,            // number of text tokens
            num_audio_frames,           // number of audio frames
        );
        // this test has 2 heads, and 3 text tokens, and 10 audio frames. the
        // first token is aligned to the start of the audio, the second token is
        // aligned to the middle of the audio, and the third token is aligned to
        // the end of the audio.
        #[rustfmt::skip]
        let layer = Tensor::from_slice(
                &[
                    0.9f32, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,
                    0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,
                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,

                    0.9f32, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,
                    0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,
                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                ],
                shape,
                &device,
            )
            .unwrap();

        let query_key_tensors_vec = vec![layer.clone(), layer.clone()];

        let result =
            align_text_token_to_audio(&query_key_tensors_vec, &alignment_heads_vec, prefix_len);
        assert!(
            result.is_ok(),
            "align function returned an error: {:?}",
            result.err()
        );
        let path = result.unwrap();

        let dtw_rows = num_text_tokens - prefix_len;
        let dtw_cols = num_audio_frames;

        if dtw_rows == 0 || dtw_cols == 0 {
            assert!(
                path.is_empty(),
                "Path should be empty if effective DTW dimensions are zero after prefix removal."
            );
        } else {
            assert!(
                !path.is_empty(),
                "Path should not be empty for the given dimensions after prefix removal."
            );
        }
    }
}
