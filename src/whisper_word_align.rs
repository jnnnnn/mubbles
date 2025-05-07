#[derive(Debug, Clone)]
pub struct AlignedWord {
    pub word: String,
    pub start: f32,
    pub end: f32,
    pub probability: f32,
    pub tokens: Vec<u32>,
}

/// Align tokens to frames using Dynamic Time Warping (DTW).
/// This function assumes placeholder attention weights for now.
pub fn find_alignment_with_dtw(
    tokens: &[u32],
    num_frames: usize,
    attention_weights: &[Vec<f32>], // Placeholder for attention weights
    tokenizer: &tokenizers::Tokenizer,
) -> Vec<AlignedWord> {
    // Decode tokens into text
    let text = tokenizer.decode(tokens, true).unwrap_or_default();
    let words: Vec<&str> = text.split_whitespace().collect();

    // Placeholder: Generate dummy attention weights if not provided
    let attention_weights = if attention_weights.is_empty() {
        vec![vec![1.0 / num_frames as f32; num_frames]; tokens.len()]
    } else {
        attention_weights.to_vec()
    };

    // DTW alignment: Map tokens to frames
    let mut token_to_frame = vec![0; tokens.len()];
    let mut frame_to_token = vec![0; num_frames];

    // Naive DTW implementation (replace with optimized version later)
    for (token_idx, token_weights) in attention_weights.iter().enumerate() {
        let max_frame = token_weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(frame_idx, _)| frame_idx)
            .unwrap_or(0);
        token_to_frame[token_idx] = max_frame;
        frame_to_token[max_frame] = token_idx;
    }

    // Group tokens into words and assign timestamps
    let mut aligned_words = Vec::new();
    let mut current_frame = 0;
    let mut token_index = 0;

    for word in words {
        let start_frame = token_to_frame.get(token_index).cloned().unwrap_or(current_frame);
        let end_frame = token_to_frame
            .get(token_index + word.len())
            .cloned()
            .unwrap_or(start_frame + 1);

        // Assign tokens to this word
        let tokens_for_word = tokens[token_index..token_index + word.len().min(tokens.len())].to_vec();
        token_index += word.len();

        aligned_words.push(AlignedWord {
            word: word.to_string(),
            start: start_frame as f32 / num_frames as f32,
            end: end_frame as f32 / num_frames as f32,
            probability: 1.0, // Placeholder probability
            tokens: tokens_for_word,
        });

        current_frame = end_frame;
    }

    aligned_words
}