#[derive(Debug, Clone)]
pub struct AlignedWord {
    pub word: String,
    pub start: f32,
    pub end: f32,
    pub probability: f32,
    pub tokens: Vec<u32>,
}

/// Aligns tokens to words and assigns dummy timestamps.
pub fn align_words(
    tokens: &[u32],
    num_frames: usize,
    tokenizer: &tokenizers::Tokenizer,
) -> Vec<AlignedWord> {
    // Decode tokens into text
    let text = tokenizer.decode(tokens, true).unwrap_or_default();
    let words: Vec<&str> = text.split_whitespace().collect();

    // Calculate frame duration per word
    let num_words = words.len().max(1);
    let frame_per_word = num_frames as f32 / num_words as f32;

    let mut aligned_words = Vec::new();
    let mut current_frame = 0.0;
    let mut token_index = 0;

    for word in words {
        let start_frame = current_frame;
        let end_frame = (current_frame + frame_per_word).min(num_frames as f32);

        // Assign tokens to this word (naive: one token per word for now)
        let tokens_for_word = if token_index < tokens.len() {
            vec![tokens[token_index]]
        } else {
            vec![]
        };
        token_index += 1;

        aligned_words.push(AlignedWord {
            word: word.to_string(),
            start: start_frame,
            end: end_frame,
            probability: 1.0, // Placeholder probability
            tokens: tokens_for_word,
        });

        current_frame = end_frame;
    }

    aligned_words
}