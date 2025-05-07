# Task tracking

This file tracks the things I want to work on next.

1. Word-level timestamps, and display them over the mel at the correct positions

### Word-level timestamps

Prompt:

> I want word-level timestamps to come from whisper.rs::decode(). Make a plan. The original python code is in whisper_word_align.rs

1. Understand the Python Reference The Python code uses cross-attention and dynamic time warping to align tokens to audio frames, then groups tokens into words and assigns start/end times. The main logic is in add_word_timestamps() and find_alignment().
2. Identify Required Data Token-to-frame alignment: For each token, you need to know which audio frames it aligns to. Word grouping: You need to group tokens into words (using the tokenizer). Timestamps: For each word, calculate start and end times based on aligned frames.
3. Expose Attention Weights in Rust The Python code uses attention weights from the model to perform alignment. Action: Modify your Rust model forward pass to optionally return cross-attention weights for each token.
4. Implement Alignment Logic in Rust Port the logic of find_alignment() to Rust: Use attention weights to align tokens to frames. Use dynamic time warping (DTW) or a similar algorithm to map tokens to time. Tip: You may need to add a new struct for alignment results (e.g., AlignedWord { word: String, start: f32, end: f32, probability: f32 }).
5. Group Tokens into Words Use the tokenizer to group consecutive tokens into words. For each word, aggregate the start and end frames from its tokens.
6. Calculate Word Timestamps Convert frame indices to seconds using the known hop length and sample rate. Assign start/end/probability to each word.
7. Return Word-Level Timestamps from decode() Extend your DecodingResult struct to include a Vec<AlignedWord>. Populate this field during decoding.
8. Update Downstream Code Wherever you use decode(), update to handle/display word-level timestamps.
9. Testing Write unit tests to verify that word timestamps are reasonable and match expected output.
10. (Optional) Refactor Consider moving alignment logic to a separate module for clarity.

