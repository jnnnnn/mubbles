To determine if it's possible to update whisper.rs to perform DTW-based timestamps without modifying the Candle model, we need to analyze the following:

Key Observations from the Original Whisper Code
Attention Weights:

The original Whisper implementation uses cross-attention weights from the decoder to align tokens with audio frames.
These weights are extracted during the decoding process and used for alignment.
Dynamic Time Warping (DTW):

DTW is applied to map tokens to audio frames based on the attention weights.
The alignment logic is implemented in Python functions like find_alignment and add_word_timestamps.
Candle Model:

The Candle implementation of Whisper does not currently expose attention weights directly.
Modifying the Candle model to expose these weights would require changes to the MultiHeadAttention and ResidualAttentionBlock implementations.
Can We Perform DTW Without Modifying the Candle Model?
Yes, it is possible, but with some limitations:

Use Logits Instead of Attention Weights:

Instead of using attention weights, we can use the logits from the decoder to approximate token-to-frame alignment.
This approach is less precise but avoids modifying the Candle model.
Post-Processing Logits:

During decoding, capture the logits for each token at every step.
Use these logits to estimate the alignment between tokens and audio frames.
Implement DTW in whisper.rs:

Implement a DTW algorithm in Rust to map tokens to frames based on the captured logits.
This can be done entirely within whisper.rs.
Steps to Implement DTW Timestamps Without Modifying Candle
Capture Logits During Decoding:

Modify the decode method in whisper.rs to store the logits for each token.
Estimate Token-to-Frame Alignment:

Use the logits to approximate the alignment between tokens and audio frames.
For example, the frame with the highest logit value for a token can be considered its aligned frame.
Apply DTW:

Implement a DTW algorithm in Rust to refine the alignment between tokens and frames.
Generate Word Timestamps:

Group tokens into words and calculate start/end timestamps for each word based on the aligned frames.
Limitations
Using logits instead of attention weights may result in less accurate alignments.
This approach assumes that the logits provide sufficient information for alignment, which may not always be the case.
Next Steps
Would you like me to:

Implement the changes to capture logits during decoding?
Add a DTW implementation in whisper.rs?
Explore modifying the Candle model to expose attention weights for more accurate alignment?