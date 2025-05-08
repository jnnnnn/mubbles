## see if it's possible to update whisper.rs to do DTW timestamps without changing the candle model

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

## what are logits? why would they be less accurate?


Logits are the raw, unnormalized outputs of a neural network before applying a probability function like softmax. In the context of Whisper, logits represent the model's confidence scores for each possible token at each decoding step. These scores are used to determine the most likely next token during transcription.

Why Would Logits Be Less Accurate for Alignment?
No Temporal Information:

Logits only indicate the model's confidence for each token at a specific decoding step.
They do not inherently contain information about the relationship between tokens and audio frames, unlike attention weights, which explicitly model this relationship.
Indirect Alignment:

Attention weights directly map tokens to audio frames by showing how much each frame contributes to a token's prediction.
Using logits for alignment requires approximations (e.g., associating the highest logit value with a frame), which can introduce errors.
Lack of Context:

Attention weights consider the entire sequence of tokens and frames, providing a global view of alignment.
Logits are computed step-by-step and may not capture the broader context as effectively.
Noise and Ambiguity:

Logits can be noisy, especially for tokens with low confidence, making it harder to determine precise alignments.

## Whisper Paper on reliable long-form transcription

4.5. Strategies for Reliable Long-form Transcription
Transcribing long-form audio using Whisper relies on ac-
curate prediction of the timestamp tokens to determine the
amount to shift the model’s 30-second audio context win-
dow by, and inaccurate transcription in one window may
negatively impact transcription in the subsequent windows.
We have developed a set of heuristics that help avoid fail-
ure cases of long-form transcription, which is applied in
the results reported in sections 3.8 and 3.9. First, we use
beam search with 5 beams using the log probability as the
score function, to reduce repetition looping which happens
more frequently in greedy decoding. We start with tem-
perature 0, i.e. always selecting the tokens with the high-
est probability, and increase the temperature by 0.2 up to
1.0 when either the average log probability over the gen-
erated tokens is lower than −1 or the generated text has a
gzip compression rate higher than 2.4. Providing the tran-
scribed text from the preceding window as previous-text
conditioning when the applied temperature is below 0.5
further improves the performance. We found that the proba-
bility of the <|nospeech|> token alone is not sufficient
to distinguish a segment with no speech, but combining
the no-speech probability threshold of 0.6 and the average
log-probability threshold of −1 makes the voice activity
detection of Whisper more reliable. Finally, to avoid a fail-
ure mode where the model ignores the first few words in
the input, we constrained the initial timestamp token to be
between 0.0 and 1.0 second. Table 7 shows that adding each
of the interventions above incrementally reduces the WER
overall, but not evenly across the dataset. These heuristics
serve as a workaround for the noisy predictions of the model,
and more research would be needed to further improve the
reliability of long-form decoding.
Improved decoding strategies: As we have scaled Whis-
per, we have observed that larger models have made steady
and reliable progress on reducing perception-related errors
such as confusing similar-sounding words. Many remaining
errors, particularly in long-form transcription seem more
stubborn in nature and decidedly non-human/perceptual.
They are a combination of failure modes of seq2seq mod-
els, language models, and text-audio alignment and include
problems such as getting stuck in repeat loops, not tran-
scribing the first or last few words of an audio segment, or
complete hallucination where the model will output a tran-
script entirely unrelated to the actual audio. Although the
decoding details discussed in Section 4.5 help significantly,
we suspect fine-tuning Whisper models on a high-quality
supervised dataset and/or using reinforcement learning to
more directly optimize for decoding performance could help
further reduce these errors.


