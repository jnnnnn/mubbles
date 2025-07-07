# Task tracking

This file tracks the things I want to work on next.

0. fix up the level-based voice detection to work better with low volume audio. maybe an adjustable (logarithmic) threshold. also make the plot logarithmic and show the level.
0. implement proper voice detection using silero. the current levels approach is ok but doesn't work well with low volume audio. 
1. optimize fft / cuda to run entirely on GPU (using either a custom kernel or cuFFT) and not transfer data back and forth for transcribing
2. implement quantization so the large model can run well in 8GB
3. debug why faster-whisper is so much faster than candle
4. add beam search to improve transcription quality like faster-whisper
5. add a tab for transcribing audio files (drag and drop or select file)
6. add a playback button where you click a button and it plays the audio from the current line of text (the line the cursor is on) to help correcting poor transcriptions
7. fix up summarization so it is properly continuous
8. add a diagram mode where the summarization generates nodes and edges as the conversation progresses
9. add an option to use a local llm for summarization
10. 