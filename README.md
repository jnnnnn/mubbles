# Mubbles

Multiple real-time sub titles.

A tool to help you keep track of what people were saying while you weren't paying attention.

## Usage

Obtain a ggml-format Whisper model. You can get a pytorch formatted model from [huggingface](https://huggingface.co/openai/whisper-base) and then convert it to ggml using [this script](https://github.com/ggerganov/whisper.cpp/blob/master/models/convert-pt-to-ggml.py).

Once you have a model file, run the app:

```sh
cargo run
```