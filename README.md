# Mubbles

An Egui wrapper around Whisper.cpp, the OpenAI speech-to-text model.

Can record from microphone and speakers.

## Usage

Obtain a ggml-format Whisper model. The easiest way is probably to download one from https://huggingface.co/ggerganov/whisper.cpp/tree/main, and also see more documentation about models [here].

The app expects to find a model at `./small.bin` or `~/.cache/whisper/small.bin`. 

I have found that the `small` model is a good balance between performance and quality. 
- If you're on a M1+ Mac or have a beefy Cuda card, maybe `medium` is better. 
- If you're on cpu-only, use `tiny`.

If necessary, you can also get a pytorch-format model from [huggingface](https://huggingface.co/openai/whisper-base) and then convert it to ggml using [this script](https://github.com/ggerganov/whisper.cpp/blob/master/models/convert-pt-to-ggml.py). 

Once you have a model file, run the app:

```sh
cargo run
```

You may need to remove the `cuda` feature from the `whisper-rs` dependency if you don't have a CUDA-capable GPU (and the cuda tookit installed). In this case, I would recommend using the Tiny model.

## Regular usage

Install the app into the default cargo bin directory (probably `~/.cargo/bin`):

    cargo install --path .

## Screenshot

![mubbles screenshot.png](./doc/mubbles-screenshot.png) 

