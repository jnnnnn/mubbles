# Mubbles

An Egui wrapper around HuggingFace's Candle Whisper model, which implements Whisper, the OpenAI speech-to-text model.

Can record from microphone and speakers.

## Usage

Run the app.

Cuda builds require Nvidia / Cuda drivers.

Huggingface models will be downloaded automatically when transcription starts.

Some things are still a bit broken because I don't use them (this is really a personal project).

PRs welcome but I may not get to them for a few weeks.

## Regular usage

Install the app into the default cargo bin directory (probably `~/.cargo/bin`):

    cargo install --path . --features cuda

Depending on the accelerator you have available, choose which feature -- `mkl` for intel CPUs or `cuda` for Nvidia cards with CudaNN installed in the system.

## Screenshot

![mubbles screenshot.png](./doc/mubbles-screenshot.png) 

## Build on ubuntu

```sh
apt install build-essential libssl-dev pkg-config libasound2-dev
```

Also, if you don't have cuda but do have an intel cpu, use --features mkl. 

https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html?operatingsystem=linux&linux-install=apt

## Roadmap

1. Summarization should use a standard configurable endpoint
2. UI is a bit crowded, simplify
3. fix autotype
4. fix partials to show the mel spectrogram in real time
   - last attempt at this failed, wrangling texture memory is a little complex in egui
5. fix input file transcription
6. implement snippet saving, where you can click on a fragment and hear it / save the audio file.
7. figure out why the accuracy for the first and last word is so bad
8. implement a better speech detection / segmenting algorithm. 
    - I've had a go with silero and:
    - couldn't get the model to download automatically / easily and
    - I don't like onnx because it's not super transparent and
    - adds a whole lot of dependencies. 

