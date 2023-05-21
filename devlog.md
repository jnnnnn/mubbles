# devlog

## 2023-05-06

started dev. I tried out a few other whisper desktop things but they don't work very well.

I also didn't find one that will record what both the mic and the speakers are saying, in two separate columns. I feel like that's a pretty cool feature.

## 2023-05-07

First step: get the audio from the mic and the speakers. Seems like the best rust audio library is soundio. See if I can get it to work at all.

Soundio doesn't compile:

    C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.34.31933\include\stdatomic.h(15,1): fatal  error C1189: #error:  <stdatomic.h> is not yet supported when compiling as C, but this is planned for a future release. [C:\Users\J\Source\mubbles\target\debug\build\libsoundio-sys-e7b60699a3621a70\out\build\libsoundio_static.vcxproj]

Google (and copilot) suggests cpal. Try that instead. Run the example. It works. Sweet, let's use that instead.

OMG, it worked. I had to use the following libraries:

-   whisper_rs
-   cpal (for audio input)
-   rubato (for resampling)
-   hound (for writing wav files -- this won't be needed in the final app)

Now to put it all together.

Streaming audio input works. Need to convert to the right sample rate though.

OK, now it's translated several utterances.

It needs a fair bit of refinement before it's tidy and fast.

## 2023-05-08

Trying to stream data into the UI. Nearly there. Just struggling with closing the sender channel early even though it seems to be held by a closure.

Error is

    thread '<unnamed>' panicked at 'Failed to receive data: receiving on a closed channel', src\whisper.rs:65:33

Reading [Chapter 13: closures](https://doc.rust-lang.org/book/ch13-01-closures.html#capturing-references-or-moving-ownership).

OK, all that code is ok. the problem is actually that I am dropping `stream` which holds the last channel sender.

So I need to keep that somewhere. Return it.

Ah, that's better. The app now works.

Need to add newlines between each utterance, and make the textbox bigger.

OK, made a few more fixes:

-   whisper in a separate thread (mostly just so that the UI updates while a transcribe is running, can take several seconds)
-   show recording level
-   hide noisy outputs like `[BLANK AUDIO]` and `(crickets chirping)`
-   add a newline between each utterance
-   the textbox is now scrollable
-   you can now choose your microphone source
-   dual-channel microphones now work
-   the default level is 0.05 instead of 0 so that being quiet doesn't trigger a transcription

CPU usage is basically zero unless a transcription is running.

I'm testing with the base model.

It'd be nice to report transcription speed and also for the user to be able to
set the beam size (I'm using 8 at the moment, it seems to take a few seconds to
transcribe). Higher will be more accurate but also slower. 1 was pretty fast but
terribly inaccurate. A beam size of 0 could just be the Greedy algorithm (which
is the default in the demo).

I think it's time to post this.

Actually, one more thing. I need to make it download models if necessary, and automatically cache models in the users' home.

This is tricky because the only way I know to convert the model from the huggingface download is with a python script. Ew.

I think it makes the most sense to release them both together.

## 2023-05-15

https://lib.rs/crates/wasapi can do loopback audio recording. cpal doesn't.

It's a pretty horrible interface though. The example shows how loopback works. I guess I'll try it.

https://github.com/HEnquist/wasapi-rs/blob/master/examples/loopback.rs is the
example. It's not actually capturing output audio though. It points at the
[windows WASAPI docs](https://learn.microsoft.com/en-us/windows/win32/coreaudio/core-audio-interfaces)
anyway. Read those. And [about loopback specifically](https://learn.microsoft.com/en-us/windows/win32/coreaudio/loopback-recording).

> To open a stream in loopback mode, the client must:

> Obtain an IMMDevice interface for the rendering endpoint device.
> Initialize a capture stream in loopback mode on the rendering endpoint device.

Derp. After looking through various docs and the cpal code, I found that the place where I would add WASAPI loopback in CPAL is already configured for it! https://github.com/RustAudio/cpal/commit/78e84521507a3aa4760ec81ac62943165f5218cd . I just need to treat an output device as input.

Struggling with this a bit. My test is getting `StreamTypeNotSupported`. There have been several pull requests about this as part of cpal. Here's some example code for using loopback:

https://github.com/RustAudio/cpal/pull/478

## 2023-05-16

OK, got WASAPI loopback working last night. The user (me) can now choose any input or output for recording, so I can avoid missing what someone said in a meeting.

I think the last thing is to add better Voice Activity Detection. My earlier research uncovered Silero, which unfortunately doesn't have rust bindings. My earlier research also uncovered Rust-onnx, which is a common format for ML models. It turns out there is an [onnx release](https://github.com/snakers4/silero-models#onnx) of silero. Hooray!

OMG. Silero has an absolutely amazing [Text To Speech model](https://thegradient.pub/towards-an-imagenet-moment-for-speech-to-text/). It's beautiful.

```py
import torch

device = torch.device('cuda')
torch.set_num_threads(4)

model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language='en', speaker='v3_en')
model.to(device)
path = model.save_wav(text="Hello world!", speaker='random', sample_rate=48000)
```

Ugh that sounds bad. Wow, the v3 model is heaps better. 55MB. The first voice is great. The random voices are terrible.

```py
# V3
import os
import torch

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'v3_en.pt'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/en/v3_en.pt',
                                   local_file)

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)

example_text = '''Crab bucket, thought Glenda as they hurried towards the Night Kitchen. That's how it works. People from the Sisters disapproving when a girl takes the trolley bus. That's crab bucket. Practically everything my mum ever told me, that's crab bucket. Practically everything I've ever told Juliet, that's crab bucket, too. Maybe it's just another word for the Shove. It's so nice and warm on the inside that you forget that there's an outside. The worst of it is, the crab that mostly keeps you down is you... The realization had her mind on fire.'''
sample_rate = 48000
speaker='en_0'
model.save_wav(text=example_text, sample_rate=sample_rate, speaker=speaker)
```

Ah, much better. And much faster. Generates the above in about 4 seconds. 8 cpu threads is 2 seconds. Not bad.

## 2023-05-21

Porting whisper.cpp to pure rust. Loading the model is working. The functions I'm currently calling from mubbles are

```
whisper_init_state
whisper_full_with_state
```

That's it! Let's have a quick read through of the code to see what I'm in for. The first one looks pretty simple. The second one is [long](https://github.com/ggerganov/whisper.cpp/blob/fd01209d0923de14bd0640eb4e386e937789d063/whisper.cpp#L4065) but noisy.

```
whisper_pcm_to_mel_with_state
whisper_lang_auto_detect_with_state
get_signal_energy
main loop starts
    whisper_encode_internal
    for each temperature
        whisper_decode_internal
        whisper_process_logits
        for each decoder
            whisper_sample_token
        whisper_sequence_score
```

OK, this seems reasonably manageable. I haven't got down to the level of tensors yet.

There's a few references to the original python/pytorch whisper code. I see that there are Rust bindings to torch as well. That would be an easy way to get accelerators running.

Googling about tflops. This computer's Nvidia Geforce 1080 is 8 or so. A current Mac that can run large whisper [in milliseconds](https://github.com/ggerganov/whisper.cpp/issues/89) is 15. So a properly-optimized implementation running on a cuda backend should be good.

https://github.com/Const-me/Whisper is whisper.cpp implemented on DirectCompute to get to graphics cards. it's 3 times faster than the original Pytorch. It also uses [this approach](https://www.researchgate.net/publication/255667085_A_simple_but_efficient_real-time_voice_activity_detection_algorithm) for voice activity detection.

https://github.com/guillaumekln/faster-whisper uses CTranslate2. It's about 4 times faster than the original.

> CTranslate2 is a C++ and Python library for efficient inference with Transformer models. The project implements a custom runtime that applies many performance optimization techniques such as weights quantization, layers fusion, batch reordering, etc., to accelerate and reduce the memory usage of Transformer models on CPU and GPU. 

https://github.com/sanchit-gandhi/whisper-jax claims to be 70x faster on TPUs but idgaf because you can't buy TPUs. It also claims 10x on the original pytorch. It also mentions that the HuggingFace Transformers library implementation of whisper is 3x faster than the original.

Modal [seems like](https://modal.com/docs/guide/llm-voice-chat) a pretty good platform for cloud apps. I wonder how the cost works out.

Trying out https://github.com/metavoicexyz/tortoise-tts . 1.2G model. Sucks at downloading. Demo is impressive though.



