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

## 2023-05-22

Doing some quick perf testing to see why loading is so slow.

Loading the large model takes 28 seconds. Copying the whole file large-v2 file into memory takes about 4 seconds from disk, and copying from one memory buffer to another is a pretty consistent 0.6s. So I should be able to hit that sort of loading time eventually, given that even the large file has very little overhead -- there are only ~1200 tensors so almost all the work is reading values to fill them.

Possibly I could even memory map the file and support very large models. That's probably going to be too slow anyway but fun to consider if the model didn't iterate much.

Breaking down the remaining work some more

```
whisper_pcm_to_mel_with_state
    log_mel_spectrogram
        log_mel_spectrogram_worker_thread
            fft // // Cooley-Tukey FFT. poor man's implementation - use something better
                dft
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

Starting on whisper_full_with_state

## 2023-05-23

https://github.com/MahmoudAshraf97/whisper-diarization looks interesting.
https://github.com/Purfview/whisper-standalone-win is an easy way to run batch
jobs.

https://github.com/LostRuins/koboldcpp too. Oh, and [kobold has a large
collection](https://github.com/KoboldAI/KoboldAI-Client) of llama-based models
for storytelling.

https://github.com/oobabooga/text-generation-webui/ also worth investigating.

done whisper_pcm_to_mel_with_state. wasn't too bad, as expected. copilot is
pretty awesome. 813ms for a 30-second log-mel spectrogram. Once it's all done,
I'll find a rust library for FFT and see if I can speed it up. Or maybe just
BLAS or something.

https://google.github.io/comprehensive-rust/welcome.html

## 2023-05-24

Development continues. Getting into decoding and encoding. This is very interesting.

Whisper.cpp only uses 46 out of the 126 ggml functions in `ggml.h`:

    add blck_size build_forward_expand build_forward conv_1d_1s conv_1d_2s cpy diag_mask_inf_inplace element_size flash_attn flash_ff ftype_to_ggml_type gelu get_data get_rows graph_compute graph_print init mul_mat_str mul_mat mul nbytes nelements new_f32 new_tensor_1d new_tensor_2d new_tensor_3d norm permute repeat reshape_2d reshape_3d scale_inplace set_scratch soft_max_inplace time_init time_us transpose type_name type_size type_sizef type used_mem view_1d view_2d view_3d

Discovered the Numbered Bookmarks extension in VSCode. It's magic, just like
starcraft :joy:.

I've also just discovered that sometimes copilot suggests multiple completions,
but you only see the interface to iterate through them if you mouse over the
suggestion.

Copilot has been truly amazing in the conversion. I can paste in some CPP code,
multicursor-prefix each line with `// CPP:`, and then let copilot complete each
block in Rust. I can even comment out its first attempt and ask for a functional
version of the same code (which I have found so far to be less readable). It was
also great at implementing the sliding window for the energy efficiently
(instead of performing `64` f32 ops per sample, the better algorithm performs `2`).

Thinking about that a bit more, I've just realized that this lighting window
algorithm might accumulate floating point error if there's a big point that then
later gets subtracted and you lose the resolution of the rest. There's
definitely a approach with a kind of binary tree that would not have to do the
subtractions again, and it would use N.logN memory instead of linear memory, but
there'd be no risk of floating point error. Surely there's a library somewhere
that'll do that. Never mind, look later.

I keep getting distracted. Try a five minute sprint. Very effective.

## 2023-05-25

Egui 0.22. Updating all cargo dependencies. The VSCode integration shows which dependencies are out of date inline in the cargo.toml file. Very nice.

Oooh the whisper-rs bindings now support cuda! Update to 0.8. State management interface has changed. Oh well, keep it in the closure.

Ugh, opencl and cuda build errors:

```log
$ cargo build --release
    Blocking waiting for file lock on package cache
    Blocking waiting for file lock on package cache
    Blocking waiting for file lock on package cache
   Compiling whisper-rs-sys v0.6.0
error: failed to run custom build command for `whisper-rs-sys v0.6.0`

Caused by:
  process didn't exit successfully: `C:\Users\J\Source\TARGET\release\build\whisper-rs-sys-d444a7d5701a545e\build-script-build` (exit code: 101)
  --- stdout
  cargo:rustc-link-search=C:\Users\J\Source\TARGET\release\build\whisper-rs-sys-715090dd6c474455\out
  cargo:rustc-link-lib=static=whisper
  cargo:rustc-link-lib=cublas
  cargo:rustc-link-lib=culibos
  cargo:rustc-link-lib=cudart
  cargo:rustc-link-lib=cublasLt
  cargo:rustc-link-search=/usr/local/cuda/lib64
  cargo:rustc-link-search=/opt/cuda/lib64
  cargo:rerun-if-changed=wrapper.h
  cargo:rerun-if-changed=./whisper.cpp\whisper.h
  cargo:rerun-if-changed=C:\Program Files\LLVM\lib\clang\8.0.0\include\stddef.h
  cargo:rerun-if-changed=C:\Program Files\LLVM\lib\clang\8.0.0\include/__stddef_max_align_t.h
  cargo:rerun-if-changed=C:\Program Files\LLVM\lib\clang\8.0.0\include\stdint.h
  cargo:rerun-if-changed=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.34.31933\include\stdint.h
  cargo:rerun-if-changed=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.34.31933\include\vcruntime.h
  cargo:rerun-if-changed=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.34.31933\include\sal.h
  cargo:rerun-if-changed=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.34.31933\include\concurrencysal.h
  cargo:rerun-if-changed=C:\Program Files\LLVM\lib\clang\8.0.0\include\vadefs.h
  cargo:rerun-if-changed=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.34.31933\include\vadefs.h
  cargo:rerun-if-changed=C:\Program Files\LLVM\lib\clang\8.0.0\include\stdbool.h
  -- Building for: Visual Studio 17 2022
  -- Selecting Windows SDK version 10.0.22621.0 to target Windows 10.0.19045.
  -- The C compiler identification is unknown
  -- The CXX compiler identification is MSVC 19.34.31935.0
  -- Detecting CXX compiler ABI info
  -- Detecting CXX compiler ABI info - done
  -- Check for working CXX compiler: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.34.31933/bin/Hostx64/x64/cl.exe - skipped
  -- Detecting CXX compile features
  -- Detecting CXX compile features - done
  -- Configuring incomplete, errors occurred!
  See also "C:/Users/J/.cargo/registry/src/github.com-1ecc6299db9ec823/whisper-rs-sys-0.6.0/whisper.cpp/build/CMakeFiles/CMakeOutput.log".
  See also "C:/Users/J/.cargo/registry/src/github.com-1ecc6299db9ec823/whisper-rs-sys-0.6.0/whisper.cpp/build/CMakeFiles/CMakeError.log".

  --- stderr
  CMake Error at CMakeLists.txt:3 (project):
    No CMAKE_C_COMPILER could be found.



  thread 'main' panicked at 'Failed to run `cmake`', C:\Users\J\.cargo\registry\src\github.com-1ecc6299db9ec823\whisper-rs-sys-0.6.0\build.rs:109:9
  note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
```

The way to fix this is to ask Copilot to complete the following sentence:

    The way to fix this is to

And it does so! It says

     install the [Visual Studio 2022 C++ Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022-preview).

That didn't help. Those are already installed. Searching google finds [a better answer](https://stackoverflow.com/a/48400271/412529):

    Adding the Windows 10 SDKs to Visual Studio 14 2015 solved the problem for me.

It seems I don't have an SDK installed, so use the visual studio installer to add the latest one. 426MB download, 2.5GB disk. Ok, I can wait.

OK, adding the windows SDK fixed the `No CMAKE_C_COMPILER could be found.`. Stackoverflow beats copilot this time.

Next error is `No CUDA toolset found.`. Fix this by...

    installing the CUDA toolkit from https://developer.nvidia.com/cuda-downloads. 2.5GB download, 5.5GB disk. Ok, I can wait.

Nah, I just did this. [Stackoverflow](https://stackoverflow.com/a/68120870/412529) again.

> Copy files to `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Microsoft\VC\v160\BuildCustomizations`
>
> The four files are:
>
>     CUDA 11.3.props
>     CUDA 11.3.targets
>     CUDA 11.3.xml
>     Nvda.Build.CudaTasks.v11.3.dll

Ugh. Search my disk for `Nvda.Build.CudaTasks.`, I just installed it a couple of weeks ago...

    et | rg Nvda.Build.CudaTasks.
    cargo install erdtree
        Compiling erdtree v2.0.0
        error[E0554]: `#![feature]` may not be used on the stable release channel
    # use the nightly channel to build
    cargo +nightly install erdtree
    cd /c/
    erd --pattern Nvda.Build.CudaTasks.*

Ah, it's installed for V140:

    │              ┌─ Nvda.Build.CudaTasks.v10.2.dll
    │           ┌─ BuildCustomizations
    │           │     ┌─ Nvda.Build.CudaTasks.v10.2.dll
    │           │  ┌─ BuildCustomizations
    │           ├─ V140
    │        ┌─ v4.0
    │     ┌─ Microsoft.Cpp
    │  ┌─ MSBuild
    ├─ Program Files (x86)
    \\?\C:\

Remove V140 and install cuda again. Ah, uninstall Visual Studio Code 2015. That
took ages. Update VSCommunity 2022. Try build again. `No CUDA toolset found.`
Installs to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\` .

Now error is `Cannot find source file: ggml-cuda.cu`.

It's still using whisper-rs-sys-0.6.0. SHould be 0.8 ? Try `cargo clean`, wait three minutes for a full build.

Ah, looks like they forgot something. They've "included" [some files](https://github.com/tazz4843/whisper-rs/commit/31845bbe942e3c7e453ef9e46fee53b798f05bcb) in the crate, but missed the .cu files. Patch. OK, that's fixed. Next error:

    cannot open input file 'cublas.lib'

This is starting to get frustrating.

    erd --pattern cublas.lib

It's definitely there, at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64\cublas.lib`. Check linker command. Ah, the libpaths are:

    "/LIBPATH:C:\\Users\\J\\Source\\TARGET\\release\\deps"
    "/LIBPATH:C:\\Users\\J\\.cargo\\registry\\src\\github.com-1ecc6299db9ec823\\windows_x86_64_msvc-0.42.2\\lib"
    "/LIBPATH:C:\\Users\\J\\.cargo\\registry\\src\\github.com-1ecc6299db9ec823\\windows_x86_64_msvc-0.48.0\\lib"
    "/LIBPATH:C:\\Users\\J\\.cargo\\registry\\src\\github.com-1ecc6299db9ec823\\windows_x86_64_msvc-0.34.0\\lib"
    "/LIBPATH:C:\\Users\\J\\Source\\TARGET\\release\\build\\whisper-rs-sys-945773be75e1cb0b\\out"
    "/LIBPATH:/usr/local/cuda/lib64"
    "/LIBPATH:/opt/cuda/lib64"
    "/LIBPATH:C:\\Users\\J\\.rustup\\toolchains\\stable-x86_64-pc-windows-msvc\\lib\\rustlib\\x86_64-pc-windows-msvc\\lib"
    "/LIBPATH:C:\\Users\\J\\.rustup\\toolchains\\stable-x86_64-pc-windows-msvc\\lib\\rustlib\\x86_64-pc-windows-msvc\\lib"

So nothing pointing at the cuda lib folder. The usr local looks weird too, I bet that's coming from whisper-rs. [Yup](https://github.com/jnnnnn/whisper-rs/blob/805dfcf36a19dbc20ac4595ea13595cf88dd6207/sys/build.rs#L40).

Disable sccache. Pointing all cargo builds at the same target directory with `CARGO_TARGET_DIR` is faster and results in less duplication everywhere.

Oh bugger, updating the patch just means `cargo update`. Oh well, I now know about `~/.cargo` caching lots of stuff.

Ok, I didn't push my patch branch. Delete checkout. OK, that's better.

Use local filesystem anyway. Next error:

    fatal error LNK1181: cannot open input file 'culibos.lib

Ugh. Comes from https://github.com/jnnnnn/whisper-rs/blob/b9079343c034a95fe54417b41864a246bced9b15/sys/build.rs#L37 . Try commenting it out.

Ok, fixed properly. [PR](https://github.com/tazz4843/whisper-rs/pull/60) for whisper-rs.

Wow, even the large model runs in about 8s on this Nvidia GeForce 1080. Nice.

The small model runs a 30s full transcription in about 3s with a beam size of 5. Good enough, and it uses way less GPU so I can still do other things. Dropping the beam size down to 2 takes it down to 2s.

I can speak in french or spanish and it translates to english. I'm impressed.

Try quantized model? https://github.com/ggerganov/whisper.cpp#quantization and the following models still have a Word Error Rate of below

```sh
winget install GnuWin32.Make
cd whisper.cpp
# this command builds without cuda
cmake --build .
# this command sets WHISPER_CUBLAS=1 to build with cuda
cmake --build . --config Release --target whisper -- WHISPER_CUBLAS=1

# build all in visual studio
./bin/release/quantize ~/.cache/whisper/small.bin ~/.cache/whisper/small-q5_0.bin q5_0
```

Compare results

```sh
ffmpeg -i "The TV Screen.mp3" -ar 16000 -ac 1 -c:a pcm_s16le tvscreen.wav
./main -m small.bin tvscreen.wav  > small.txt 2>&1
./main -m small-q5_0.bin tvscreen.wav  > small-q5_0.txt 2>&1
```

Both take about 40 seconds. The q5 one is noticably worse in transcription quality. It also uses 447MB instead of 743MB.

Conclusion: This code is not well optimized for BLAS. The developer is clearly focusing more on CoreML optimization.

In particular, the encoder is cuda, but the decoder is not, and the decoder is half the runtime:

```
# small model timings

  load time =  1275.67 ms
  fallbacks =   1 p /   0 h
   mel time =  1729.96 ms
sample time =   829.62 ms /   971 runs (    0.85 ms per run)
encode time = 14815.59 ms /     9 runs ( 1646.18 ms per run)
decode time = 20371.54 ms /   970 runs (   21.00 ms per run)
 total time = 39081.51 ms
```

Also it's only using half my CPU and barely any GPU, an average of about 10%.

We can definitely do better than this. Continue rust rewrite, and plan for some sort of accelerated backend, as explored above.

```
# medium model timings

  load time =   876.60 ms
  fallbacks =   2 p /   0 h
   mel time =  1739.46 ms
sample time =   835.96 ms /   980 runs (    0.85 ms per run)
encode time = 16327.36 ms /    10 runs ( 1632.74 ms per run)
decode time = 16713.51 ms /   978 runs (   17.09 ms per run)
 total time = 36558.70 ms
```

Interestingly, the gg_mat_mul benchmark is wicked fast:

```
  64 x   64: Q4_0     1.3 GFLOPS (128 runs) | Q4_1     1.3 GFLOPS (128 runs)
  64 x   64: Q5_0     1.3 GFLOPS (128 runs) | Q5_1     1.4 GFLOPS (128 runs) | Q8_0     1.4 GFLOPS (128 runs)
  64 x   64: F16      1.3 GFLOPS (128 runs) | F32      1.3 GFLOPS (128 runs)
 128 x  128: Q4_0    10.9 GFLOPS (128 runs) | Q4_1    10.9 GFLOPS (128 runs)
 128 x  128: Q5_0     9.3 GFLOPS (128 runs) | Q5_1    10.7 GFLOPS (128 runs) | Q8_0    10.5 GFLOPS (128 runs)
 128 x  128: F16     10.5 GFLOPS (128 runs) | F32     11.0 GFLOPS (128 runs)
 256 x  256: Q4_0    84.7 GFLOPS (128 runs) | Q4_1    85.8 GFLOPS (128 runs)
 256 x  256: Q5_0    84.0 GFLOPS (128 runs) | Q5_1    84.0 GFLOPS (128 runs) | Q8_0    85.8 GFLOPS (128 runs)
 256 x  256: F16     83.7 GFLOPS (128 runs) | F32     68.2 GFLOPS (128 runs)
 512 x  512: Q4_0   365.5 GFLOPS (128 runs) | Q4_1   412.9 GFLOPS (128 runs)
 512 x  512: Q5_0   387.7 GFLOPS (128 runs) | Q5_1   412.1 GFLOPS (128 runs) | Q8_0   337.9 GFLOPS (128 runs)
 512 x  512: F16    381.0 GFLOPS (128 runs) | F32    353.7 GFLOPS (128 runs)
1024 x 1024: Q4_0  1211.4 GFLOPS (128 runs) | Q4_1  1106.4 GFLOPS (128 runs)
1024 x 1024: Q5_0  1244.4 GFLOPS (128 runs) | Q5_1  1149.9 GFLOPS (128 runs) | Q8_0  1166.5 GFLOPS (128 runs)
1024 x 1024: F16   1094.8 GFLOPS (128 runs) | F32    719.8 GFLOPS (128 runs)
2048 x 2048: Q4_0  2268.0 GFLOPS (128 runs) | Q4_1  2266.9 GFLOPS (128 runs)
2048 x 2048: Q5_0  2378.0 GFLOPS (128 runs) | Q5_1  2325.3 GFLOPS (128 runs) | Q8_0  2263.9 GFLOPS (128 runs)
2048 x 2048: F16   1991.8 GFLOPS (116 runs) | F32   1900.0 GFLOPS (111 runs)
4096 x 4096: Q4_0  3744.3 GFLOPS ( 28 runs) | Q4_1  3734.2 GFLOPS ( 28 runs)
4096 x 4096: Q5_0  3704.3 GFLOPS ( 27 runs) | Q5_1  3677.2 GFLOPS ( 27 runs) | Q8_0  3636.7 GFLOPS ( 27 runs)
4096 x 4096: F16   2527.6 GFLOPS ( 19 runs) | F32   3157.8 GFLOPS ( 23 runs)
```

And there's no explanation of how F32 is faster than F16! I wonder if that's the
Nvidia fuckery about hobbling consumer-grade hardware to sell more datacenter
shit.

And there's no explanation of why it's so far away from [the theoretical
limit](https://www.techpowerup.com/gpu-specs/geforce-gtx-1080.c2839) of 8873
GFLOPS. It's only getting to 35% of the theoretical limit. Ugh.

Well, hopefully cublas does better. Gotta finish coding first though.

Wow, the 4090 [can do](https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889) 82 TFLOPS.
That's pretty amazing. I guess memory bandwidth is going to be the bottleneck.
