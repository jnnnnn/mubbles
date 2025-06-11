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

## 2023-05-26

The medium model with a Greedy search seems to work really well. I'm going to keep implementing the pure rust version now.

## 2023-05-30

Got right into the guts of ggml.c today. It turns out that computations are queued up into a graph, and then a rudimentary scheduler executes them all.

Basically the sort of thing that tokio excels at?

So now I'm wondering what approach to take.

Do I:

1. convert ggml graph to imperative Rust, and not bother building a complete graph at any point
2. convert ggml graph to Rust graph, and have to write an evaluation engine for it

I think 2 is the way to go if I'm going to want to implement accelerators.

I think 1 is probably simpler. But actually that might not be true. I might just do a full translation of the ggml stuff -- the scheduler etc. is not very complicated.

Owning the tensors in the graph is going to be a nightmare. I think I'll have the graph own them for now? Nah, that probably won't work. Ugh.

Ah, I understand why GGML is not using much GPU now. Most of the operations are not accelerated, only matrix multiplications. There are nearly fifty other ops.

Additionally, for each accelerated operation, the memory is copied to the GPU, the operation is performed, and then the memory is copied back. That's a lot of copying.

I mean, it works as a first pass but it's a long way from optimal.

I think it might be time to try the onnx implementation, it seems like that's what I'm going to end up implementing anyway.

Upon further reading, it's not much better. https://github.com/zhuzilin/whisper-openvino reports about a 2x speedup over the original Pytorch implementation. Fuck.

Now I am starting to understand all that stuff about tensorflow-lite exports. Google had the right idea all along. I should have bloody known, they invented this field.

Read openai's tiktoken. It's [in rust](https://github.com/openai/tiktoken/blob/main/src/lib.rs#L184)! And python.

Continue with rust conversion. Use approach 2, ggml graph. Fun. I need to write
some actual tests, I'm never going to be able to get it all right first try. At
least the compiler will stop me doing anything really stupid.

### Implementing ggml_graph_compute

OK, there's a fair bit to do here.

1. Work out how memory allocation is going to work -- where to put the tensor elements?
2. figure out what's going on with the various `use_buf` of the state
3. transate the various ggml_mul and so on to build the graph
4. Translate ggml_build_forward_expand
5. Translate ggml_graph_compute

I think I'll do the last one first, just to see what data structures fall out of the logic. Maybe that will help with memory usage.

## 2023-06-02

I've had chatgpt have a few attempts at the `ggml_cgraph` and related structs. I
can't find a neat way to get around Rust's single-mutable-reference rule. I
think the problem is really that ownership is implicit in the CPP and so the
Rust translation doesn't really have a way forward. I can't see how to make
parts of the graph mutable and have the rust compiler understand that it's safe.

The first step is probably to make the graph immutable, and then see if I can
make it mutable again. I think I'll have to use interior mutability.

That means I'll have to use `RefCell` and `Rc` and `Cell` and `Ref` and `RefMut`. I'm not sure how to use them yet.

After some reading, here are the definitions and typical usages:

### RefCell

A mutable memory location with dynamically checked borrow rules. Pretty similar to a `Box` but with runtime borrow checking.

```rust
let x = RefCell::new(vec![1, 2, 3, 4]);
let y = x.borrow();
let z = x.borrow_mut(); // panic
```

A RefCell is a single-threaded RwLock (not mutex). It's not thread safe.

### Rc

A reference-counted pointer. It's not thread safe. It's like a `Box` but with reference counting.

```rust
let x = Rc::new(vec![1, 2, 3, 4]);
let y = x.clone();
let z = x.clone();
```

### Cell

A mutable memory location with statically checked borrow rules. It's like a `Box` but with compile-time borrow checking.

```rust

```

### Internet advice

[src](https://www.reddit.com/r/rust/comments/11ie1n9/comment/jayegyh/?utm_source=share&utm_medium=web2x&context=3)

> Trees and more general graphs are a classic case where reference-counting is a good idea. It may still be not the best idea, mind you. If you're trying to write a performant data structure, you should carefully think about memory allocations, when you want to free memory and how you'd want to synchronize multithreaded accesses. Perhaps you'd like to go with immutable data structures and use `Rc` without inner `RefCell`. Perhaps you'd want to use `Box` instead of `Rc`, if you don't plan to clone around your subgraphs. Perhaps you'd want to allocate all nodes in an arena, to amortize memory allocation costs, or maybe you would even use a garbage collector. There are many options, but `Rc<RefCell<T>>` is certainly a valid possibility.

Ah, look into arenas. Ah, [here](https://lib.rs/memory-management) we go:

1. bumpalo has 2M downloads, looks promising. support for `Box` allocations in a bump arena.
2. typed_arena is a bit more barebones, it only supports allocating a single type of object. All objects have the same lifetime so that simplifies cyclic dependencies.

https://github.com/gfx-rs/wgpu

> wgpu is a cross-platform, safe, pure-rust graphics api. It runs natively on Vulkan, Metal, D3D12, D3D11, and OpenGLES; and on top of WebGPU on wasm.
>
> The api is based on the WebGPU standard. It serves as the core of the WebGPU integration in Firefox, Servo, and Deno.

Here's [some good advice](https://manishearth.github.io/blog/2015/05/27/wrapper-types-in-rust-choosing-your-guarantees/#composition) about memory abstractions:

> When choosing a composed type, we must do the reverse; figure out which guarantees we want, and at which point of the composition we need them. For example, if there is a choice between Vec<RefCell<T>> and RefCell<Vec<T>>, we should figure out the tradeoffs as done above and pick one.

### Implementing forward expand

I'm going to try to implement `ggml_build_forward_expand` first, and skip all the grad stuff. That should simplify things a fair bit.

Each tensor can just own its own Vec. To figure out mutability, I'm going to walk through the code and see what it mutates. Seems obvious now that it's occurred to me.

OK, walking is pretty simple. Here's the algorithm:

```py
# each node is a tensor with a vec of elements and a shape, and pointers to other tensors (src0, src1, opt[0..3])
for each node in graph:
    ggml_compute_forward(INIT)
for each node in graph:
    ggml_compute_forward(COMPUTE)
for each node in graph:
    ggml_compute_forward(FINALIZE)
```

I think the only data that gets modified is the `data` field of each tensor. So I can just make that a `RefCell<Vec<f32>>` and then I can mutate it. However, I've just read some advice saying it's actually better to use the type system as much as possible -- if something is only meant to be called once, have the method return a different type. Applying this pattern, Copilot suggests:

1. a `ggml_compute_forward` that returns a `ggml_compute_forward_compute` that returns a `ggml_compute_forward_finalize`
2. a tensor that has a `data` field that is something like

    ```rust
    enum TensorData {
        Uncomputed(RefCell<Vec<f32>>),
        Computed(Vec<f32>),
    }
    ```

I would pass a mutable graph to the compute method. This would allow me to mutate the nodes of the graph. Except I could not construct such a graph, because the nodes would have immutable references to other nodes, which the borrow checker wouldn't allow. Hmm. I think I'll have to use interior mutability. Google some more first.
[interconnected futures](https://stackoverflow.com/questions/70646911/a-collection-of-interconnected-futures-in-async-rust-w-tokio)?
[mutable directed graph](https://users.rust-lang.org/t/appropriate-way-to-model-mutable-directed-graph-for-realtime-dataflow-computation/39830/2)?

> The standard suggestion for graph structures is to use indexes into a vector instead of references.

Huh, that's pretty much what the first one recommends as well. Can't see how
that would work in this case. Maybe it's best to just use an RWLock, there are
only 1200 tensors in the largest Whisper model. Although I guess the graph could
apply the same tensors over and over..

Let's keep it simple for the first prototype and sort out the rest later. Everything's behind RWLocks.

### trying whisper-faster

    .\whisper-faster.exe --compute_type int8 --word_timestamps True --model small --model_dir c:\users\j\.cache\whisper "G:\videos\Captures\Zoom Meeting 2023-03-07 14-35-55.mp4"
    RuntimeError: Library cublas64_11.dll is not found or cannot be loaded

argh, I've got cuda 12 installed as the default, but this python shit can't find 11.

    CUDA_PATH_V11_1=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1

has the dll this thing wants. Maybe I'll add the bin folder to the path. in a cmd shell:

```cmd
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\
```

Ugh setting paths doesn't work either. https://stackoverflow.com/a/64303856/412529 -- python 3.8 changed to use add_dll_directory as a security fix.

### back to prs

It's probably also possible to not allocate so much memory all the time. the whole graph being allocated at once is almost certainly unnecessary -- we can free earlier results that are already incorporated into later nodes and won't be used again, and we don't need to allocate data arrays for later nodes until it's time to write to them.

Yeah GGML_MAX_NODES = 4096 so RWlocks are fine. 4096 write locks is not a performance problem.

How to store references to the other tensors? src0 and src1 are references to previous tensors in the graph. Let's just go with Rc for now? Arc? `Arc<RwLock<Tensor>>`?

### dfdx

https://github.com/coreylowman/dfdx

aha. This is what I would have eventually ended up writing! I think I should switch to using this instead of ggml. I wonder if there is already a whisper implementation using this.

There's a llama! but no whisper.

fun.

base it on the original pytorch code I think.

## 2023-06-04

https://github.com/sonos/tract is a simple pure-rust Tensorflow/onnx inference engine. Interesting docs. They store their nodes in a vec and (following the same pattern as was recommended above) reference them from the graph structure [using indices](https://github.com/sonos/tract/blob/main/doc/graph.md).

The pytorch whisper is not very long but I'm finding it hard to read. I think the next step is to do a few pytorch tutorials.

## 2023-07-11

Update deps.

## 2023-08-09

[Candle](https://github.com/huggingface/candle/tree/main/candle-examples/examples/whisper) Time! No more cmake. yay.

## 2023-08-21

Experimenting with candle whisper. It's fast as. cudnn helps too. plus tracer makes it easier to optimize. The example transcribes (tiny.en, JFK example) in 0.6s on my GTX1080. Downloading small and medium models now.

https://github.com/Lisoveliy/StarCoderEx + StarCoder: LLM specialized to code generation. from https://github.com/huggingface/candle = local copilot?

## 2023-11-05

Looking at duplicated words problems.

https://github.com/m-bain/whisperX looks pretty fun. ctranslate2 backend. claims 90x realtime with large-v2 (!!!). Ah, that's with a Quadro 8000.
130 TFlops for tensors, 48GB / 672 GB/s memory; 4608 cuda cores and 576 tensor cores. whereas my GTX1080 has 8GB / 320 GB/s,

quadro specs: https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/quadro-rtx-8000-us-nvidia-946977-r1-web.pdf

https://www.microway.com/knowledge-center-articles/comparison-of-nvidia-geforce-gpus-and-nvidia-tesla-gpus/ says that 1080TI gets 0.177 tensor TFlops whereas Quadro 8000 gets 32.6 tensor Tflops. That's 184x faster. The 1080 is so hobbled! Here's a table:

| Category           | GTX1080  | Quadro   | Pixel 7 (Tensor G2 / EdgeTPU) |
| ------------------ | -------- | -------- | ----------------------------- |
| FP16 Tensor TFlops | 0.177    | 32.6     | 4.0                           |
| Memory             | 8GB      | 48GB     | 3GB                           |
| Memory Bandwidth   | 320 GB/s | 672 GB/s |
| CUDA Cores         | 2560     | 4608     | -                             |
| Tensor Cores       | 0        | 576      |

Testing between the python faster-whisper and the candle-examples whisper shows that the python one is ~10x faster for the `medium` model. The candle one also misses a few words at the start or end of each segment. Candle takes 12s to transcribe each 30s segment, whereas faster-whisper is about 0.7s. Faster-whisper's transcription is also much more accurate.

So, work to do on candle:

-   implement Silero Voice Activity Detection as a model in Candle -- see https://github.com/openai/whisper/discussions/29#discussioncomment-3726710. this requires about a week's work. This will allow us to split the audio into segments and transcribe each segment separately, resulting in higher accuracy and more responsiveness.
-   Tune the various settings. https://github.com/ggerganov/whisper.cpp/pull/291 . Each improvement is probably a few days work, and there are several: beam search, temperature adjustment, failure detection, repetition detection, context clearing (note that faster-whisper doesn't use previous context because of its improved VAD, and the maximum context is ~250 tokens (100 words / 5 lines) anyway, so it's not very useful).

## 2023-11-08

Experimented with faster-whisper this evening. It's great. Even the large model can transcribe short utterances in <1s on my GTX1080. Plus it's basically zero-config -- it downloads models and picks up Cuda automatically.

Haven't figured out loopback yet, so I can only record the microphone for now.

## 2025-04-28

picking up candle again ? Maybe it's better now. From memory I was having compilation errors last time.

I've been using python faster-whisper for a year or two and it's been good. But I want a better UI.

https://crates.io/crates/earshot (filters out silence) or silero-vad-rs (filters out non-speech, but requires gpu memory)

model cached at

```
C:\Users\J\.cache\huggingface\hub\models--distil-whisper--distil-large-v3.5\blobs\76ec9f754fc4b4810845dc36b71d1897c1342e702810c179e1569690084cfb0c.part
```

## 2025-05-02

mel spectrogram generation using the rust library rufft is probably about 5x the speed of Whisper's implementation `pcm_to_mel` on my cpu:

Custom mel spectrogram took: 45.7557ms
Candle mel spectrogram took: 242.6701ms

But I've gotta get it outputting the same thing (or close enough). The AIs haven't been able to get it right yet despite inputting the actual whisper code.

Maybe I can just copy the candle code and ask the AI to replace the FFT with the fast one.

Timings

```log
Finished `release` profile [optimized] target(s) in 21.83s
Running `target\release\mel2.exe`
stft shape: [257, 2998]
filters shape: [80, 257]
result shape: [80, 2998]
Custom mel spectrogram took: 47.0296ms
Candle mel spectrogram took: 212.6067ms
Candle rufft mel spectrogram took: 456.6036ms
```

So custom is way faster but doesn't look quite right.

The unrolled ggml one candle rufft (not using rufft yet) is hardcoded to 2 threads which is why it is slower than the normal Candle one.

factor implementation out to separate file so copilot can work on it faster

ok, with this implementation, we're 30% faster already.

warning: `mel2` (bin "mel2") generated 5 warnings (run `cargo fix --bin "mel2"` to apply 1 suggestion)
Finished `release` profile [optimized] target(s) in 13.16s
Running `target\release\mel2.exe`
stft shape: [257, 2998]
filters shape: [80, 257]
result shape: [80, 2998]
Custom mel spectrogram took: 25.7204ms
Candle mel spectrogram took: 201.4842ms
Candle rufft mel spectrogram took: 171.4573ms

Removed T:Float and now just f32. Much faster again:

     Running `target\release\mel2.exe`

stft shape: [257, 2998]
filters shape: [80, 257]
result shape: [80, 2998]
Custom mel spectrogram took: 48.9897ms
Candle mel spectrogram took: 224.9054ms
Candle rufft mel spectrogram took: 90.3094ms

Oh there's actually two rust fft libraries. ChatGPT says:

-   Use rufft if you want FFT acceleration on a GPU.
-   Use rustfft if you want simple, portable, CPU-side FFTs in pure Rust.

Even the simple cpu one is WAY faster than the ggml version. I will get it to try rufft.

Uh, chatgpt is wrong. rufft is just immature, it's not GPU. Stick with RustFFT.

Continue optimizing.

Run `cargo flamegraph`. Requires admin on windows. Whatever, do that.

Interesting. Biggest time sinks:

-   log*mel_spectrogram*() -- 86% of samples
    -   log_mel_spectrogram_w() -- 82% of samples
        -   fft() -- 37% of samples
        -   alloc Complex 9%
        -   ??? -- 82 - 47 = 35%

So the fft is only taking up a third of the function time. The rest is spent on ??

Alright, whatever. Allocations, hanning windows, whatever. Clean up later.

Focus on how to compute incrementally as audio becomes available, because that's really what I'm trying to optimize.

I need to keep my own buffers, of input and mel (and any necessary intermediate, like scratch for rustfft).

Then I wonder how to implement fft just for a small part of the audio buffer that has just been populated, continuing on from the previous.

Reading the simplified code, I get it. Each frame (column in the mel) is independent -- it is the fft of step_size samples.

Mel parameters are:

Started writing an incremental Mel class but I think the best thing is probably to get the full thing displaying first.

## 2025-05-05

looks like the range isn't quite right. Mapping the mel from -1..1 to 0..255 grayscale makes the black parts of the image gray. Maybe candle's implementation has a bug? Or maybe that's what openai spectra look like as well and they just didn't get the normalization right. todo: check openai spectra.

Next steps: 0. just for fun, overlay previous transcript on mel

1. separate out audio capture from whisper model.
2. add speech detector so that segments are neater. copy from python.
3. add second transcription stream for partials
4. incrementally generate mel and stream it to screen in real time
5. add partial transcription overlay on top of mel

tried to overlay. candle-whisper's per-word timestamping is not implemented.
looked at how openai does it and it's a couple of pages of python. I guess I'll
see if copilot can translate it, I can't face doing it myself.

I think I do need to pull the audio stuff out first. let's do that.

## 2025-05-06

Done!

Next thing: per-word timestamps.

Oh, candle already has support for Silero voice activity detection. It's not much code either. Nice.

Ooh and it has qwen as well so I could do the summarization bit in the same app...

https://github.com/linto-ai/whisper-timestamped is some further work on word alignment with whisper.

wow ok that's 2400LOC. I am not that dedicated. It talks a lot about the naive approach -- run a full pass once to get the text, then a second pass to get the timestamps. Not sure how this works exactly.

But I can use the large model to get the text, and then a tiny model to do alignment. Since I'm going to need the tiny model loaded anyway to compute partials.

I wonder what happens if I just don't feed in the whole fully-padded mel to the model. Does it still work? Is it faster?

### word level timestamps

Copy implementation from original whisper python, get copilot to translate as much as possible automatically.

Ooh https://github.com/nyrahealth/CrisperWhisper/blob/develop/README1.md is interesting, it's aiming to be a more exact transcription including pauses, false starts, and so on. It's more accurate?

huh, https://github.com/nyrahealth/CrisperWhisper/blob/develop/run_experiments/visualize_timestamped_transcripts.py is almost exactly what I'm aiming for.

https://huggingface.co/spaces/Amrrs/openai-whisper-live-transcribe doesn't work ?

https://github.com/ufal/whisper_streaming

> we consecutively process new audio chunks, emit the transcripts that are confirmed by 2 iterations, and scroll the audio processing buffer on a timestamp of a confirmed complete sentence. The processing audio buffer is not too long and the processing is fast.
>
> In more detail: we use the init prompt, we handle the inaccurate timestamps, we re-process confirmed sentence prefixes and skip them, making sure they don't overlap, and we limit the processing buffer window.

## 2025-05-12

Added a status display, pretty cool. Mels are real fast now.

OK, let's try to get this more accurate.

Split the 30-second window into three. The first ten seconds is context; the second 10 seconds' transcript is saved, and the last ten seconds is context for the next lot. No, can just split in half?

Save as soon as two transcripts overlap enough?

Look at whisper's decoding.py. It feeds in previous tokens (see `_get_initial_tokens`). The candle model in whisper.rs and model.rs (`TextDecoder`) doesn't do that. Let's change that.

## 2025-05-13

feeding in previous tokens seems to make it go crazy. all sorts of repetition. possibly more trouble than its worth. the prev text token and SOT token have to be in the right spots.

I think getting timestamps working will make segmenting much better.

## 2025-05-18

https://github.com/Zackriya-Solutions/meeting-minutes is not bad but:

-   tauri frontend -- js / python build process

recommends LLMs above 32B for summarization as otherwise hallucinate

## 2025-05-24

Ah, I now understand what a 'segment' is in the original openai code:

The encoder takes mel frames and produces audio tokens (ints). Each audio token represents two mel frames, or 0.02s.

The decoder takes audio tokens and produces text tokens. Text tokens are decoded into strings, with the tokenizer converting single or paired text tokens into particular strings.

Some text tokens represent timestamps instead of output text. The whisper model takes a 30-second mel audio (1500 mel frames) and produces several segments of text tokens from it. This represents distinct utterances. These segments are separated by timestamp tokens in the stream of text tokens output.

The openai code returns these segments separately; the candle whisper model just ignores the timestamp tokens, as the heuristics to separate segments are a little spicy.

I think a VAD system would be better; doing the segmentation before feeding the audio into Whisper means I don't have to implement the heuristics. It also makes things more responsive and more efficient. If I rely on Whisper for segmentation, I will have to process some audio multiple times.

The downside is having to incorporate a VAD into my application, making it larger and slower to build. But pretty much every whisper implementation I have seen makes this decision so it seems obvious.

The other major advantage of adding a VAD is that it is cheap to run continuously, even if there may not be speech. This makes listening continuously much more efficient, as non-speech audio will not be processed through whisper at all.

The downside of a VAD is slightly more memory usage. But I guess I could unload the whisper model after a few seconds of silence and that would mean far lower memory usage while the VAD is not detecting speech.

It is frustrating that both the openai and the candle implementations make no distinction between the different types of tokens.

It is also hard to keep track of the shapes of the various matrices and what each dimension represents.

## 2025-05-26

### understanding cross-attention

Reading openai whisper `qkv_attention()` in `model.py`:

The resulting `qk` matrix in the `qkv_attention` function represents the scaled dot-product attention scores between the query (`q`) and key (`k`) tensors. Its shape is determined by the batch size, number of attention heads, and the sequence lengths of the query and key tensors.

#### Breakdown of the `qk` Shape (General Case)

1.  **Input Shapes**:

    -   `q` (query): Shape `(n_batch, n_ctx_q, n_state)`
        -   `n_batch`: Number of samples in the batch.
        -   `n_ctx_q`: Query sequence length (e.g., number of text tokens).
        -   `n_state`: Hidden state size (dimensionality of the model).
    -   `k` (key): Shape `(n_batch, n_ctx_k, n_state)`
        -   `n_ctx_k`: Key sequence length (e.g., number of audio frames or text tokens).
    -   `v` (value): Shape `(n_batch, n_ctx_k, n_state)` (same shape as `k`)

2.  **Reshaping for Multi-Head Attention**:
    The `qkv_attention` function in `model.py` reshapes `q` and `k` as follows:

    -   `q = q.view(n_batch, n_ctx_q, self.n_head, -1).permute(0, 2, 1, 3)`
        -   Results in `q` having shape `(n_batch, self.n_head, n_ctx_q, head_dim)`, where `head_dim = n_state // self.n_head`.
    -   `k = k.view(n_batch, n_ctx_k, self.n_head, -1).permute(0, 2, 1, 3)`
        -   Results in `k` having shape `(n_batch, self.n_head, n_ctx_k, head_dim)`.

3.  **Dot Product to Compute `qk`**:
    The scaled dot product is computed:
    ```python
    qk = (q * scale) @ (k * scale).transpose(-1, -2)
    ```
    -   `q` has shape `(n_batch, self.n_head, n_ctx_q, head_dim)`.
    -   `(k * scale).transpose(-1, -2)` has shape `(n_batch, self.n_head, head_dim, n_ctx_k)`.
    -   The resulting `qk` has shape `(n_batch, self.n_head, n_ctx_q, n_ctx_k)`.

#### Final Shape of `qk`

-   `qk` is a 4D tensor with shape:
    ```
    (n_batch, n_head, n_ctx_q, n_ctx_k)
    ```
    -   `n_batch`: Number of samples in the batch.
    -   `n_head`: Number of attention heads.
    -   `n_ctx_q`: Sequence length of the query.
    -   `n_ctx_k`: Sequence length of the key.
    -   **Note**: For self-attention, `q`, `k`, and `v` are derived from the same input sequence, so `n_ctx_q == n_ctx_k`. For cross-attention, `q` is derived from one sequence (e.g., text tokens) and `k`, `v` from another (e.g., audio features), so `n_ctx_q` and `n_ctx_k` can be different.

#### Interpretation of `qk`

-   Each slice `qk[b, h]` (for a specific batch `b` and head `h`) is a matrix of shape `(n_ctx_q, n_ctx_k)`.
-   Each element `qk[b, h, i, j]` represents the attention score between the `i`-th query token (from the query sequence) and the `j`-th key token (from the key sequence) for the `h`-th attention head in the `b`-th batch.

#### Related Code in `timing.py` (Cross-Attention for Alignment)

The `timing.py` script uses hooks on the `cross_attn` modules of the decoder blocks. In this context:

-   The **query (`q`)** is derived from the text tokens. So, `n_ctx_q` corresponds to the number of text tokens in the current sequence (let's call this `n_text_tokens`).
-   The **key (`k`)** and **value (`v`)** are derived from the encoded audio features (`xa`). So, `n_ctx_k` corresponds to the number of audio frames in the encoded audio (let's call this `n_audio_frames`, which is typically `mel_num_frames // 2`).

The hook `QKs.__setitem__(index, outs[-1][0])` stores `qk[0]` (assuming batch size `n_batch=1` for typical inference in `timing.py`).

-   The `[-1]` is because `MultiHeadAttention.forward()` returns `(wv, qk)`, and `[0]` selects the only batch at index 0.
-   Thus, `QKs[layer_index]` will have the shape `(n_text_head, n_text_tokens, n_audio_frames)`.
-   `QKs[layer_index][head_index]` will be a 2D matrix of shape `(n_text_tokens, n_audio_frames)`.

#### Multiple Iterations

The whole forward pass (qk generation) generates a single text token, so the model is run iteratively until it generates the End-Of-Text token.

In the OpenAI code, the timing pass is after the text tokens are generated, and any generated token is dropped; this is when the hooks are added (and removed once finished) to record the QKs.

Recording the QKs during the model run means only the last run needs to be saved for timing purposes.

#### The timing weights

The `weights` tensor is constructed as:

```python
weights = torch.stack([QKs[_l][_h] for _l, _h in model.alignment_heads.indices().T])
```

-   `QKs[_l][_h]` is the 2D attention score matrix `(n_text_tokens, n_audio_frames)` for a specific alignment head `_h` in layer `_l`.
-   The resulting `weights` tensor will have the shape:
    ```
    (N_selected_alignment_heads, n_text_tokens, n_audio_frames)
    ```
    where `N_selected_alignment_heads` is the total number of heads selected for alignment across all layers.

The subsequent operations in `timing.py` like `weights = weights[:, :, : num_frames // 2]` use these dimensions.

#### Summary

-   The general shape of `qk` from `qkv_attention` is `(n_batch, n_head, n_ctx_q, n_ctx_k)`.
-   For **self-attention**, `n_ctx_q = n_ctx_k = n_ctx` (sequence length of the input).
-   For **cross-attention** (e.g., in decoder attending to encoder output, as used for alignment in `timing.py`), `n_ctx_q` is the target sequence length (e.g., text tokens) and `n_ctx_k` is the source sequence length (e.g., audio frames).
-   The `weights` tensor in `timing.py`, derived from cross-attention `qk` values, has the shape `(N_selected_alignment_heads, n_text_tokens, n_audio_frames)`.

## 2025-06-03

## 2025-06-03

tpde alternative to LLVM just released. no instructions for using it with rust yet. try cranelift.

```sh
rustup component add rustc-codegen-cranelift-preview --toolchain nightly
cargo +nightly build
# Finished `dev` profile [unoptimized + debuginfo] target(s) in 4m 28s
```

that does not seem faster. try normal build

oh it's a lot faster. normal build

```sh
cargo build
# Finished `dev` profile [unoptimized + debuginfo] target(s) in 13m 39s
```

## 2025-06-04

OK, now that word alignment is done and showing up in the UI, the next step is to make it continuous.

That means:

1. streaming MEL frames to the UI
2. streaming MEL frames to a new preview model
3. setting up the new preview model
4. setting up some sort of chunking (last five seconds?) for previews
5. voice activity chunking

## 2025-06-05

I wrote the UI code to receive individual columns (frames) of mel spectrogram. then I tried to modify the audio loop. how much audio to process through pcm_to_mel ? turns out, the fft takes 400 samples, but then it only steps forward 160 samples. so it's actually looking ahead by about 2.5 frames.

## 2025-06-06

partials mostly implemented now. but not showing up in the ui.

I think the frame shape is wrong -- columns vs. rows.

got this error again as well

```log
ERROR mubbles::whisper: Whisper thread failed: narrow invalid args start + len > dim_len: [4, 750], dim: 1, start: 0, len:867
```

I think because of a very short input; the attention bit can't do the median thing neatly. wait I disabled that. I dunno. this will be probably be fixed by VAD and requiring at least 2s of audio. a stack trace would help.

make sure the image is getting updated

it's not

ok, spectrogram working now. It's upside down and the default value should be -10, but it's working.

## 2025-06-07

### time travel debugging

Tomorrow Corporation Game Devs post about how this works:

> The act of going forward and back is not itself recorded – just the evolution of the game’s state.
>
> The snapshots happen according to 2 different schedules – a coarse grain schedule that records a new snapshot every so often based on time (we do every 2 minutes currently) and a fine grain limited set of snapshots that move around depending on where you are currently working on the timeline. That’s why the initial reverse debugger step causes a brief pause and then becomes fast – the first one seeks back to the most recent coarse grain snapshot, simulates forward creating fine grained snapshots that are exponentially spaced out backwards from your seek target, and then the subsequent steps will tend to have a snapshot that is right on the frame you need (or very close – unless you step back far enough to need to go create more snapshots but that is the rare case.)
>
> The state capture is mostly just a memcpy of the game’s heap (snapshots only happen on frame boundaries so the stack is never needed.) It for sure could be too big to keep as many snapshots as we currently do – that will just be game dependent. Something to use to calibrate what you expect is possible though is to remember that games like Metroid Prime, LOZ The Wind Waker, RE4, Mario Sunshine, etc. all ran on a system that basically had 24MB of RAM to use for your game. And it wasn’t just game state filling up that 24MB, it was your code and other read only resources – the kind of stuff that we don’t have to include in our snapshots. So while it’s true that this system is not a general purpose solution for any and all kinds of games, it’s also true that you can make some pretty incredible games with not a ton of actual game state.
>
> Yes in theory you could totally fork the timeline in the past and create a new session based off of the old one up to that point. That is a feature that I had in the reverse engineering debugger I made before this because it was good for creating what were basically tool assisted speed run videos for code coverage purposes. For our current system though it hasn’t been something that I thought we would actually use enough to justify spending the time to implement it.
>
> The code gen is totally custom but keep in mind that this toolchain only needs to run on our development platform which is Windows. To ship the finished game we will transpile the code to C and then compile it with the native toolchains on whatever platforms we’re targeting.

The main trick here is replaying inputs (deterministically) to recreate the state of the program. Dumping the heap is an optimization to avoid having to replay from the beginning.

Code modifications are also tracked (I guess the same way inputs are?).

## 2025-06-09

getting ready to call the incremental model

## 2025-06-10

incremental kinda working. now the main model's broken, despite making sure at least three seconds goes into the transcriber.

```log
Temperature error running at 0: narrow invalid args start + len > dim_len: [9, 750], dim: 1, start: 0, len:1122
```

Improved the error messages. I think the problem is in real_audio_len being a bit long -- calculating to 795 when model only has 750 audio tokens. Just disable this trim for now.

yep that fixed it. 

ok, now the timestamps are showing up well. i can segment by them.

## 2025-06-11

ok, everything's sort of working now. why is partials so slow ? 

also profile everything.

also need to split segments into phrases and remove the timestamp tokens.

## 2025-06-12

tracing shows DistilWhisperLargeV3 takes about 5ms per token. if looping or something that's bad. Each mel frame takes 