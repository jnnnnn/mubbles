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

>    Obtain an IMMDevice interface for the rendering endpoint device.
>    Initialize a capture stream in loopback mode on the rendering endpoint device.

Derp. After looking through various docs and the cpal code, I found that the place where I would add WASAPI loopback in CPAL is already configured for it! https://github.com/RustAudio/cpal/commit/78e84521507a3aa4760ec81ac62943165f5218cd . I just need to treat an output device as input.

Struggling with this a bit. My test is getting `StreamTypeNotSupported`. There have been several pull requests about this as part of cpal. Here's some example code for using loopback:

https://github.com/RustAudio/cpal/pull/478