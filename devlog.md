# devlog

## 2023-05-06

started dev. I tried out a few other whisper desktop things but they don't work very well. 

I also didn't find one that will record what both the mic and the speakers are saying, in two separate columns. I feel like that's a pretty cool feature.

## 2023-05-07

First step: get the audio from the mic and the speakers. Seems like the best rust audio library is soundio.  See if I can get it to work at all.

Soundio doesn't compile:

    C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.34.31933\include\stdatomic.h(15,1): fatal  error C1189: #error:  <stdatomic.h> is not yet supported when compiling as C, but this is planned for a future release. [C:\Users\J\Source\subbles\target\debug\build\libsoundio-sys-e7b60699a3621a70\out\build\libsoundio_static.vcxproj]

Google (and copilot) suggests cpal. Try that instead.

