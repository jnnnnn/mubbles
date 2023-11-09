# stream data from microphone, through whisper transcriber, to text output

# https://github.com/snakers4/silero-vad/blob/master/examples/pyaudio-streaming/pyaudio-streaming-examples.ipynb

try:
    print("Importing dependencies...")
    import sys
    import os
    import time
    from datetime import datetime
    import numpy as np
    import torch
    # pyaudiowpatch allows recording from loopbacks/outputs
    import pyaudiowpatch as pyaudio
    import librosa
    # this was for checking waveforms while debugging how bytes are converted to floats
    #import pylab
    from faster_whisper import WhisperModel
except ImportError:
    print(
        """Import error. Please run the following command:
            pip install faster-whisper numpy torch pyaudiowpatch pydub --upgrade
        """
    )
    sys.exit(-1)

print("Completed imports")

outfile = os.path.expanduser("~/transcripts/transcript-speakers.txt")
if not os.path.exists(os.path.dirname(outfile)):
    os.makedirs(os.path.dirname(outfile))
with open(outfile, "a") as f:
    print(f"Starting new transcript at {datetime.now()}", file=f)

print("Printing audio devices:")
pa = pyaudio.PyAudio()
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    print(f"{info['index']}: {info['name']} ")

wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)

# Get default WASAPI speakers
default_speakers = pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

if not default_speakers["isLoopbackDevice"]:
    for loopback in pa.get_loopback_device_info_generator():
        """
        Try to find loopback device with same name(and [Loopback suffix]).
        Unfortunately, this is the most adequate way at the moment.
        """
        if default_speakers["name"] in loopback["name"]:
            default_speakers = loopback
            break
    else:
        print("Default loopback output device not found.\n\nRun `python -m pyaudiowpatch` to check available devices.\nExiting...\n")
        exit()
        
print(f"Recording from: ({default_speakers['index']}){default_speakers['name']}: {default_speakers}")

# 1 second chunks, smaller means sentences keep getting split up
CHUNK_SECONDS = 1
INPUT_CHANNELS = int(default_speakers["maxInputChannels"])
INPUT_SAMPLE_RATE = int(default_speakers["defaultSampleRate"])
INPUT_CHUNK = int(INPUT_SAMPLE_RATE * CHUNK_SECONDS)
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE * CHUNK_SECONDS)  

# this is in the format whisper expects, 16khz mono
available_data = np.zeros((0), dtype=np.float32)

def callback(input_data, frame_count, time_info, flags):
    #print(f"callback: {type(input_data)} - {frame_count} frames, {len(input_data)} samples, {flags}, {input_data[:20]}")
    # resample to what Whisper expects: 16khz mono f32
    floats = librosa.util.buf_to_float(input_data, n_bytes = 2, dtype=np.float32)
    if INPUT_CHANNELS > 1:
        floats = np.reshape(floats, (INPUT_CHANNELS, -1), order='F')
        floats = librosa.to_mono(floats)
    input_data = librosa.resample(floats, orig_sr=INPUT_SAMPLE_RATE, target_sr=SAMPLE_RATE)
    global available_data
    available_data = np.append(available_data, input_data)
    return input_data, pyaudio.paContinue

stream = pa.open(
    format=pyaudio.paInt16, # taking in raw f32 was too hard, something about little-endian
    channels=INPUT_CHANNELS,
    rate=INPUT_SAMPLE_RATE,
    input=True,
    input_device_index=default_speakers["index"],
    stream_callback=callback,
    frames_per_buffer=INPUT_CHUNK,
)

torch.set_num_threads(1)

print("Loading VAD model...")
(
    vad_model,
    _,
) = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
print("Loading whisper model 'large'...")
transcribe_model = WhisperModel("large-v2", compute_type="int8")
#transcribe_model = WhisperModel("tiny", compute_type="int8")
print("Loaded whisper model")


def validate(model, inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs


def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()
    return sound


voice_data = np.zeros((0), dtype=np.float32)


# transcribe repeatedly, overwriting until silence is detected
PARTIALS = False

# Silero-Voice-Activity-Detector's reported "Confidence" that speech is present
confidence = 0.0
# Above this confidence, we assume that speech is present
VAD_THRESHOLD = 0.4
prev_confidence = confidence
prev_audio = np.zeros((0), dtype=np.float32)
# keep stream open
last_time = datetime.now()
while stream.is_active():
    samples = available_data
    available_data = np.zeros((0), dtype=np.float32)
    read_data = False
    if len(samples):
        confidence = vad_model(torch.from_numpy(samples), 16000).item()
        # when confidence starts to be above the threshold,
        # we want to keep the chunk before (incase there was part of a word in it),
        if confidence > VAD_THRESHOLD and prev_confidence < VAD_THRESHOLD:
            # keep the previous audio after all
            voice_data = np.append(voice_data, prev_audio)
        # and the chunk after it drops (just to be sure we don't cut off a word)
        if confidence > VAD_THRESHOLD or prev_confidence > VAD_THRESHOLD:
            voice_data = np.append(voice_data, samples)
            prev_audio = np.zeros((0), dtype=np.float32)
        else:
            # we only want to keep previous audio if we haven't already
            # added it to voice_data, to make sure we don't add it twice
            prev_audio = samples

        prev_confidence = confidence
        idle = False
    else:
        # if we didn't read any data, sleep for a bit to avoid busy-waiting
        idle = True
        time.sleep(0.1)

    # transcribe once confidence that someone is still speaking drops below 0.5
    if (
        confidence < VAD_THRESHOLD
        and prev_confidence < VAD_THRESHOLD
        and len(voice_data) > 0
    ):
        # print(" " * 60, end="\r")  # clear line
        # print(f"transcribing {len(voice_data) / SAMPLE_RATE} seconds", end="\r")
        transcribe_data = voice_data
        voice_data = np.zeros((0), dtype=np.int16)
        segments, info = transcribe_model.transcribe(
            transcribe_data, beam_size=6, language="en"
        )
        # walk the generator so we don't clear line "transcribing" too soon
        result = " ".join(s.text.strip() for s in segments)
        print(" " * 60, end="\r")  # clear line
        print("t: " + result)
        # save to ~/transcripts/transcript-speakers.txt
        # if the minute has changed, log a line saying the current time.
        if last_time.minute != datetime.now().minute:
            last_time = datetime.now()
            with open(outfile, "a") as f:
                print(f"The time is now {last_time}", file=f)
        with open(outfile, "a") as f:
            f.write(f"t: {result}\n")
    elif not idle or len(voice_data) == 0 or not PARTIALS:
        print(
            f"confidence: {confidence:.1f}; voice_data: {len(voice_data) / SAMPLE_RATE}s; idle: {idle}",
            end="\r",
        )
    else:  # idle and voice_data is not empty
        # print partial transcription?!
        start = datetime.now()
        segments, info = transcribe_model.transcribe(
            voice_data, beam_size=6, language="en"
        )
        result = " ".join(s.text.strip() for s in segments)
        print(" " * 60, end="\r")
        proctime = datetime.now() - start
        print(f"p: {result} ({proctime.total_seconds():.2f}s)", end="\r")
