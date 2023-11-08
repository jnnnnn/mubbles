# stream data from microphone, through whisper transcriber, to text output

# https://github.com/snakers4/silero-vad/blob/master/examples/pyaudio-streaming/pyaudio-streaming-examples.ipynb

try:
    import sys
    import time
    from datetime import datetime
    import numpy as np
    import torch
    import pyaudio
    from faster_whisper import WhisperModel
except ImportError:
    print(
        """Import error. Please run the following command:
            pip install faster-whisper numpy torch pyaudio --upgrade
        """
    )
    sys.exit(-1)

torch.set_num_threads(1)

print("Loading VAD model...")
(
    vad_model,
    _,
) = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
print("Loading whisper model 'large'...")
transcribe_model = WhisperModel("large-v2", compute_type="int8")
print("Loaded whisper model")

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 2)  # 1/2 second chunks

audio = pyaudio.PyAudio()


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


available_data = []
voice_data = np.zeros((0), dtype=np.float32)


def callback(input_data, frame_count, time_info, flags):
    available_data.append(input_data)
    return input_data, pyaudio.paContinue


pa = pyaudio.PyAudio()

stream = pa.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    stream_callback=callback,
    frames_per_buffer=CHUNK,
)
# transcribe repeatedly, overwriting until silence is detected
PARTIALS = False

# Silero-Voice-Activity-Detector's reported "Confidence" that speech is present
confidence = 0.0
# Above this confidence, we assume that speech is present
VAD_THRESHOLD = 0.4
prev_confidence = confidence
prev_audio = np.zeros((0), dtype=np.float32)
# keep stream open
while stream.is_active():
    data = available_data
    available_data = []
    read_data = False
    for chunk in data:
        audio_int16 = np.frombuffer(chunk, np.int16)
        audio_float32 = int2float(audio_int16)
        confidence = vad_model(torch.from_numpy(audio_float32), 16000).item()
        # when confidence starts to be above the threshold,
        # we want to keep the chunk before (incase there was part of a word in it),
        if confidence > VAD_THRESHOLD and prev_confidence < VAD_THRESHOLD:
            # keep the previous audio after all
            voice_data = np.append(voice_data, prev_audio)
        # and the chunk after it drops (just to be sure we don't cut off a word)
        if confidence > VAD_THRESHOLD or prev_confidence > VAD_THRESHOLD:
            voice_data = np.append(voice_data, audio_float32)
            prev_audio = np.zeros((0), dtype=np.float32)
        else:
            # we only want to keep previous audio if we haven't already
            # added it to voice_data, to make sure we don't add it twice
            prev_audio = audio_float32

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
        with open("transcript.txt", "a") as f:
            f.write(result + "\n")
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
