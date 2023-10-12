# Changelog

## 1.4
 - statistical summary shows five most unusual words for each ten lines of transcript

## 1.3 
 - auto-scrolling each time transcription is added

## 1.2
 - Log to file in case of accidental clear

## 1.1 

 - Always on Top checkbox keeps the window visible while Zoom is open.
 - Open window faster (load model in thread)

## 1.0

 - Initial release
 - Select input or output audio stream
 - Transcribes to textbox
 - Autotype uses OS to type transcript through keyboard to whatever app is foreground
 - Shows chart of input volume
 - Set beam size for speech recognition -- lower is faster but less polished output
 - Uses CUDA / whisper-rs / whisper.cpp for speech recognition