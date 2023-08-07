# Changelog

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