# TODO: Implement Real-Time Translation Pipeline (Backend)

- [ ] Segment incoming PCM data into utterances (e.g., by size or silence detection)
- [ ] Run Canary AST (ASR+translation) on each segment, log input/output
- [ ] Run Silero TTS on translated text, log input/output
- [ ] Resample Silero output to 16kHz mono Int16LE PCM, log step
- [ ] Send translated PCM back to client, log transmission
- [ ] Send transcript JSON to client, log event
- [ ] Handle errors gracefully and log them
- [ ] Test end-to-end: verify translation and audio output in browser
