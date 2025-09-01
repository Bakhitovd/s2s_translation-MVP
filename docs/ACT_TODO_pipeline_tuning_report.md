# TODO: Pipeline Tuning Report for EN↔RU Speech-to-Speech MVP

- [ ] Review frontend audio capture and encoding (app.js, encoder.worklet.js, index.html)
- [ ] Identify tunable parameters in frontend (input device, sample rate, chunk size, VAD, etc.)
- [ ] Review WebSocket streaming and protocol (buffer sizes, message types)
- [ ] Review backend audio segmentation and preprocessing (server/app.py, audio_util.py)
- [ ] Identify tunable parameters in Canary AST (model config, segment size, etc.)
- [ ] Identify tunable parameters in Silero TTS (model config, speaker, sample rate, etc.)
- [ ] Review resampling and audio format conversions (audio_util.py)
- [ ] Review playback and output device selection (frontend)
- [ ] Summarize all parameters, their effects, and how to tune them
- [ ] Prepare final report in docs/pipeline_tuning_report.md
