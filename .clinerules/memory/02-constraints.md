# Constraints
- No cloud APIs. Everything offline.
- Chrome cannot capture system audio directly; use Windows loopback (Stereo Mix or VB-CABLE).
- Canary v2 expects 16 kHz mono input; returns translated text; use timestamps later.
- Silero models support RU/EN at 8/16 kHz; runs CPU or CUDA.
- Keep repo simple: no monorepo tooling.
