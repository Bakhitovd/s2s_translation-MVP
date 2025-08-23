# AUDIO ROUTING

## Overview
This document explains how to configure audio routing for the EN↔RU speech-to-speech MVP.

## Microphone Input
- By default, the browser captures microphone input via `getUserMedia`.
- Ensure the correct microphone device is selected in Chrome.

## System / Meeting Audio
Chrome cannot capture system audio directly. To route system or meeting audio into the MVP:

### Option 1: Stereo Mix (if available)
1. Right-click the speaker icon in Windows taskbar → "Sounds".
2. Go to the "Recording" tab.
3. Enable "Stereo Mix" device.
4. Select "Stereo Mix" as the input device in Chrome.

### Option 2: Virtual Audio Cable (VB-CABLE)
1. Download and install VB-CABLE (https://vb-audio.com/Cable/).
2. Set system output device to "CABLE Input".
3. In Chrome, select "CABLE Output" as the microphone input.
4. This routes all system audio into the MVP.

## Testing
- Play a YouTube video or meeting audio.
- Ensure Chrome is capturing the loopback device.
- The MVP should receive audio frames and return translated speech.

## Troubleshooting
- If no audio is captured, verify Chrome permissions for microphone.
- Ensure only one loopback device is active to avoid conflicts.
- Check Windows Sound settings to confirm routing.
