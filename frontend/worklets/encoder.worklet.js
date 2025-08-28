class EncoderWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = [];
    this.inputSampleRate = sampleRate; // usually 48kHz
    this.targetSampleRate = 16000;
    this.chunkSize = Math.floor(this.targetSampleRate * 0.02); // 20ms ~320 samples
    this.resampleRatio = this.inputSampleRate / this.targetSampleRate;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input.length === 0) return true;

    // Downmix to mono
    const mono = input[0].map((_, i) => {
      let sum = 0;
      for (let ch = 0; ch < input.length; ch++) {
        sum += input[ch][i];
      }
      return sum / input.length;
    });

    // Resample to 16kHz
    for (let i = 0; i < mono.length; i += this.resampleRatio) {
      const idx = Math.floor(i);
      if (idx < mono.length) {
        this.buffer.push(mono[idx]);
      }
      if (this.buffer.length >= this.chunkSize) {
        const chunk = this.buffer.splice(0, this.chunkSize);
        // Convert float32 [-1,1] to Int16LE
        const int16 = new Int16Array(chunk.length);
        for (let j = 0; j < chunk.length; j++) {
          let s = Math.max(-1, Math.min(1, chunk[j]));
          int16[j] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        this.port.postMessage(int16.buffer, [int16.buffer]);
      }
    }

    return true;
  }
}

registerProcessor("encoder-worklet", EncoderWorklet);
