class EncoderWorklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.buffer = [];
    this.inputSampleRate = sampleRate; // usually 48kHz
    this.targetSampleRate = 16000;
    // Allow chunk size to be set via processorOptions.chunkMs (default 20 ms)
    const chunkMs = options && options.processorOptions && options.processorOptions.chunkMs
      ? options.processorOptions.chunkMs
      : 20;
    this.chunkSize = Math.floor(this.targetSampleRate * (chunkMs / 1000));
    this.resampleRatio = this.inputSampleRate / this.targetSampleRate;
    // Simple energy gate configuration
    this.gateThreshold = (options && options.processorOptions && typeof options.processorOptions.gateThreshold === 'number')
      ? options.processorOptions.gateThreshold
      : 0.02;
    const hangMs = (options && options.processorOptions && typeof options.processorOptions.hangoverMs === 'number')
      ? options.processorOptions.hangoverMs
      : 300;
    this.hangoverSamples = Math.max(0, Math.floor(this.inputSampleRate * (hangMs / 1000)));
    this.isVoiced = false;
    this.belowCounter = 0;
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
      return sum / (input.length || 1);
    });

    // Simple energy gate (RMS on input-rate block)
    let sumsq = 0;
    for (let i = 0; i < mono.length; i++) {
      const v = mono[i];
      sumsq += v * v;
    }
    const rms = Math.sqrt(sumsq / (mono.length || 1));

    if (rms >= this.gateThreshold) {
      this.isVoiced = true;
      this.belowCounter = 0;
    } else if (this.isVoiced) {
      this.belowCounter += mono.length;
      if (this.belowCounter >= this.hangoverSamples) {
        // Close gate: flush any residual samples, then notify server
        if (this.buffer.length > 0) {
          const tail = this.buffer.splice(0, this.buffer.length);
          const int16tail = new Int16Array(tail.length);
          for (let j = 0; j < tail.length; j++) {
            let s = Math.max(-1, Math.min(1, tail[j]));
            int16tail[j] = s < 0 ? s * 0x8000 : s * 0x7fff;
          }
          this.port.postMessage(int16tail.buffer, [int16tail.buffer]);
        }
        this.isVoiced = false;
        this.belowCounter = 0;
        this.port.postMessage({ type: 'flush' });
      }
    }

    // If gate is open, resample and emit chunks
    if (this.isVoiced) {
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
    }

    return true;
  }
}

registerProcessor("encoder-worklet", EncoderWorklet);
