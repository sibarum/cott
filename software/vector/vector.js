

// ===
// Vectors
// ===

VECTOR = {}
VECTOR.Vector = function(numSamples) {
    this.samples = new Float32Array(numSamples);
}
VECTOR.Vector.prototype.blend = function(waveform, blendFunc) {
    const step = 1.0 / this.samples.length;
    let phase = 0;
    for (let i=0; i<this.samples.length; i++) {
        this.samples[i] = blendFunc(this.samples[i], waveform.sample(i, phase));
        phase += step;
    }
}
VECTOR.Vector.prototype.add = function(waveform) {
    this.blend(waveform, (v1, v2) => v1 + v2);
}

// ===
// Waveforms
// ===

VECTOR.WAVEFORM = {};
VECTOR.WAVEFORM.Chirp = function(offset, value) {
    this.offset = offset;
    this.value = value;
}
VECTOR.WAVEFORM.Chirp.prototype.sample = function(index, phase) {
    if (index === this.offset) {
        return this.value;
    } else if (index === this.offset + 1) {
        return -this.value/2;
    } else {
        return 0;
    }
}

VECTOR.WAVEFORM.Sine = function(offset=0, frequency=1, amplitude=1) {
    this.offset = offset;
    this.frequency = frequency;
    this.amplitude = amplitude;
    this.phaseModifiers = [];
}
VECTOR.WAVEFORM.Sine.prototype.addPhaseModifier = function(phaseModifier) {
    this.phaseModifiers.push(phaseModifier)
}
VECTOR.WAVEFORM.Sine.prototype.sample = function(index, phase) {
    let sinePhase = 2*Math.PI*(this.offset + this.frequency * phase);
    for (let i=0; i<this.phaseModifiers.length; i++) {
        sinePhase = this.phaseModifiers[i](sinePhase);
    }
    return this.amplitude * Math.sin(sinePhase);
}