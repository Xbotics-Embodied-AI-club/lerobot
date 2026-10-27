const SILENT_WAV = 'data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQIAAAAA';
const STORAGE_KEY = 'so101-calibration-speech';

export class SpeechPlayer {
  constructor({ onError } = {}) {
    this.onError = onError || (() => {});
    this.enabled = this._readEnabled();
    this.unlocked = false;
    this.pendingText = '';
    this.currentText = '';
    this.requestId = 0;
    this.audio = null;
    this.audioUrl = null;
    this.errorReported = false;
  }

  _readEnabled() {
    try {
      return localStorage.getItem(STORAGE_KEY) !== 'off';
    } catch {
      return true;
    }
  }

  setEnabled(enabled) {
    this.enabled = Boolean(enabled);
    try {
      localStorage.setItem(STORAGE_KEY, this.enabled ? 'on' : 'off');
    } catch {
      // Ignore storage restrictions.
    }
    if (!this.enabled) this.stop();
    else this.speak(this.pendingText, { force: true });
  }

  async unlock() {
    if (this.unlocked) return true;
    try {
      const audio = new Audio(SILENT_WAV);
      await audio.play();
      this.unlocked = true;
      this.speak(this.pendingText, { force: true });
      return true;
    } catch {
      return false;
    }
  }

  speak(text, { force = false } = {}) {
    const normalized = String(text || '').trim();
    this.pendingText = normalized;
    if (!this.enabled || !this.unlocked || !normalized) return;
    if (!force && normalized === this.currentText) return;
    this.currentText = normalized;
    const requestId = ++this.requestId;
    this._stopAudio();

    fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: normalized }),
    })
      .then(async (response) => {
        if (!response.ok) throw new Error('TTS request failed');
        return response.blob();
      })
      .then((blob) => {
        if (requestId !== this.requestId || !this.enabled) return;
        this.audioUrl = URL.createObjectURL(blob);
        this.audio = new Audio(this.audioUrl);
        this.audio.onended = () => this._stopAudio();
        this.audio.onerror = () => this._reportError();
        return this.audio.play().catch((error) => {
          if (requestId === this.requestId) this._reportError(error);
        });
      })
      .catch(() => {
        if (requestId === this.requestId) this._reportError();
      });
  }

  stop() {
    this.requestId += 1;
    this.currentText = '';
    this._stopAudio();
  }

  _stopAudio() {
    if (this.audio) {
      this.audio.pause();
      this.audio.src = '';
      this.audio = null;
    }
    if (this.audioUrl) {
      URL.revokeObjectURL(this.audioUrl);
      this.audioUrl = null;
    }
  }

  _reportError() {
    this._stopAudio();
    if (this.errorReported) return;
    this.errorReported = true;
    this.onError('语音播报不可用，已保留文字提示。');
  }
}
