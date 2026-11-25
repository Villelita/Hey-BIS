/**
 * ============================================================================
 * BIS VOICE ASSISTANT — CONTINUOUS VOICE & MICROPHONE INTELLIGENCE ENGINE
 * ============================================================================
 * 1. Captures live microphone stream via Web Audio API to make the 3D neural
 *    core deform and ripple to your voice in real time.
 * 2. Continuously listens for the wake phrase "Hey Bis" (or "Bis", "Oye Bis",
 *    "Ey Bis", "Hola Bis") using Web Speech Recognition (auto-restarting 24/7).
 * 3. Shows live real-time audio volume VU-meter and transcription feedback.
 * 4. Generates instant natural responses and speaks them aloud.
 * ============================================================================
 */

class AudioReactive {
  constructor(core, ui) {
    this.core = core;
    this.ui = ui;

    // Web Audio API State
    this.audioCtx = null;
    this.analyser = null;
    this.microphone = null;
    this.scriptProcessor = null;
    this.mediaStream = null;
    this.dataArray = null;
    this.animId = null;
    this.isMicActive = false;

    // Neural Network Classification State (Client Mic -> Raspberry Pi /api/classify)
    this.rollingSamples16k = new Float32Array(48000); // 3.0s @ 16kHz
    this.rollingFilled = 0;
    this.isClassifying = false;
    this.isManualCapturing = false;
    this.manualChunks = [];
    this.manualTranscript = '';
    this.lastAutoClassifyTime = 0;
    this.lastWakeCooldownUntil = 0;
    this.speechStartedAt = 0;
    this.lastSpeechAt = 0;
    this.isAssistantSpeaking = false;
    this.ignoreInputUntil = 0;

    // Speech Recognition State
    this.recognition = null;
    this.isRecognizing = false;
    this.isAwake = false;
    this.silenceTimer = null;
    this.idleSubtextTimer = null;
    this.accumulatedQuery = '';

    // Cache UI Elements
    this.micBadge = document.getElementById('kiosk-badge');
    this.micBtn = document.getElementById('dev-btn-mic');
    this.hudSubtext = document.getElementById('hud-subtext');
    this.micPromptBanner = document.getElementById('mic-prompt-banner');

    this.initSpeechRecognition();
    this.initTriggers();
  }

  // ==========================================================================
  // INITIALIZATION & USER PERMISSION TRIGGERS
  // ==========================================================================
  initTriggers() {
    // 1. Click anywhere on page or banner to grant/start mic
    const triggerStart = async () => {
      if (!this.isMicActive) {
        await this.start();
      }
    };

    window.addEventListener('click', triggerStart, { once: false });
    window.addEventListener('touchstart', triggerStart, { once: false });
    window.addEventListener('keydown', (e) => {
      if (e.code === 'KeyM') triggerStart();
    });

    if (this.micPromptBanner) {
      this.micPromptBanner.addEventListener('click', (e) => {
        e.stopPropagation();
        triggerStart();
      });
    }

    if (this.micBadge) {
      this.micBadge.addEventListener('click', (e) => {
        e.stopPropagation();
        triggerStart();
      });
    }

    if (this.micBtn) {
      this.micBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        if (!this.isMicActive) {
          await this.start();
        } else {
          this.stop();
        }
      });
    }

    // Try starting immediately (works if permission was previously granted or in kiosk mode)
    setTimeout(() => {
      this.start().catch(() => {});
    }, 300);
  }

  // ==========================================================================
  // WEB AUDIO API (Captures Voice & Deforms 3D Core)
  // ==========================================================================
  async start() {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        if (window.location.protocol === 'http:' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
          const httpsUrl = `https://${window.location.hostname}:8443${window.location.pathname}${window.location.search}`;
          this.updateStatusBadge(false, '🔒 CLICK PARA ABRIR EN HTTPS (MIC)');
          window.location.href = httpsUrl;
          return false;
        }
        console.warn('[Bis Voice] getUserMedia not available in this browser context.');
        this.updateStatusBadge(false, 'Navegador no soporta micrófono');
        return false;
      }

      // Request live microphone stream from the connected client device
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      this.mediaStream = stream;

      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      this.audioCtx = new AudioContextClass();

      if (this.audioCtx.state === 'suspended') {
        await this.audioCtx.resume();
      }

      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.smoothingTimeConstant = 0.7;

      this.microphone = this.audioCtx.createMediaStreamSource(stream);
      this.microphone.connect(this.analyser);

      // Capture raw PCM from client device microphone for Raspberry Pi Neural Network (/api/classify)
      this.scriptProcessor = this.audioCtx.createScriptProcessor(4096, 1, 1);
      this.scriptProcessor.onaudioprocess = (e) => {
        if (!this.isMicActive) return;
        const input = e.inputBuffer.getChannelData(0);
        if (this.isManualCapturing) {
          this.manualChunks.push(new Float32Array(input));
        }
        this.pushToRollingBuffer(input, this.audioCtx.sampleRate);
      };
      this.analyser.connect(this.scriptProcessor);
      this.scriptProcessor.connect(this.audioCtx.destination);

      this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
      this.isMicActive = true;

      // Start volume processing loop
      this.processAudio();

      // Use instant browser SpeechRecognition when available (zero RAM load on Raspberry Pi)
      this.startRecognition();

      this.updateStatusBadge(true, '🟢 MIC ON // ATENTO A "HEY BIS"');
      if (this.micPromptBanner) {
        this.micPromptBanner.style.display = 'none';
      }

      console.log('%c[Bis Voice]%c Micrófono en vivo activo. Escuchando "Hey Bis"...', 'color: #00ff88; font-weight: bold;', 'color: #fff;');
      return true;
    } catch (err) {
      console.warn('[Bis Voice] Permiso de micrófono pendiente o denegado:', err);
      this.updateStatusBadge(false, '🎙️ CLICK PARA PERMITIR MIC');
      return false;
    }
  }

  stop() {
    this.isMicActive = false;
    this.speechStartedAt = 0;
    this.lastSpeechAt = 0;
    this.isAssistantSpeaking = false;
    this.ignoreInputUntil = 0;
    if (this.animId) cancelAnimationFrame(this.animId);
    if (this.scriptProcessor) {
      try { this.scriptProcessor.disconnect(); } catch (e) {}
      this.scriptProcessor = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(t => t.stop());
      this.mediaStream = null;
    }
    if (this.audioCtx && this.audioCtx.state !== 'closed') {
      this.audioCtx.close();
    }
    this.audioCtx = null;
    this.analyser = null;
    this.microphone = null;
    if (this.core) this.core.setAudioLevel(0.0);

    this.stopRecognition();
    this.updateStatusBadge(false, 'MIC EN ESPERA (CLICK)');
    if (this.micPromptBanner) {
      this.micPromptBanner.style.display = 'flex';
    }
  }

  processAudio() {
    if (!this.isMicActive || !this.analyser) return;

    this.animId = requestAnimationFrame(() => this.processAudio());

    this.analyser.getByteFrequencyData(this.dataArray);

    let sum = 0;
    const len = this.dataArray.length;
    for (let i = 0; i < len; i++) {
      sum += this.dataArray[i];
    }
    const avg = sum / len; // 0 to 255
    const normalized = Math.min(1.0, (avg / 55.0));

    // Modulate 3D Neural Core displacement & scale in real time with ANY ambient sound/voice
    if (this.core) {
      this.core.setAudioLevel(normalized);
    }
  }

  // ==========================================================================
  // NEURAL WAKE WORD PIPELINE (Client Device Mic -> Raspberry Pi /api/classify)
  // ==========================================================================
  resampleTo16k(samples, fromRate, toRate = 16000) {
    if (fromRate === toRate) return samples;
    const ratio = toRate / fromRate;
    const outLen = Math.round(samples.length * ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const pos = i / ratio;
      const idx = Math.floor(pos);
      const next = Math.min(idx + 1, samples.length - 1);
      const frac = pos - idx;
      out[i] = samples[idx] * (1 - frac) + samples[next] * frac;
    }
    return out;
  }

  encodeWAV16k(samples, sampleRate = 16000) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const writeStr = (offset, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, 'data');
    view.setUint32(40, samples.length * 2, true);
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([view], { type: 'audio/wav' });
  }

  pushToRollingBuffer(chunkNative, nativeRate) {
    const chunk16k = this.resampleTo16k(chunkNative, nativeRate, 16000);
    const n = chunk16k.length;
    if (n >= 48000) {
      this.rollingSamples16k.set(chunk16k.subarray(n - 48000));
      this.rollingFilled = 48000;
    } else {
      this.rollingSamples16k.copyWithin(0, n);
      this.rollingSamples16k.set(chunk16k, 48000 - n);
      this.rollingFilled = Math.min(48000, this.rollingFilled + n);
    }

    const now = performance.now();
    // Ignorar completamente el audio del micrófono mientras Bis está hablando o en enfriamiento anti-eco
    if (
      this.isAssistantSpeaking ||
      now < this.ignoreInputUntil ||
      this.isManualCapturing ||
      this.isClassifying ||
      !['idle', 'listening'].includes(this.ui.state)
    ) {
      this.speechStartedAt = 0;
      this.lastSpeechAt = 0;
      return;
    }
    let energy = 0;
    for (const sample of chunk16k) energy += sample * sample;
    if (n && Math.sqrt(energy / n) >= 0.008) {
      if (!this.speechStartedAt) this.speechStartedAt = now;
      this.lastSpeechAt = now;
    }
    // Wait for a short pause, or the three-second window limit, before decoding.
    if (this.speechStartedAt && (now - this.lastSpeechAt >= 350 || now - this.speechStartedAt >= 2700)) {
      this.speechStartedAt = 0;
      this.lastSpeechAt = 0;
      this.classifyRollingWindow();
    }
  }

  async classifyRollingWindow() {
    if (this.isClassifying || this.isManualCapturing || !['idle', 'listening'].includes(this.ui.state)) return;
    this.isClassifying = true;
    const wasAwake = this.isAwake;
    if (wasAwake && this.silenceTimer) clearTimeout(this.silenceTimer);
    try {
      const snapshot = new Float32Array(this.rollingSamples16k);
      const wavBlob = this.encodeWAV16k(snapshot, 16000);
      // En navegadores con SpeechRecognition (Chrome/Safari móvil), solo consultamos la red neuronal ligera (/api/classify)
      // para no saturar la RAM de la Raspberry Pi con Whisper en cada frase.
      const useServerWhisper = !this.recognition;
      const [res, transcription] = await Promise.all([
        fetch('/api/classify', {
          method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: wavBlob
        }),
        useServerWhisper ? this.transcribeClip(wavBlob) : Promise.resolve({ text: '', language: 'es-MX' })
      ]);
      if (!this.isMicActive || this.isManualCapturing || !['idle', 'listening'].includes(this.ui.state)) return;
      const data = res.ok ? await res.json() : null;
      const text = transcription.text;
      const wakePatterns = this.getWakePatterns();
      const hasWakeWord = text && wakePatterns.some(pattern => pattern.test(text));
      if (text && (this.isAwake || hasWakeWord)) {
        let command = text;
        wakePatterns.forEach(pattern => { command = command.replace(pattern, '').trim(); });
        command = command.replace(/^[\s,.!?¿¡]+|[\s,.!?¿¡]+$/g, '');
        if (!this.isAwake) this.triggerWake(text);
        if (command) await this.processUserCommand(command, transcription.language);
      } else if (!this.isAwake && data?.success && (data.is_wake_word || data.label === 'wake_word')) {
        const pct = (data.wake_word_probability * 100).toFixed(1);
        this.lastWakeCooldownUntil = performance.now() + 3500;
        this.triggerWake('Hey Bis', {
          source: 'neural_network',
          confidencePct: pct,
          latencyMs: data.inference_time_ms,
        });
      } else if (wasAwake && useServerWhisper) {
        this.returnToSleep();
      }
    } catch (e) {
      if (wasAwake && !this.recognition) this.returnToSleep();
      console.warn('[Bis Audio] Error procesando ventana:', e);
    } finally {
      this.isClassifying = false;
    }
  }

  async captureAndClassifyFromClientMic(durationSec = 3.0) {
    if (this.isManualCapturing) return;

    // 1. Asegurar que el micrófono del dispositivo cliente esté activo
    if (!this.isMicActive) {
      const ok = await this.start();
      if (!ok) return;
    }

    this.isManualCapturing = true;
    this.manualChunks = [];
    this.manualTranscript = '';
    const btn = document.getElementById('btn-manual-mic');
    if (btn) btn.textContent = `🔴 Grabando (${durationSec.toFixed(0)}s)...`;

    if (this.ui) {
      this.ui.setState('listening');
      this.ui.setTranscript('Escuchando desde tu micrófono... Di "Hey Bis" o "Hola Bis"', 'user', false);
      this.ui.updateSubtext(`🎙️ Grabando ${durationSec.toFixed(1)}s en tu dispositivo y enviando a la IA...`);
    }

    await new Promise(resolve => setTimeout(resolve, durationSec * 1000));
    this.isManualCapturing = false;
    if (btn) btn.textContent = '🎙️ Hablar ahora';

    const totalLen = this.manualChunks.reduce((acc, c) => acc + c.length, 0);
    if (totalLen === 0) {
      this.returnToSleep();
      return;
    }

    const merged = new Float32Array(totalLen);
    let offset = 0;
    for (const c of this.manualChunks) {
      merged.set(c, offset);
      offset += c.length;
    }

    if (this.ui) {
      this.ui.setState('thinking');
      this.ui.updateSubtext('Analizando espectrograma Log-Mel en la Red Neuronal...');
    }

    try {
      const samples16k = this.resampleTo16k(merged, this.audioCtx.sampleRate, 16000);
      const wavBlob = this.encodeWAV16k(samples16k, 16000);
      const transcription = await this.transcribeClip(wavBlob);
      this.manualTranscript = transcription.text;
      const res = await fetch('/api/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'audio/wav' },
        body: wavBlob,
      });
      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      const wakePatterns = this.getWakePatterns();
      let command = this.manualTranscript;
      wakePatterns.forEach(pattern => { command = command.replace(pattern, '').trim(); });
      command = command.replace(/^[\s,.!?¿¡]+|[\s,.!?¿¡]+$/g, '');
      if (command) {
        await this.processUserCommand(command, transcription.language);
        return;
      }
      const matchedPhonetic = this.manualTranscript && wakePatterns.some(p => p.test(this.manualTranscript));
      const isWakeDetected = Boolean(data.is_wake_word || data.label === 'wake_word' || matchedPhonetic);

      const effectiveWakeProb = isWakeDetected
        ? Math.max(data.wake_word_probability, matchedPhonetic ? 0.95 : 0.0)
        : data.wake_word_probability;
      const pWake = (effectiveWakeProb * 100).toFixed(1);
      const pSpeech = (data.probabilities.other_speech * 100).toFixed(1);
      const pBg = (data.probabilities.background * 100).toFixed(1);

      if (isWakeDetected) {
        this.playWakeChime();
        if (this.ui) {
          this.ui.setState('speaking');
          const msg = transcription.language === 'en-US'
            ? 'Hi! I’m listening. How can I help you?'
            : '¡Hola! Te escucho. ¿En qué te puedo ayudar?';
          this.ui.setTranscript(msg, 'bis', true);
          this.ui.speakAudio(msg, transcription.language);
          this.ui.updateSubtext(
            `✨ Wake Word: ${pWake}% • Otra voz: ${pSpeech}% • Fondo: ${pBg}% (${data.inference_time_ms} ms)`
          );
          setTimeout(() => this.returnToSleep(), 4500);
        }
      } else if (data.label === 'other_speech') {
        if (this.ui) {
          this.ui.setState('idle');
          this.ui.setTranscript(
            `Se detectó voz humana (${pSpeech}%), pero no un llamado como "Hey Bis" o "Hola Bis" (${pWake}%).`,
            'bis',
            true
          );
          this.ui.updateSubtext(
            `🗣️ Otra voz: ${pSpeech}% • Wake Word: ${pWake}% • Fondo: ${pBg}% (${data.inference_time_ms} ms)`
          );
        }
      } else {
        if (this.ui) {
          this.ui.setState('idle');
          this.ui.setTranscript(
            `Ruido ambiental o silencio detectado (${pBg}%). Acércate al micrófono y di "Hey Bis" o "Hola Bis".`,
            'bis',
            true
          );
          this.ui.updateSubtext(
            `🔊 Fondo: ${pBg}% • Wake Word: ${pWake}% • Otra voz: ${pSpeech}% (${data.inference_time_ms} ms)`
          );
        }
      }
    } catch (err) {
      if (this.ui) {
        this.ui.setState('error');
        this.ui.setTranscript(`Error al clasificar audio: ${err.message}`, 'bis', false);
        setTimeout(() => this.returnToSleep(), 3000);
      }
    }
  }

  // ==========================================================================
  // CONTINUOUS SPEECH RECOGNITION (Wake Phrase Detection)
  // ==========================================================================
  initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.warn('[Bis Voice] Web Speech Recognition no soportado en este navegador. Usa Chrome o Chromium.');
      return;
    }

    this.recognition = new SpeechRecognition();
    this.recognition.continuous = true;
    this.recognition.interimResults = true;
    this.recognition.maxAlternatives = 3;

    // Configura español (es-MX) por defecto porque reconoce tanto "Hey Bis" como "Hola Bis" / "Oye Bis"
    this.recognition.lang = 'es-MX';

    this.recognition.onstart = () => {
      this.isRecognizing = true;
    };

    this.recognition.onresult = (event) => {
      this.handleSpeechResult(event);
    };

    this.recognition.onerror = (event) => {
      if (event.error !== 'no-speech' && event.error !== 'aborted') {
        console.log('[Bis Voice] Recognition notification:', event.error);
      }
    };

    this.recognition.onend = () => {
      this.isRecognizing = false;
      // Only auto-restart when mic is active AND Bis is NOT currently speaking (prevents self-hearing loop)
      if (this.isMicActive && !this.isAssistantSpeaking) {
        setTimeout(() => {
          if (this.isMicActive && !this.isAssistantSpeaking) {
            this.startRecognition();
          }
        }, 180);
      }
    };
  }

  startRecognition() {
    if (!this.recognition || this.isRecognizing) return;
    try {
      this.recognition.lang = 'es-MX';
      this.recognition.start();
    } catch (e) {
      // Already running
    }
  }

  stopRecognition() {
    if (!this.recognition) return;
    try {
      this.recognition.stop();
    } catch (e) {}
    this.isRecognizing = false;
  }

  // ==========================================================================
  // WAKE WORD & CONVERSATION PIPELINE
  // ==========================================================================
  getWakePatterns() {
    return [
      /\b(?:hey|ey|hi|hello|okay|ok|a)\s*(?:bis|vis|biz|bees|beast|beasts|best|beats|beat|bits|bit|piece|peace|miss|liz|luis|gris|pis|fis|dis|this|bitch|beach|peach|fish|wish|base|bass|vase)\b/i,
      /\b(?:hola|ola|holla|oye|oyi|oh\s*yeah|all\s+of|all\s+the|hold\s+a)\s*(?:bis|vis|biz|bees|beast|beasts|best|beats|beat|bits|bit|piece|peace|miss|liz|luis|gris|pis|fis|dis|these|this|bitch|beach)\b/i,
      /\b(?:bis|hevis|heavies|elvis|pelvis|avis|ibis|abis|abies|habis|holabis|olabis|oyebis|heybis|eybis)\b/i,
    ];
  }

  normalizeWakeTranscript(rawText) {
    if (!rawText) return 'Hey Bis';
    let text = rawText;
    text = text.replace(
      /\b(?:hola|ola|holla|all\s+of|all\s+the|hold\s+a)\s*(?:bis|vis|biz|bees|beast|beasts|best|beats|beat|bits|bit|piece|peace|miss|liz|luis|gris|pis|fis|dis|these|this|bitch|beach)\b/gi,
      'Hola Bis'
    );
    text = text.replace(
      /\b(?:oye|oyi|oh\s*yeah)\s*(?:bis|vis|biz|bees|beast|beasts|best|beats|beat|bits|bit|piece|peace|miss|liz|luis|gris|pis|fis|dis|this|bitch|beach)\b/gi,
      'Oye Bis'
    );
    text = text.replace(
      /\b(?:hey|ey|hi|hello|okay|ok|a)\s*(?:bis|vis|biz|bees|beast|beasts|best|beats|beat|bits|bit|piece|peace|miss|liz|luis|gris|pis|fis|dis|this|bitch|beach|peach|fish|wish|base|bass|vase)\b/gi,
      'Hey Bis'
    );
    text = text.replace(
      /\b(?:hevis|heavies|elvis|pelvis|avis|ibis|abis|abies|habis|heybis|eybis)\b/gi,
      'Hey Bis'
    );
    text = text.replace(/\b(?:holabis|olabis)\b/gi, 'Hola Bis');
    text = text.replace(/\b(?:oyebis)\b/gi, 'Oye Bis');
    return text;
  }

  handleSpeechResult(event) {
    // Bloquear cualquier eco mientras Bis está pensando, hablando o recién terminó de hablar
    if (
      this.isAssistantSpeaking ||
      performance.now() < this.ignoreInputUntil ||
      this.ui?.state === 'speaking' ||
      this.ui?.state === 'thinking'
    ) {
      return;
    }

    let interimTranscript = '';
    let finalTranscript = '';
    let allAlternativesText = '';

    for (let i = event.resultIndex; i < event.results.length; ++i) {
      const res = event.results[i];
      const transcript = res[0].transcript;
      if (res.isFinal) {
        finalTranscript += transcript;
      } else {
        interimTranscript += transcript;
      }
      for (let a = 0; a < res.length; a++) {
        allAlternativesText += ' ' + res[a].transcript;
      }
    }

    const rawText = (finalTranscript || interimTranscript).trim().toLowerCase();
    if (!rawText) return;

    const wakePatterns = this.getWakePatterns();
    const hasWakeWord = wakePatterns.some(
      pattern => pattern.test(rawText) || pattern.test(allAlternativesText)
    );
    const currentText = hasWakeWord ? this.normalizeWakeTranscript(rawText) : rawText;

    if (this.isManualCapturing) {
      this.manualTranscript = currentText;
    }

    // Show live real-time text of what the microphone is hearing directly in the single subtle footer line
    if (!this.isAwake && this.hudSubtext) {
      this.hudSubtext.innerHTML = `<span>🟢 ${this.escapeHtml(currentText)}</span> <span class="dot"></span> <span>Di "Hey Bis" o "Hola Bis"</span>`;
      if (this.idleSubtextTimer) clearTimeout(this.idleSubtextTimer);
      this.idleSubtextTimer = setTimeout(() => {
        if (!this.isAwake && this.hudSubtext && this.isMicActive) {
          this.hudSubtext.innerHTML = `<span>🟢 Micrófono activo</span> <span class="dot"></span> <span>Di "Hey Bis" o "Hola Bis"</span>`;
        }
      }, 2500);
    }

    // CASE 1: Asleep, waiting for wake phrase
    if (!this.isAwake) {
      if (hasWakeWord) {
        this.triggerWake(currentText);
      }
      return;
    }

    // CASE 2: Awake, capturing user command
    if (this.isAwake) {
      let cleaned = currentText;
      wakePatterns.forEach(pattern => {
        cleaned = cleaned.replace(pattern, '').trim();
      });

      if (cleaned.length > 0) {
        this.accumulatedQuery = cleaned;
        if (this.hudSubtext) {
          this.hudSubtext.innerHTML = `<span>🔵 ${this.escapeHtml(cleaned)}</span>`;
        }
      }

      // Faster debounce: answer in 350ms when speech result is final, or 700ms on interim pause
      const isFinalResult = Boolean(finalTranscript);
      if (this.silenceTimer) clearTimeout(this.silenceTimer);
      this.silenceTimer = setTimeout(() => {
        this.processUserCommand(this.accumulatedQuery, this.detectLanguage(this.accumulatedQuery));
      }, isFinalResult ? 350 : 700);
    }
  }

  triggerWake(fullText, meta = null) {
    this.isAwake = true;
    this.accumulatedQuery = '';
    this.playWakeChime();

    const displayWake = this.normalizeWakeTranscript(fullText || 'Hey Bis');

    if (this.ui) {
      this.ui.setState('listening');
      this.ui.setTranscript(`"${displayWake}..."`, 'user', true);
      if (meta && meta.source === 'neural_network') {
        this.ui.updateSubtext(
          `✨ Wake Word detectada por la Red Neuronal (${meta.confidencePct}% • ${meta.latencyMs} ms) — ¡Di tu pregunta!`
        );
      } else {
        this.ui.updateSubtext('🟢 ¡Te escucho! Di tu pregunta o comando...');
      }
    }

    const wakePatterns = this.getWakePatterns();

    let queryInSameSentence = fullText || '';
    wakePatterns.forEach(pattern => {
      queryInSameSentence = queryInSameSentence.replace(pattern, '').trim();
    });

    if (queryInSameSentence.length > 2) {
      this.accumulatedQuery = queryInSameSentence;
      if (this.silenceTimer) clearTimeout(this.silenceTimer);
      this.silenceTimer = setTimeout(() => {
        this.processUserCommand(this.accumulatedQuery, this.detectLanguage(this.accumulatedQuery));
      }, 450);
    } else {
      // 6-second timeout if user doesn't say anything after "Hey Bis" / "Hola Bis"
      if (this.silenceTimer) clearTimeout(this.silenceTimer);
      this.silenceTimer = setTimeout(() => {
        if (!this.accumulatedQuery) {
          this.returnToSleep();
        }
      }, 6000);
    }
  }

  async transcribeClip(blob) {
    const response = await fetch('/api/transcribe', {
      method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: blob
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'No se pudo transcribir el audio.');
    return result;
  }

  detectLanguage(text) {
    const english = /\b(what|when|where|how|hello|please|time|weather|today|music|play|thanks|thank|your|you|date|system|status|timer|calculate|plus|minus|divided|times)\b/gi;
    const spanish = /\b(que|qué|cuando|cuándo|donde|dónde|como|cómo|hola|hora|clima|hoy|musica|música|gracias|fecha|sistema|estado|cuanto|cuánto|mas|más|menos|entre)\b/gi;
    return (text.match(english) || []).length > (text.match(spanish) || []).length ? 'en-US' : 'es-MX';
  }

  async processUserCommand(query, language = null) {
    if (!query) {
      this.returnToSleep();
      return;
    }

    this.isAwake = false;
    if (this.silenceTimer) clearTimeout(this.silenceTimer);

    // 1. Switch to Thinking state briefly so the 8-petal convergence animation plays smoothly
    if (this.ui) {
      this.ui.setState('thinking');
      this.ui.updateSubtext('Procesando...');
    }

    // Use already-transcribed query immediately (avoid re-running heavy Whisper on Raspberry Pi)
    language = language || this.detectLanguage(query);
    await new Promise(resolve => setTimeout(resolve, 250));
    this.ui.setTranscript(query, 'user', false);
    // 2. Generate Intelligent Local Response
    const reply = await this.generateResponse(query, language);

    // 3. Switch to Speaking state and mute mic recognition until speech finishes (prevents echo loop)
    this.isAssistantSpeaking = true;
    this.stopRecognition();
    this.rollingSamples16k.fill(0);
    this.rollingFilled = 0;

    let resumed = false;
    const finishSpeaking = () => {
      if (resumed) return;
      resumed = true;
      this.isAssistantSpeaking = false;
      // 900ms post-speech acoustic echo guard
      this.ignoreInputUntil = performance.now() + 900;
      this.rollingSamples16k.fill(0);
      this.rollingFilled = 0;
      this.returnToSleep();
      setTimeout(() => {
        if (this.isMicActive && !this.isAssistantSpeaking) {
          this.startRecognition();
        }
      }, 950);
    };

    // Safety ceiling in case browser never fires onend
    const maxSpeechMs = Math.max(4000, reply.length * 85 + 2000);
    const safetyTimer = setTimeout(finishSpeaking, maxSpeechMs);

    if (this.ui) {
      this.ui.setState('speaking');
      this.ui.setTranscript(reply, 'bis', true);
      this.ui.updateSubtext(`✨ ${this.escapeHtml(reply)}`);
      this.ui.speakAudio(reply, language, () => {
        clearTimeout(safetyTimer);
        finishSpeaking();
      });
    } else {
      clearTimeout(safetyTimer);
      finishSpeaking();
    }
  }

  returnToSleep() {
    this.isAwake = false;
    this.accumulatedQuery = '';
    if (this.silenceTimer) clearTimeout(this.silenceTimer);

    if (this.ui && this.ui.state !== 'idle') {
      this.ui.setState('idle');
      this.ui.updateSubtext('<span>🟢 Micrófono activo</span> <span class="dot"></span> <span>Di "Hey Bis" o "Hola Bis"</span>');
    }
  }

  // ==========================================================================
  // NATURAL LOCAL INTELLIGENCE ENGINE (Instant answers without delay)
  // ==========================================================================
  async generateResponse(query, language = this.detectLanguage(query)) {
    const q = query.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '');
    const english = language === 'en-US';
    const now = new Date();

    // Time query
    if (q.includes('hora') || q.includes('time')) {
      const timeStr = now.toLocaleTimeString(language, { hour: '2-digit', minute: '2-digit' });
      return english ? `It's ${timeStr}.` : `Son las ${timeStr}.`;
    }

    // Date query
    if (q.includes('fecha') || q.includes('dia') || q.includes('date') || q.includes('day')) {
      const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
      return english ? `Today is ${now.toLocaleDateString(language, options)}.` : `Hoy es ${now.toLocaleDateString('es-MX', options)}.`;
    }

    // Weather query
    if (q.includes('clima') || q.includes('tiempo') || q.includes('weather') || q.includes('temperatura')) {
      return english ? 'I need a weather service connection to check the current weather.' : 'Necesito conectarme a un servicio del clima para consultar el tiempo actual.';
    }

    // System diagnostics
    if (q.includes('sistema') || q.includes('estado') || q.includes('diagnostico') || q.includes('status') || q.includes('diagnostic') || q.includes('system')) {
      return english ? 'I’m running and receiving your voice. The dashboard shows the model status.' : 'Estoy activo y recibo tu voz. Puedes consultar el estado del modelo en el dashboard.';
    }

    // Greetings
    if (q.includes('hola') || q.includes('como estas') || q.includes('hello') || q.includes('how are you')) {
      return english ? 'Hi! I’m listening. How can I help you?' : '¡Hola! Aquí estoy. ¿En qué te puedo ayudar?';
    }

    // Music
    if (q.includes('musica') || q.includes('music') || q.includes('play') || q.includes('cancion')) {
      return english ? 'Music playback has not been connected yet.' : 'La reproducción de música todavía está pendiente de conectar.';
    }

    // Calculations / Math
    const mathMatch = q.match(/(?:cuanto es|what is|what's|calculate)\s+(\d+)\s*([\+\-\*\/]|mas|menos|por|entre|plus|minus|times|divided by)\s*(\d+)/i);
    if (mathMatch) {
      const n1 = parseFloat(mathMatch[1]);
      const op = mathMatch[2];
      const n2 = parseFloat(mathMatch[3]);
      let res = 0;
      if (['+', 'mas', 'plus'].includes(op)) res = n1 + n2;
      else if (['-', 'menos', 'minus'].includes(op)) res = n1 - n2;
      else if (['*', 'por', 'times'].includes(op)) res = n1 * n2;
      else res = n2 !== 0 ? (n1 / n2) : (english ? 'undefined' : 'indefinido');
      return english ? `The result is ${res}.` : `El resultado es ${res}.`;
    }

    // Default conversational response
    return english ? `I heard: "${query}". I don't have an answer for that request yet.` : `Escuché: "${query}". Todavía no tengo una respuesta para esa petición.`;
  }

  // ==========================================================================
  // SYNTHESIZED CYBER WAKE CHIME (Zero external audio files required)
  // ==========================================================================
  playWakeChime() {
    if (!this.audioCtx) return;
    try {
      const now = this.audioCtx.currentTime;
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(587.33, now); // D5
      osc.frequency.exponentialRampToValueAtTime(880.00, now + 0.12); // A5
      osc.frequency.exponentialRampToValueAtTime(1174.66, now + 0.22); // D6

      gain.gain.setValueAtTime(0.001, now);
      gain.gain.linearRampToValueAtTime(0.18, now + 0.05);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.35);

      osc.connect(gain);
      gain.connect(this.audioCtx.destination);

      osc.start(now);
      osc.stop(now + 0.36);
    } catch (e) {}
  }

  // ==========================================================================
  // UI STATUS BADGE
  // ==========================================================================
  updateStatusBadge(isActive, label) {
    if (this.micBadge) {
      this.micBadge.textContent = label || (isActive ? 'MIC ON // ATENTO A "HEY BIS"' : 'MIC EN ESPERA (CLICK)');
      if (isActive) {
        this.micBadge.style.color = 'var(--neon-green)';
        this.micBadge.style.borderColor = 'rgba(0, 255, 136, 0.45)';
        this.micBadge.style.background = 'rgba(0, 255, 136, 0.15)';
        this.micBadge.style.boxShadow = '0 0 12px rgba(0, 255, 136, 0.3)';
        this.micBadge.style.cursor = 'default';
      } else {
        this.micBadge.style.color = '#f59e0b';
        this.micBadge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
        this.micBadge.style.background = 'rgba(245, 158, 11, 0.12)';
        this.micBadge.style.boxShadow = '';
        this.micBadge.style.cursor = 'pointer';
      }
    }

    if (this.micBtn) {
      this.micBtn.textContent = isActive ? 'MIC: ONLINE (HEY BIS)' : 'ACTIVAR MIC';
      this.micBtn.style.background = isActive ? 'rgba(0, 255, 136, 0.25)' : '';
      this.micBtn.style.borderColor = isActive ? 'var(--neon-green)' : '';
      this.micBtn.style.color = isActive ? '#fff' : '';
    }
  }

  escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
}

// Export to window
window.AudioReactive = AudioReactive;
