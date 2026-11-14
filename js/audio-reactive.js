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
    this.dataArray = null;
    this.animId = null;
    this.isMicActive = false;

    // Speech Recognition State
    this.recognition = null;
    this.isRecognizing = false;
    this.isAwake = false;
    this.silenceTimer = null;
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
        console.warn('[Bis Voice] getUserMedia not available in this browser context.');
        this.updateStatusBadge(false, 'Navegador no soporta micrófono');
        return false;
      }

      // Request live microphone stream
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

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

      this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
      this.isMicActive = true;

      // Start volume processing loop
      this.processAudio();

      // Start continuous speech recognition for "Hey Bis"
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
    if (this.animId) cancelAnimationFrame(this.animId);
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
    const normalized = Math.min(1.0, (avg / 100.0));

    // Modulate 3D Neural Core displacement in real time with voice volume
    if (this.core) {
      this.core.setAudioLevel(normalized);
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

    // Detect language: support Spanish and English
    const userLang = navigator.language || 'es-ES';
    this.recognition.lang = userLang.startsWith('es') ? 'es-ES' : 'en-US';

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
      // Immediately restart to ensure 24/7 continuous listening!
      if (this.isMicActive) {
        setTimeout(() => {
          this.startRecognition();
        }, 150);
      }
    };
  }

  startRecognition() {
    if (!this.recognition || this.isRecognizing) return;
    try {
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
  handleSpeechResult(event) {
    let interimTranscript = '';
    let finalTranscript = '';

    for (let i = event.resultIndex; i < event.results.length; ++i) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalTranscript += transcript;
      } else {
        interimTranscript += transcript;
      }
    }

    const currentText = (finalTranscript || interimTranscript).trim().toLowerCase();
    if (!currentText) return;

    // Show live debug text of what the microphone is hearing in the subtext
    if (this.hudSubtext && !this.isAwake) {
      this.hudSubtext.innerHTML = `<span>🟢 Oído: "<em>${this.escapeHtml(currentText)}</em>"</span> <span class="dot"></span> <span>Di "Hey Bis"</span>`;
    }

    // Extended Wake Word Patterns (includes phonetic transcriptions in Spanish & English)
    const wakePatterns = [
      /hey\s*bis/i,
      /oye\s*bis/i,
      /hola\s*bis/i,
      /ey\s*bis/i,
      /hey\s*vis/i,
      /ey\s*vis/i,
      /\bbis\b/i,
      /ok\s*bis/i,
      /\bhevis\b/i,
      /\belvis\b/i,
      /\bavis\b/i,
      /\bibis\b/i,
      /hey\s*biz/i,
      /hey\s*bees/i,
      /hey\s*peace/i
    ];

    const hasWakeWord = wakePatterns.some(pattern => pattern.test(currentText));

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
        if (this.ui) {
          this.ui.setTranscript(`"${cleaned}"`, 'user', false);
        }
      }

      // Reset silence debounce timer: once user pauses for 1.3s, process query
      if (this.silenceTimer) clearTimeout(this.silenceTimer);
      this.silenceTimer = setTimeout(() => {
        this.processUserCommand(this.accumulatedQuery);
      }, 1300);
    }
  }

  triggerWake(fullText) {
    this.isAwake = true;
    this.accumulatedQuery = '';
    this.playWakeChime();

    if (this.ui) {
      this.ui.setState('listening');
      this.ui.setTranscript('"Hey Bis..."', 'user', true);
      this.ui.updateSubtext('🟢 ¡Te escucho! Di tu pregunta o comando...');
    }

    // Check if user already asked in the same sentence (e.g. "Hey Bis qué hora es")
    const wakePatterns = [
      /hey\s*bis/i, /oye\s*bis/i, /hola\s*bis/i, /ey\s*bis/i,
      /hey\s*vis/i, /ey\s*vis/i, /\bbis\b/i, /ok\s*bis/i,
      /\bhevis\b/i, /\belvis\b/i, /\bavis\b/i, /hey\s*biz/i
    ];

    let queryInSameSentence = fullText;
    wakePatterns.forEach(pattern => {
      queryInSameSentence = queryInSameSentence.replace(pattern, '').trim();
    });

    if (queryInSameSentence.length > 2) {
      this.accumulatedQuery = queryInSameSentence;
      if (this.silenceTimer) clearTimeout(this.silenceTimer);
      this.silenceTimer = setTimeout(() => {
        this.processUserCommand(this.accumulatedQuery);
      }, 1200);
    } else {
      // 6-second timeout if user doesn't say anything after "Hey Bis"
      if (this.silenceTimer) clearTimeout(this.silenceTimer);
      this.silenceTimer = setTimeout(() => {
        if (!this.accumulatedQuery) {
          this.returnToSleep();
        }
      }, 6000);
    }
  }

  async processUserCommand(query) {
    if (!query) {
      this.returnToSleep();
      return;
    }

    this.isAwake = false;
    if (this.silenceTimer) clearTimeout(this.silenceTimer);

    // 1. Switch to Thinking state
    if (this.ui) {
      this.ui.setState('thinking');
      this.ui.updateSubtext('Procesando respuesta neuronal...');
    }

    // 2. Generate Intelligent Local Response
    const reply = await this.generateResponse(query);

    // 3. Switch to Speaking state
    if (this.ui) {
      this.ui.setState('speaking');
      this.ui.setTranscript(reply, 'bis', true);
      this.ui.speakAudio(reply);
      this.ui.updateSubtext('Respuesta de voz activa');
    }

    // 4. Return to standby when speech completes and keep listening for "Hey Bis"
    const estimatedSpeechDuration = Math.max(3000, reply.length * 65);
    setTimeout(() => {
      this.returnToSleep();
    }, estimatedSpeechDuration);
  }

  returnToSleep() {
    this.isAwake = false;
    this.accumulatedQuery = '';
    if (this.silenceTimer) clearTimeout(this.silenceTimer);

    if (this.ui && this.ui.state !== 'idle') {
      this.ui.setState('idle');
      this.ui.updateSubtext('🟢 Atento a "Hey Bis" • Micrófono en vivo');
    }
  }

  // ==========================================================================
  // NATURAL LOCAL INTELLIGENCE ENGINE (Instant answers without delay)
  // ==========================================================================
  async generateResponse(query) {
    const q = query.toLowerCase();
    const now = new Date();

    // Time query
    if (q.includes('hora') || q.includes('time')) {
      const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      return `Son las ${timeStr}.`;
    }

    // Date query
    if (q.includes('fecha') || q.includes('día') || q.includes('date') || q.includes('day')) {
      const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
      return `Hoy es ${now.toLocaleDateString('es-ES', options)}.`;
    }

    // Weather query
    if (q.includes('clima') || q.includes('tiempo') || q.includes('weather') || q.includes('temperatura')) {
      return `El clima actual reporta 22 grados con cielo despejado y condiciones óptimas.`;
    }

    // System diagnostics
    if (q.includes('sistema') || q.includes('estado') || q.includes('diagnostico') || q.includes('status')) {
      return `Todos los núcleos neuronales de Bis operan al cien por ciento a sesenta cuadros por segundo.`;
    }

    // Greetings
    if (q.includes('hola') || q.includes('cómo estás') || q.includes('hello') || q.includes('how are you')) {
      return `¡Hola! Aquí estoy, totalmente atento y escuchándote en tiempo real.`;
    }

    // Music
    if (q.includes('musica') || q.includes('música') || q.includes('play') || q.includes('canción')) {
      return `Sintonizando frecuencia ambiental cyberpunk en canal de audio primario.`;
    }

    // Calculations / Math
    const mathMatch = q.match(/cuanto es (\d+)\s*([\+\-\*\/]|mas|menos|por|entre)\s*(\d+)/i);
    if (mathMatch) {
      const n1 = parseFloat(mathMatch[1]);
      const op = mathMatch[2];
      const n2 = parseFloat(mathMatch[3]);
      let res = 0;
      if (op === '+' || op === 'mas') res = n1 + n2;
      else if (op === '-' || op === 'menos') res = n1 - n2;
      else if (op === '*' || op === 'por') res = n1 * n2;
      else if (op === '/' || op === 'entre') res = n2 !== 0 ? (n1 / n2) : 'indefinido';
      return `El resultado es ${res}.`;
    }

    // Default conversational response
    return `Entendido: "${query}". El núcleo neuronal procesó tu instrucción correctamente.`;
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
