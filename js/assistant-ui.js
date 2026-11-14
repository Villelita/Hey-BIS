/**
 * ============================================================================
 * BIS VOICE ASSISTANT — USER INTERFACE MANAGER
 * ============================================================================
 * Coordinates the glassmorphism HUD, real-time transcription streamer,
 * telemetry indicators, clock, state visualizers, and interactive controls.
 * ============================================================================
 */

class AssistantUI {
  constructor(options = {}) {
    this.core = options.core || null;
    this.state = 'idle'; // 'idle', 'listening', 'thinking', 'speaking', 'error'
    this.typewriterInterval = null;
    this.audioMeterInterval = null;

    // Cache DOM Elements
    this.dom = {
      body: document.body,
      statePill: document.getElementById('state-pill'),
      stateLabel: document.getElementById('state-label'),
      stateDot: document.getElementById('state-dot'),
      clockValue: document.getElementById('clock-value'),
      speakerName: document.getElementById('speaker-name'),
      speakerAvatar: document.getElementById('speaker-avatar'),
      transcriptionText: document.getElementById('transcription-text'),
      hudTimestamp: document.getElementById('hud-timestamp'),
      hudSubtext: document.getElementById('hud-subtext'),
      devPanel: document.getElementById('dev-panel'),
      devToggleBtn: document.getElementById('dev-toggle-btn'),
      cardBars: document.querySelectorAll('.card-spectrum-bars .bar'),
      pillBars: document.querySelectorAll('.state-wave-bars .wave-bar'),
      promptChips: document.querySelectorAll('.prompt-chip'),
      kioskBadge: document.getElementById('kiosk-badge')
    };

    this.initClock();
    this.initControls();
    this.initPrompts();
    this.initManualMicBtn();
    this.startAudioVisualizerLoop();
  }

  initManualMicBtn() {
    const btn = document.getElementById('btn-manual-mic');
    if (btn) {
      btn.addEventListener('click', () => {
        if (window.app && window.app.audio && !window.app.audio.isMicActive) {
          window.app.audio.start();
        }
        if (this.state === 'idle') {
          this.setState('listening');
          this.setTranscript('"Hey Bis..."', 'user', true);
          this.updateSubtext('🟢 Micrófono escuchando... Di tu pregunta');
        } else {
          this.setState('idle');
        }
      });
    }
  }

  // ==========================================================================
  // REAL-TIME SYSTEM CLOCK
  // ==========================================================================
  initClock() {
    const update = () => {
      const now = new Date();
      const hours = String(now.getHours()).padStart(2, '0');
      const minutes = String(now.getMinutes()).padStart(2, '0');
      const seconds = String(now.getSeconds()).padStart(2, '0');
      if (this.dom.clockValue) {
        this.dom.clockValue.textContent = `${hours}:${minutes}:${seconds}`;
      }
    };
    update();
    setInterval(update, 1000);
  }

  // ==========================================================================
  // STATE MANAGEMENT
  // ==========================================================================
  setState(newState, options = {}) {
    this.state = newState;
    this.dom.body.setAttribute('data-state', newState);

    const labels = {
      idle: 'STANDBY // READY',
      listening: 'LISTENING...',
      thinking: 'THINKING...',
      speaking: 'SPEAKING...',
      error: 'ERROR // RECONNECTING'
    };

    if (this.dom.stateLabel) {
      this.dom.stateLabel.textContent = labels[newState] || newState.toUpperCase();
    }

    // Sync with 3D Neural Core
    if (this.core) {
      this.core.setState(newState);
    }

    // Update dev buttons active state
    document.querySelectorAll('.dev-btn[data-state]').forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-state') === newState);
    });

    if (options.updateSubtext) {
      this.updateSubtext(options.updateSubtext);
    }
  }

  updateSubtext(text) {
    if (this.dom.hudSubtext) {
      this.dom.hudSubtext.innerHTML = text;
    }
  }

  // ==========================================================================
  // SPEECH SYNTHESIS HELPER (Optional local browser audio)
  // ==========================================================================
  speakAudio(text) {
    if (!('speechSynthesis' in window)) return;
    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.05;
      utterance.pitch = 0.95;
      const voices = window.speechSynthesis.getVoices();
      const preferred = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Samantha')));
      if (preferred) utterance.voice = preferred;
      window.speechSynthesis.speak(utterance);
    } catch (e) {
      // Ignored if user hasn't interacted or speech synth is disabled
    }
  }

  // ==========================================================================
  // TRANSCRIPTION STREAMING & TYPEWRITER EFFECT
  // ==========================================================================
  setTranscript(text, speaker = 'bis', animate = true) {
    if (this.typewriterInterval) {
      clearInterval(this.typewriterInterval);
      this.typewriterInterval = null;
    }

    // Update speaker indicator
    const isUser = speaker.toLowerCase() === 'user';
    if (this.dom.speakerName) {
      this.dom.speakerName.textContent = isUser ? 'User' : 'Bis';
      this.dom.speakerName.style.color = isUser ? 'var(--neon-cyan)' : 'var(--neon-magenta)';
    }

    if (this.dom.speakerAvatar) {
      this.dom.speakerAvatar.textContent = isUser ? 'U' : 'B';
      this.dom.speakerAvatar.style.borderColor = isUser ? 'rgba(0, 240, 255, 0.4)' : 'rgba(255, 0, 127, 0.4)';
      this.dom.speakerAvatar.style.color = isUser ? 'var(--neon-cyan)' : 'var(--neon-magenta)';
      this.dom.speakerAvatar.style.background = isUser ? 'rgba(0, 240, 255, 0.12)' : 'rgba(255, 0, 127, 0.12)';
    }

    // Update timestamp
    if (this.dom.hudTimestamp) {
      const now = new Date();
      this.dom.hudTimestamp.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }

    const textEl = this.dom.transcriptionText;
    if (!textEl) return;

    if (!animate) {
      textEl.innerHTML = `${this.escapeHtml(text)}<span class="cursor"></span>`;
      return;
    }

    // Streaming Typewriter Effect
    let index = 0;
    textEl.innerHTML = '<span class="cursor"></span>';

    const speed = Math.max(12, Math.min(28, 400 / (text.length || 1)));

    this.typewriterInterval = setInterval(() => {
      index++;
      const current = text.slice(0, index);
      textEl.innerHTML = `${this.escapeHtml(current)}<span class="cursor"></span>`;

      if (index >= text.length) {
        clearInterval(this.typewriterInterval);
        this.typewriterInterval = null;
      }
    }, speed);
  }

  // ==========================================================================
  // AUDIO WAVEFORM SIMULATION & VISUALIZER BARS
  // ==========================================================================
  startAudioVisualizerLoop() {
    let phase = 0;

    this.audioMeterInterval = setInterval(() => {
      phase += 0.25;

      const isListening = this.state === 'listening';
      const isSpeaking = this.state === 'speaking';
      const isThinking = this.state === 'thinking';

      // Animate card spectrum bars
      if (this.dom.cardBars) {
        this.dom.cardBars.forEach((bar, i) => {
          let height = 3;
          if (isListening || isSpeaking) {
            const wave = Math.sin(phase + i * 0.7) * 0.5 + 0.5;
            const noise = Math.random() * 0.3;
            height = 4 + (wave + noise) * 12;
          } else if (isThinking) {
            height = 3 + Math.sin(phase * 1.5 + i) * 3;
          }
          bar.style.height = `${Math.round(height)}px`;
        });
      }

      // Animate top state pill bars
      if (this.dom.pillBars) {
        this.dom.pillBars.forEach((bar, i) => {
          let height = 3;
          if (isListening || isSpeaking) {
            const wave = Math.sin(phase * 1.2 + i * 0.9) * 0.5 + 0.5;
            height = 3 + wave * 9;
          }
          bar.style.height = `${Math.round(height)}px`;
        });
      }
    }, 50);
  }

  // ==========================================================================
  // PROMPTS & DEMO SCENARIOS
  // ==========================================================================
  initPrompts() {
    this.dom.promptChips.forEach(chip => {
      chip.addEventListener('click', () => {
        const query = chip.getAttribute('data-query');
        const reply = chip.getAttribute('data-reply');
        this.runDemoConversation(query, reply);
      });
    });
  }

  runDemoConversation(userQuery, assistantReply) {
    // 1. User is speaking / recognized
    this.setState('listening');
    this.setTranscript(`"${userQuery}"`, 'user', true);
    this.updateSubtext('Voice detected • Parsing neural intent...');

    // 2. Bis is thinking
    setTimeout(() => {
      this.setState('thinking');
      this.updateSubtext('Synthesizing reasoning & contextual knowledge...');
    }, 1800);

    // 3. Bis speaks reply
    setTimeout(() => {
      this.setState('speaking');
      this.setTranscript(assistantReply, 'bis', true);
      this.speakAudio(assistantReply);
      this.updateSubtext('Audio response streaming • Core sync active');

      // 4. Return to standby when finished
      const duration = Math.max(3000, assistantReply.length * 50);
      setTimeout(() => {
        if (this.state === 'speaking') {
          this.setState('idle');
          this.updateSubtext('Say "Hey Bis" or tap the screen to activate');
        }
      }, duration);
    }, 3400);
  }

  // ==========================================================================
  // DEV CONTROLS & SHORTCUTS
  // ==========================================================================
  initControls() {
    // Toggle dev panel
    if (this.dom.devToggleBtn && this.dom.devPanel) {
      this.dom.devToggleBtn.addEventListener('click', () => {
        this.dom.devPanel.classList.toggle('open');
      });

      const closeBtn = document.getElementById('dev-panel-close');
      if (closeBtn) {
        closeBtn.addEventListener('click', () => {
          this.dom.devPanel.classList.remove('open');
        });
      }
    }

    // State Switcher Buttons
    document.querySelectorAll('.dev-btn[data-state]').forEach(btn => {
      btn.addEventListener('click', () => {
        const state = btn.getAttribute('data-state');
        this.setState(state);
      });
    });

    // Color Palette Switcher (Green vs Cyberpunk)
    const btnGreen = document.getElementById('btn-palette-green');
    const btnCyber = document.getElementById('btn-palette-cyber');
    if (btnGreen && btnCyber && this.core) {
      btnGreen.addEventListener('click', () => {
        this.core.setPalette('green');
        btnGreen.style.background = 'rgba(0, 255, 136, 0.25)';
        btnGreen.style.borderColor = 'var(--neon-green)';
        btnGreen.style.color = '#fff';
        btnCyber.style.background = 'rgba(255, 0, 127, 0.1)';
        btnCyber.style.borderColor = 'rgba(255, 0, 127, 0.3)';
        btnCyber.style.color = '#ff80bf';
      });

      btnCyber.addEventListener('click', () => {
        this.core.setPalette('cyberpunk');
        btnCyber.style.background = 'rgba(255, 0, 127, 0.25)';
        btnCyber.style.borderColor = 'var(--neon-magenta)';
        btnCyber.style.color = '#fff';
        btnGreen.style.background = 'rgba(0, 255, 136, 0.1)';
        btnGreen.style.borderColor = 'rgba(0, 255, 136, 0.3)';
        btnGreen.style.color = '#a7f3d0';
      });
    }

    // Fullscreen Toggle (Crucial for Raspberry Pi kiosk)
    const fsBtn = document.getElementById('dev-btn-fullscreen');
    if (fsBtn) {
      fsBtn.addEventListener('click', () => {
        if (!document.fullscreenElement) {
          document.documentElement.requestFullscreen().catch(err => console.log(err));
        } else {
          document.exitFullscreen().catch(err => console.log(err));
        }
      });
    }

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      // Space: Toggle Listening
      if (e.code === 'Space' && e.target.tagName !== 'INPUT') {
        e.preventDefault();
        if (this.state === 'idle') {
          this.setState('listening');
          this.setTranscript('"Hey Bis..."', 'user', true);
        } else {
          this.setState('idle');
        }
      }

      // 1: Idle, 2: Listening, 3: Thinking, 4: Speaking, 5: Error
      if (e.key === '1') this.setState('idle');
      if (e.key === '2') this.setState('listening');
      if (e.key === '3') this.setState('thinking');
      if (e.key === '4') this.setState('speaking');
      if (e.key === '5') this.setState('error');

      // 'D': Toggle Dev Panel
      if (e.key.toLowerCase() === 'd' && e.target.tagName !== 'INPUT') {
        this.dom.devPanel.classList.toggle('open');
      }

      // 'F11' / 'F': Fullscreen
      if (e.key.toLowerCase() === 'f' && e.target.tagName !== 'INPUT') {
        if (!document.fullscreenElement) {
          document.documentElement.requestFullscreen();
        } else {
          document.exitFullscreen();
        }
      }
    });

    // Check URL parameters for kiosk auto-mode
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has('kiosk') || urlParams.get('mode') === 'kiosk') {
      if (this.dom.devToggleBtn) this.dom.devToggleBtn.style.display = 'none';
      if (this.dom.kioskBadge) this.dom.kioskBadge.textContent = 'KIOSK ACTIVE';
    }
  }

  escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
}

// Export to window
window.AssistantUI = AssistantUI;
