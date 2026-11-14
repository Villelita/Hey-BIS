/**
 * ============================================================================
 * BIS VOICE ASSISTANT — API & WEBSOCKET BRIDGE
 * ============================================================================
 * Exposes a clean JavaScript API and WebSocket client to interface with
 * Raspberry Pi Python backend (e.g. Vosk, Whisper, Porcupine, Piper TTS).
 * ============================================================================
 */

class BisAPIBridge {
  constructor(ui, core) {
    this.ui = ui;
    this.core = core;
    this.ws = null;
    this.wsUrl = 'ws://localhost:8765';
    this.reconnectAttempts = 0;
    this.maxReconnectDelay = 10000;
    this.reconnectTimer = null;

    this.exposeGlobalAPI();
    this.checkAutoConnect();
  }

  // ==========================================================================
  // GLOBAL JAVASCRIPT API (window.Bis)
  // ==========================================================================
  exposeGlobalAPI() {
    window.Bis = {
      // Set assistant state: 'idle' | 'listening' | 'thinking' | 'speaking' | 'error'
      setState: (state, subtext) => {
        this.ui.setState(state, { updateSubtext: subtext });
        this.emit('state_change', { state, subtext });
      },

      // Update transcription from user
      transcribeUser: (text) => {
        this.ui.setState('listening');
        this.ui.setTranscript(`"${text}"`, 'user', true);
        this.emit('transcribe_user', { text });
      },

      // Assistant speaks response
      speak: (text) => {
        this.ui.setState('speaking');
        this.ui.setTranscript(text, 'bis', true);
        this.ui.speakAudio(text);
        this.emit('speak', { text });
      },

      // Set palette: 'green' | 'cyberpunk'
      setPalette: (palette) => {
        if (this.core) this.core.setPalette(palette);
      },

      // Direct audio energy pulse (0.0 to 1.0)
      setAudioLevel: (level) => {
        if (this.core) this.core.setAudioLevel(level);
      },

      // Set subtext / prompt hint
      setSubtext: (text) => {
        this.ui.updateSubtext(text);
      },

      // Connect to WebSocket backend
      connectWebSocket: (url) => {
        this.connect(url);
      },

      // Get current state
      getState: () => this.ui.state
    };
  }

  // ==========================================================================
  // WEBSOCKET COMMUNICATION (Raspberry Pi Backend Integration)
  // ==========================================================================
  checkAutoConnect() {
    const urlParams = new URLSearchParams(window.location.search);
    const wsParam = urlParams.get('ws');

    // If explicit ?ws=ws://... was provided, or if user enables auto-connect
    if (wsParam) {
      this.connect(wsParam);
    } else if (urlParams.has('connect') || window.location.protocol.startsWith('http')) {
      // Optional auto-attempt on local server
      this.connect(this.wsUrl);
    }
  }

  connect(url = this.wsUrl) {
    this.wsUrl = url;
    if (this.ws) {
      try { this.ws.close(); } catch (e) {}
    }

    try {
      this.ws = new WebSocket(this.wsUrl);

      this.ws.onopen = () => {
        console.log(`[Bis Bridge] Connected to WebSocket backend: ${this.wsUrl}`);
        this.reconnectAttempts = 0;
        this.updateConnectionStatus(true);
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(event.data);
      };

      this.ws.onerror = (err) => {
        console.warn('[Bis Bridge] WebSocket connection error:', err);
      };

      this.ws.onclose = () => {
        this.updateConnectionStatus(false);
        this.scheduleReconnect();
      };
    } catch (e) {
      this.updateConnectionStatus(false);
      this.scheduleReconnect();
    }
  }

  scheduleReconnect() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    const delay = Math.min(1000 * Math.pow(1.5, this.reconnectAttempts), this.maxReconnectDelay);
    this.reconnectAttempts++;
    this.reconnectTimer = setTimeout(() => {
      this.connect(this.wsUrl);
    }, delay);
  }

  updateConnectionStatus(isConnected) {
    const badge = document.getElementById('kiosk-badge');
    const dot = document.querySelector('.core-pulse-indicator');
    if (badge) {
      badge.textContent = isConnected ? 'PI BACKEND // CONNECTED' : 'LOCAL KIOSK // 60 FPS';
    }
    if (dot) {
      dot.style.background = isConnected ? 'var(--neon-emerald)' : 'var(--neon-cyan)';
      dot.style.boxShadow = isConnected ? '0 0 14px var(--neon-emerald)' : '0 0 14px var(--neon-cyan)';
    }
  }

  // Handle incoming commands from Python backend
  handleMessage(dataStr) {
    try {
      const msg = JSON.parse(dataStr);
      const action = msg.action || msg.type || msg.event;

      switch (action) {
        case 'state':
          window.Bis.setState(msg.value || msg.state, msg.subtext);
          break;

        case 'user_speech':
        case 'user_transcript':
          window.Bis.transcribeUser(msg.text || msg.transcript);
          break;

        case 'speak':
        case 'reply':
          window.Bis.speak(msg.text || msg.reply);
          break;

        case 'audio_level':
          window.Bis.setAudioLevel(parseFloat(msg.level || msg.value || 0));
          break;

        case 'subtext':
          window.Bis.setSubtext(msg.text);
          break;

        default:
          console.log('[Bis Bridge] Unknown command:', msg);
      }
    } catch (err) {
      console.error('[Bis Bridge] Error parsing message:', err, dataStr);
    }
  }

  emit(eventName, detail) {
    window.dispatchEvent(new CustomEvent(`bis:${eventName}`, { detail }));
  }
}

// Export to window
window.BisAPIBridge = BisAPIBridge;
