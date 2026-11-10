# Bis // Neural Voice Assistant Interface

A modern, minimalist, and futuristic web interface designed for a Raspberry Pi voice assistant named **"Bis"**.

Features a dark mode theme with an obsidian black / deep dark violet background, a real-time glowing 3D neural core (deformed wireframe icosphere with cyan-blue and magenta-pink neon lighting, surrounded by a field of floating synaptic particles), and a sleek glassmorphism HUD card for real-time speech transcription.

---

## 🌟 Key Highlights

- **Pure WebGL 3D Neural Core (0 Dependencies)**: Custom vertex and fragment shaders featuring 3D Simplex noise deformation, barycentric wireframe antialiasing, exponential neon aura falloff, and dual-gradient cyan (`#00f0ff`) to magenta (`#ff007f`) lighting.
- **Buttery 60 FPS on Raspberry Pi**: Hardware accelerated with only ~1,280 triangles and 220 floating synapse particles. Requires zero heavy post-processing passes (e.g., UnrealBloomPass), ensuring consistent 60 FPS performance on Raspberry Pi 3, 4, and 5.
- **Organic Breathing & Dynamic State Transitions**: Smooth easing between assistant states:
  - `Standby (Idle)`: Gentle organic breathing rhythm, slow rotation, soft ambient cyan/violet aura.
  - `Listening`: Electric cyan surge, reactive audio ripples responding to microphone frequencies.
  - `Thinking`: Accelerated rotation vortex, turbulent noise waves, dominant magenta/purple glow.
  - `Speaking`: Harmonic acoustic pulsing with rhythmic expansion, synchronized visualizer bars.
  - `Error`: Amber pulse with alert vibration.
- **Glassmorphism Transcription HUD**: Frosted glass card with backdrop blur, dual-gradient top shimmer, speaker tags, real-time typewriter text animation, and live spectrum bars.
- **Audio Reactive Engine**: Integrates Web Audio API (`AnalyserNode`) to capture live microphone input and pulse the core in real time.
- **Raspberry Pi Integration Bridge**: Includes a ready-to-use WebSocket client and public JavaScript API (`window.Bis`), along with a companion Python backend service.

---

## 📁 Project Architecture

```
hey bis/
├── index.html                 # Main kiosk voice assistant user interface
├── css/
│   ├── style.css              # Cyberpunk dark mode & glassmorphism theme
│   └── fonts.css              # Typography definitions with instant offline fallbacks
├── js/
│   ├── main.js                # Application bootloader
│   ├── neural-core.js         # Pure WebGL 3D deformed wireframe mesh & synapse particles
│   ├── assistant-ui.js        # HUD manager, typewriter animator, and telemetry clock
│   ├── audio-reactive.js      # Web Audio API microphone frequency & volume analyzer
│   └── api-bridge.js          # Raspberry Pi WebSocket client & window.Bis global API
├── python/
│   ├── assistant_backend.py   # Python WebSocket daemon & automated conversation loop
│   └── requirements.txt       # Python dependencies (websockets)
├── scripts/
│   ├── start-kiosk.sh         # Raspberry Pi Chromium kiosk launch script
│   └── bis-kiosk.service      # Systemd service for auto-start on boot
└── README.md                  # Comprehensive documentation
```

---

## 🚀 Quick Start

### Option 1: Open Directly in Any Browser
Simply double-click or open `index.html` in Chrome, Chromium, Edge, or Firefox. No build step, no npm, and no local server required!

### Option 2: Run with Local HTTP Server (Optional)
```bash
# Using Node:
npx serve .

# Or using Python:
python3 -m http.server 8000
```
Open `http://localhost:8000` in your browser.

---

## 🍓 Raspberry Pi Kiosk Mode Setup

### 1. Prerequisites on Raspberry Pi OS
Install Chromium and unclutter (to hide the mouse cursor):
```bash
sudo apt update
sudo apt install -y chromium-browser unclutter
```

### 2. Test Kiosk Launch
Run the included launch script:
```bash
chmod +x scripts/start-kiosk.sh
./scripts/start-kiosk.sh
```

### 3. Autostart on Boot (Systemd)
To have Bis launch automatically when your Raspberry Pi boots up:
```bash
# Copy service file to systemd directory
sudo cp scripts/bis-kiosk.service /etc/systemd/system/

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable bis-kiosk.service
sudo systemctl start bis-kiosk.service
```

---

## 🔌 Connecting to Your Voice Pipeline (Python Backend)

Bis includes an API bridge that connects to your Python voice assistant (e.g., Vosk / Whisper STT, Porcupine Wakeword, Piper / Coqui TTS, or Ollama / OpenAI / Gemini).

### 1. Start the Python Bridge Server
```bash
pip install -r python/requirements.txt
python3 python/assistant_backend.py
```

### 2. Protocol Specification (JSON via WebSocket)
Send standard JSON messages to `ws://localhost:8765`:

#### Change State:
```json
{
  "action": "state",
  "value": "listening",
  "subtext": "Wakeword detected"
}
```
*(States: `"idle"`, `"listening"`, `"thinking"`, `"speaking"`, `"error"`)*

#### Send Transcribed User Speech:
```json
{
  "action": "user_speech",
  "text": "What is the temperature outside?"
}
```

#### Make Bis Speak & Stream Response:
```json
{
  "action": "speak",
  "text": "The current outdoor temperature is 21 degrees Celsius with clear skies."
}
```

#### Feed Live Audio Level:
```json
{
  "action": "audio_level",
  "level": 0.85
}
```

---

## 💻 JavaScript API (`window.Bis`)

When running inside the browser or webview, you can directly control Bis from JavaScript:

```javascript
// Change state
window.Bis.setState('listening', 'Microphone active');

// Update user query
window.Bis.transcribeUser('Hey Bis, turn on the studio lights');

// Stream assistant reply
window.Bis.speak('Studio lights activated. Neural link verified.');

// Update audio pulse level (0.0 to 1.0)
window.Bis.setAudioLevel(0.75);

// Check current state
console.log(window.Bis.getState()); // 'speaking'
```

---

## ⌨️ Interactive Controls & Keyboard Shortcuts

- **Spacebar**: Toggle listening mode (`"Hey Bis..."`).
- **Keys `1`, `2`, `3`, `4`, `5`**:
  - `1`: Standby (Idle)
  - `2`: Listening
  - `3`: Thinking
  - `4`: Speaking
  - `5`: Error
- **Key `D`**: Open/close the Developer & State Simulator Dock (bottom right).
- **Key `F`**: Toggle Fullscreen mode.
- **Drag / Touch**: Rotate the 3D neural core in 3D space with smooth inertia.
- **Suggestion Chips**: Click or tap any prompt chip at the bottom to trigger a realistic voice interaction simulation.

