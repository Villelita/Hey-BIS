/**
 * ============================================================================
 * BIS VOICE ASSISTANT — MAIN APPLICATION ENTRY POINT
 * ============================================================================
 */

document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('neural-canvas');
  if (!canvas) {
    console.error('Canvas element #neural-canvas not found.');
    return;
  }

  // 1. Initialize 3D WebGL Neural Core
  const core = new NeuralCore(canvas, {
    detail: 3,             // 1,280 triangles (high quality, buttery 60fps on Pi)
    baseRadius: 1.6,
    particleCount: 220
  });

  // 2. Initialize Assistant UI Manager
  const ui = new AssistantUI({ core });

  // 3. Initialize Live Audio Reactivity & Continuous "Hey Bis" Voice Listener
  const audio = new AudioReactive(core, ui);

  // 4. Initialize API & WebSocket Bridge for Raspberry Pi
  const bridge = new BisAPIBridge(ui, core);

  // Expose instances for debugging / scripting
  window.app = { core, ui, audio, bridge };

  // Set initial welcome state
  ui.setState('idle');
  ui.setTranscript('Atento a la frase "Hey Bis" en vivo con tu micrófono...', 'bis', false);
  ui.updateSubtext('🟢 Micrófono activo • Di "Hey Bis" para hablar');

  console.log('%c[Bis]%c Neural Core & Kiosk Interface initialized successfully.', 'color: #00f0ff; font-weight: bold;', 'color: #f8fafc;');
});

