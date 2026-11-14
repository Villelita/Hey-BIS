#!/usr/bin/env python3
"""
==============================================================================
BIS VOICE ASSISTANT — RASPBERRY PI PYTHON BACKEND BRIDGE
==============================================================================
Provides:
  1. A WebSocket server on port 8765 to control the Bis neural kiosk interface.
  2. Continuous live microphone listening for the wake word "Hey Bis" using
     Python SpeechRecognition (or PyAudio / Vosk / Whisper).
  3. Automatic dispatch of states (idle, listening, thinking, speaking) and
     real-time transcription back to the frontend.

Usage:
  python3 assistant_backend.py
==============================================================================
"""

import asyncio
import json
import logging
import re
import threading
import time

try:
    import websockets
except ImportError:
    print("[Bis Backend] 'websockets' library is required. Run: pip install websockets")
    exit(1)

# Optional local microphone recognition in Python
HAVE_SR = False
try:
    import speech_recognition as sr
    HAVE_SR = True
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Set of connected frontend clients (Chromium kiosk browser)
CLIENTS = set()
MAIN_LOOP = None

async def broadcast(message_dict):
    """Broadcast JSON message to all connected frontend clients."""
    if not CLIENTS:
        return
    payload = json.dumps(message_dict)
    dead_clients = set()
    for ws in CLIENTS:
        try:
            await ws.send(payload)
        except Exception:
            dead_clients.add(ws)
    CLIENTS.difference_update(dead_clients)

# Public Voice Assistant State Dispatchers
async def set_state(state: str, subtext: str = ""):
    """Set state: 'idle' | 'listening' | 'thinking' | 'speaking' | 'error'"""
    await broadcast({"action": "state", "value": state, "subtext": subtext})

async def user_spoke(text: str):
    """Send transcribed speech from the user"""
    await broadcast({"action": "user_speech", "text": text})

async def bis_reply(text: str):
    """Send spoken response from Bis"""
    await broadcast({"action": "speak", "text": text})

async def set_audio_level(level: float):
    """Set dynamic audio meter (0.0 to 1.0)"""
    await broadcast({"action": "audio_level", "level": max(0.0, min(1.0, level))})


def thread_safe_broadcast(coro):
    """Schedule a coroutine in the main asyncio loop from a background thread."""
    if MAIN_LOOP and MAIN_LOOP.is_running():
        asyncio.run_coroutine_threadsafe(coro, MAIN_LOOP)


# ==============================================================================
# LIVE PYTHON MICROPHONE LISTENER (Continuous "Hey Bis" Wake Detection)
# ==============================================================================
def run_microphone_listener():
    """Continuously listens to the Raspberry Pi hardware microphone for 'Hey Bis'"""
    if not HAVE_SR:
        logging.info("[Bis Mic] 'speech_recognition' not installed. Frontend browser mic is active.")
        return

    try:
        r = sr.Recognizer()
        r.energy_threshold = 300
        r.dynamic_energy_threshold = True

        with sr.Microphone() as source:
            logging.info("[Bis Mic] Calibrating microphone for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=1.0)
            logging.info("[Bis Mic] Python Microphone ONLINE. Listening for 'Hey Bis' 24/7...")

            wake_regex = re.compile(r'\b(hey\s*bis|oye\s*bis|hola\s*bis|\bbis\b)\b', re.IGNORECASE)

            while True:
                try:
                    # Listen for audio chunk
                    audio = r.listen(source, timeout=None, phrase_time_limit=8)
                    try:
                        text = r.recognize_google(audio, language="es-ES")
                        logging.info(f"[Bis Mic] Recognized: '{text}'")

                        if wake_regex.search(text):
                            logging.info(">> WAKE PHRASE 'HEY BIS' DETECTED VIA PYTHON MIC! <<")
                            
                            # 1. Listening
                            thread_safe_broadcast(set_state("listening", "Atento a tu voz..."))
                            thread_safe_broadcast(user_spoke(text))
                            time.sleep(1.8)

                            # 2. Thinking
                            thread_safe_broadcast(set_state("thinking", "Procesando comando..."))
                            time.sleep(1.4)

                            # 3. Speaking
                            reply = f"Comando recibido: {text}. Núcleo neuronal Bis sincronizado."
                            thread_safe_broadcast(set_state("speaking", "Respuesta activa"))
                            thread_safe_broadcast(bis_reply(reply))
                            time.sleep(max(3.5, len(reply) * 0.065))

                            # 4. Standby
                            thread_safe_broadcast(set_state("idle", "Atento a 'Hey Bis'"))
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError as e:
                        logging.warning(f"[Bis Mic] Recognition service error: {e}")
                except Exception as e:
                    time.sleep(0.5)
    except Exception as e:
        logging.warning(f"[Bis Mic] Hardware microphone init failed: {e}. Browser mic remains active.")


async def handler(websocket):
    """Handle incoming WebSocket client connections from Chromium Kiosk UI."""
    CLIENTS.add(websocket)
    logging.info(f"[Bis Backend] Kiosk UI connected from {websocket.remote_address}")
    
    # Send initial welcome synchronization
    await websocket.send(json.dumps({
        "action": "state",
        "value": "idle",
        "subtext": "Atento a 'Hey Bis' // Micrófono activo"
    }))

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                logging.info(f"[Bis Backend] Received from frontend: {data}")
            except Exception as e:
                logging.warning(f"Failed to parse incoming message: {e}")
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        CLIENTS.remove(websocket)
        logging.info("[Bis Backend] Kiosk UI disconnected")


async def main():
    global MAIN_LOOP
    MAIN_LOOP = asyncio.get_running_loop()

    # Start hardware microphone listener thread if SpeechRecognition is available
    mic_thread = threading.Thread(target=run_microphone_listener, daemon=True)
    mic_thread.start()

    HOST = "0.0.0.0"
    PORT = 8765

    logging.info(f"[Bis Backend] WebSocket server online on ws://{HOST}:{PORT}")
    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future() # keep running forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("\n[Bis Backend] Shutting down.")
