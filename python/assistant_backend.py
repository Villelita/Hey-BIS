#!/usr/bin/env python3
"""
==============================================================================
BIS VOICE ASSISTANT — RASPBERRY PI PYTHON BACKEND BRIDGE
==============================================================================
Provides:
  1. A WebSocket server on port 8765 to control the Bis neural kiosk interface.
  2. Continuous live microphone listening for the wake word "Hey Bis" using
     SpeechRecognition and optional offline Vosk transcription.
  3. Automatic dispatch of states (idle, listening, thinking, speaking) and
     real-time transcription back to the frontend.

Usage:
  python3 assistant_backend.py
==============================================================================
"""

import argparse
import asyncio
import json
import logging
import re
import threading
import time
import unicodedata
from datetime import datetime
from pathlib import Path

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
MIC_READY = threading.Event()
MANUAL_LISTEN = threading.Event()
VOICE_STATE = {"action": "state", "value": "idle", "subtext": "Conectando micrófono..."}
WAKE_PATTERN = re.compile(r"\b(?:(?:hey|ey|oye|hola)\s*(?:bis|vis|biz|bees)|bis)\b", re.IGNORECASE)
WAKE_ONLY_PATTERN = re.compile(r"(?:hey|ey|oye|hola|bis|vis|biz|bees)(?:\s+(?:bis|vis|biz|bees))?", re.IGNORECASE)
COMMAND_START_TIMEOUT = 6
PHRASE_TIME_LIMIT = 3
IDLE_PROMPT = "Atento a 'Hey Bis'"

async def broadcast(message_dict):
    """Broadcast JSON message to all connected frontend clients."""
    if not CLIENTS:
        return
    payload = json.dumps(message_dict)
    dead_clients = set()
    for ws in tuple(CLIENTS):
        try:
            await ws.send(payload)
        except Exception:
            dead_clients.add(ws)
    CLIENTS.difference_update(dead_clients)

# Public Voice Assistant State Dispatchers
async def set_state(state: str, subtext: str = ""):
    """Set state: 'idle' | 'listening' | 'thinking' | 'speaking' | 'error'"""
    VOICE_STATE.update(value=state, subtext=subtext)
    await broadcast(VOICE_STATE)

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
    else:
        coro.close()


def capture_command(recognizer, source, transcribe):
    deadline = time.monotonic() + COMMAND_START_TIMEOUT
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            audio = recognizer.listen(source, timeout=remaining, phrase_time_limit=PHRASE_TIME_LIMIT)
            command = transcribe(audio).strip(" ,.!?¿¡")
        except sr.WaitTimeoutError:
            break
        except sr.UnknownValueError:
            continue
        if command and not WAKE_ONLY_PATTERN.fullmatch(command):
            repeated_wake = WAKE_PATTERN.match(command)
            if repeated_wake:
                command = command[repeated_wake.end():].lstrip(" ,.!?¿¡")
            if command:
                return command
    return ""


def answer_command(command):
    normalized = unicodedata.normalize("NFKD", command.casefold())
    query = "".join(char for char in normalized if not unicodedata.combining(char))
    now = datetime.now()
    if re.search(r"\b(que hora|dime la hora|what time)\b", query):
        return f"Son las {now:%H:%M}."
    if re.search(r"\b(que dia|que fecha|dime la fecha|what day)\b", query):
        months = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre")
        return f"Hoy es {now.day} de {months[now.month - 1]} de {now.year}."
    return f"Escuché: {command}."


# ==============================================================================
# LIVE PYTHON MICROPHONE LISTENER (Continuous "Hey Bis" Wake Detection)
# ==============================================================================
def run_microphone_listener(model_path=None, device_index=None):
    """Continuously listens to the Raspberry Pi hardware microphone for 'Hey Bis'"""
    if not HAVE_SR:
        logging.warning("[Bis Mic] SpeechRecognition is required for the backend microphone.")
        thread_safe_broadcast(set_state("idle", "Micrófono del backend no disponible"))
        return

    try:
        r = sr.Recognizer()
        r.energy_threshold = 300
        r.dynamic_energy_threshold = True
        if model_path is not None:
            from vosk import KaldiRecognizer, Model, SetLogLevel

            SetLogLevel(-1)
            model = Model(model_path=str(model_path))

            def transcribe(audio):
                recognizer = KaldiRecognizer(model, 16000)
                pcm = audio.get_raw_data(convert_rate=16000, convert_width=2)
                segments = []
                for offset in range(0, len(pcm), 8000):
                    if recognizer.AcceptWaveform(pcm[offset:offset + 8000]):
                        segments.append(json.loads(recognizer.Result())["text"])
                segments.append(json.loads(recognizer.FinalResult())["text"])
                return " ".join(segment for segment in segments if segment)
        else:
            def transcribe(audio):
                return r.recognize_google(audio, language="es-MX")

        with sr.Microphone(device_index=device_index) as source:
            logging.info("[Bis Mic] Calibrating microphone for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=1.0)
            logging.info("[Bis Mic] Python Microphone ONLINE. Listening for 'Hey Bis' 24/7...")

            MIC_READY.set()
            thread_safe_broadcast(set_state("idle", IDLE_PROMPT))

            while True:
                text = ""
                if not MANUAL_LISTEN.is_set():
                    try:
                        audio = r.listen(source, timeout=0.25, phrase_time_limit=PHRASE_TIME_LIMIT)
                        text = transcribe(audio)
                    except (sr.WaitTimeoutError, sr.UnknownValueError):
                        pass
                    except sr.RequestError as e:
                        logging.warning("[Bis Mic] Recognition service error: %s", e)
                        time.sleep(1)

                wake = WAKE_PATTERN.search(text)
                manual = MANUAL_LISTEN.is_set()
                if not (manual or wake):
                    continue

                MANUAL_LISTEN.clear()
                logging.info("[Bis Mic] Listening started by %s.", "button" if manual else "wake phrase")
                thread_safe_broadcast(set_state("listening", "Te escucho. Di tu pregunta..."))
                idle_prompt = IDLE_PROMPT
                try:
                    command = (text[wake.end():] if wake else text).strip(" ,.!?¿¡")
                    if not command:
                        command = capture_command(r, source, transcribe)
                    if not command:
                        logging.info("[Bis Mic] No intelligible command after activation.")
                        idle_prompt = "No entendí la pregunta. Di 'Hey Bis' para intentar de nuevo."
                        continue

                    logging.info("[Bis Mic] Command transcribed (%d words).", len(command.split()))
                    thread_safe_broadcast(user_spoke(command))
                    time.sleep(1.2)
                    thread_safe_broadcast(set_state("thinking", "Procesando comando..."))
                    time.sleep(0.5)
                    reply = answer_command(command)
                    thread_safe_broadcast(set_state("speaking", "Respuesta activa"))
                    thread_safe_broadcast(bis_reply(reply))
                    time.sleep(max(2, len(reply) * 0.065))
                except sr.RequestError as e:
                    logging.warning("[Bis Mic] Recognition service error: %s", e)
                finally:
                    MANUAL_LISTEN.clear()
                    thread_safe_broadcast(set_state("idle", idle_prompt))
    except Exception as e:
        MIC_READY.clear()
        logging.exception("[Bis Mic] Hardware microphone stopped: %s", e)
        thread_safe_broadcast(set_state("error", "Micrófono del backend no disponible"))


async def handler(websocket):
    """Handle incoming WebSocket client connections from Chromium Kiosk UI."""
    CLIENTS.add(websocket)
    logging.info(f"[Bis Backend] Kiosk UI connected from {websocket.remote_address}")
    
    try:
        await websocket.send(json.dumps(VOICE_STATE))
        async for message in websocket:
            try:
                data = json.loads(message)
                if isinstance(data, dict) and data.get("action") == "start_listening":
                    if not MIC_READY.is_set():
                        await websocket.send(json.dumps({"action": "subtext", "text": "Micrófono del backend no disponible"}))
                    elif VOICE_STATE["value"] == "idle":
                        MANUAL_LISTEN.set()
            except Exception as e:
                logging.warning(f"Failed to parse incoming message: {e}")
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        CLIENTS.discard(websocket)
        logging.info("[Bis Backend] Kiosk UI disconnected")


async def main(model_path=None, device_index=None, host="0.0.0.0"):
    global MAIN_LOOP
    MAIN_LOOP = asyncio.get_running_loop()

    PORT = 8765

    async with websockets.serve(handler, host, PORT):
        logging.info(f"[Bis Backend] WebSocket server online on ws://{host}:{PORT}")
        mic_thread = threading.Thread(target=run_microphone_listener, args=(model_path, device_index), daemon=True)
        mic_thread.start()
        await asyncio.Future() # keep running forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hey BIS microphone and WebSocket backend")
    parser.add_argument("--vosk-model", type=Path, help="Local Vosk model directory for offline recognition")
    parser.add_argument("--device", type=int, help="PyAudio input device index (defaults to the system microphone)")
    parser.add_argument("--host", default="0.0.0.0", help="WebSocket bind address (use localhost for local testing)")
    args = parser.parse_args()
    if args.vosk_model and not args.vosk_model.is_dir():
        parser.error(f"Model directory not found: {args.vosk_model}")
    try:
        asyncio.run(main(args.vosk_model, args.device, args.host))
    except KeyboardInterrupt:
        logging.info("\n[Bis Backend] Shutting down.")
