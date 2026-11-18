#!/usr/bin/env python3
"""Servidor HTTP para Hey BIS Wake Word & Clasificador de Ruido.

Permite interactuar con la red neuronal y clasificador de audio desde cualquier
navegador web a través de la red local (LAN) o localhost.
"""

from __future__ import annotations

import argparse
import base64
import email
import io
import json
import mimetypes
from pathlib import Path
import socket
import sys
import time
import threading
import wave
from uuid import uuid4
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse, parse_qs

# Asegurar que el directorio raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVICE_DIR = Path(__file__).resolve().parent
STATIC_DIR = SERVICE_DIR / "static"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as scipy_signal

from audio_classifier import AudioNoiseClassifier as BaseAudioNoiseClassifier
import audio_features
from microphone import validate_recording
from main import FIELDS, LABELS, SPLITS, initialize
from import_esc50 import write_csv

RECORDING_LOCK = threading.Lock()
DATA_DIR = PROJECT_ROOT / "data"


class AudioNoiseClassifier(BaseAudioNoiseClassifier):
    """Clasificador adaptado al servicio web (permite modo demo sin .npz previo)."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        wake_threshold: float | None = None,
        *,
        allow_untrained: bool | None = None,
    ):
        if allow_untrained is None:
            allow_untrained = weights_path is None
        super().__init__(
            weights_path=weights_path,
            wake_threshold=wake_threshold,
            allow_untrained=allow_untrained,
        )

    def classify_clip(self, samples: np.ndarray) -> dict:
        """Clasifica un clip de cualquier duración (usa ventana deslizante si supera WINDOW_SAMPLES)."""
        samples = audio_features.validate_samples(samples)
        if len(samples) <= audio_features.WINDOW_SAMPLES:
            return self.classify_samples(samples)
        windows = self.scan_streaming_window(
            samples,
            window_size=audio_features.WINDOW_SAMPLES,
            hop_size=min(4_000, audio_features.WINDOW_SAMPLES),
        )
        best = max(windows, key=lambda w: w["wake_word_probability"])
        return {
            "label": best["label"],
            "is_wake_word": any(w["is_wake_word"] for w in windows),
            "confidence": best["confidence"],
            "wake_word_probability": best["wake_word_probability"],
            "probabilities": best["probabilities"],
            "windows_evaluated": len(windows),
        }


def get_lan_ips() -> List[str]:
    """Obtiene las direcciones IP disponibles para conectarse en la red local."""
    ips = []
    # 1. Método rápido por socket UDP de salida (no envía paquetes reales)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.5)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                ips.append(ip)
    except Exception:
        pass

    # 2. Respaldo en caso de interfaz sin salida exterior directa
    if not ips:
        try:
            host_name = socket.gethostname()
            resolved = socket.gethostbyname(host_name)
            if resolved and not resolved.startswith("127."):
                ips.append(resolved)
        except Exception:
            pass

    if not ips:
        ips.append("127.0.0.1")

    return list(dict.fromkeys(ips))


def decode_audio_bytes_to_16k_mono(raw_bytes: bytes) -> np.ndarray:
    """Decodifica un buffer de audio en un arreglo float32 de 16 kHz mono [-1.0, 1.0]."""
    if len(raw_bytes) == 0:
        return np.zeros(16000, dtype=np.float32)

    # Intento 1: WAV estándar o no estándar mediante scipy.io.wavfile
    try:
        sr, data = wavfile.read(io.BytesIO(raw_bytes))
        if data.dtype == np.int16:
            samples = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            samples = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            samples = (data.astype(np.float32) - 128.0) / 128.0
        elif np.issubdtype(data.dtype, np.floating):
            samples = data.astype(np.float32)
        else:
            samples = data.astype(np.float32)

        # Si es multicanal (estéreo), promediar a mono
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        # Re-muestrear a 16000 Hz si la tasa es diferente
        if sr != 16000 and len(samples) > 0:
            target_samples = int(round(len(samples) * 16000 / sr))
            samples = scipy_signal.resample(samples, target_samples).astype(np.float32)

        return samples
    except Exception:
        pass

    # Intento 2: PCM crudo 16-bit signed LE mono a 16 kHz
    try:
        samples = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32) / 32768.0
        if len(samples) > 0:
            return samples
    except Exception:
        pass

    # Intento 3: PCM float32 LE
    try:
        samples = np.frombuffer(raw_bytes, dtype="<f4").astype(np.float32)
        if len(samples) > 0:
            return samples
    except Exception:
        pass

    raise ValueError("No se pudo decodificar el formato de audio recibido (se espera WAV o PCM).")


def generate_synthetic_wav(sample_type: str = "silence", duration_sec: float = 1.0) -> bytes:
    """Genera un archivo WAV de prueba a 16 kHz mono."""
    sr = 16000
    n_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    if sample_type == "tone":
        # Tono armónico 440 Hz (La)
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    elif sample_type == "noise":
        # Ruido blanco moderado
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 0.15, n_samples)
    else:  # silence
        signal = np.zeros(n_samples, dtype=np.float32)

    signal = np.clip(signal, -1.0, 1.0)
    int_signal = (signal * 32767).astype(np.int16)

    buf = io.BytesIO()
    wavfile.write(buf, sr, int_signal)
    return buf.getvalue()


class HeyBISHandler(SimpleHTTPRequestHandler):
    """Manejador HTTP para servir la interfaz web y la API REST de inferencia."""

    classifier: AudioNoiseClassifier
    weights_path_str: str | None
    server_port: int
    lan_ips: List[str]

    def __init__(self, *args, **kwargs):
        # Configurar directorio estático por defecto
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        """Personalizar logging en consola para mayor legibilidad."""
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {args[0]} - {args[1]} {args[2]}\n")

    def send_cors_headers(self) -> None:
        """Añade cabeceras CORS para permitir consumo desde cualquier dispositivo en la red."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept, Authorization")

    def do_OPTIONS(self) -> None:
        """Maneja preflight requests de navegadores."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_cors_headers()
        self.end_headers()

    def send_json_response(self, data: Dict[str, Any], status: int = 200) -> None:
        """Envía una respuesta JSON formateada."""
        payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        """Enruta peticiones GET hacia la API o hacia los archivos estáticos del frontend."""
        parsed = urlparse(self.path)
        path = parsed.path

        # Endpoints API
        if path == "/api/status" or path == "/api/info":
            summary = self.classifier.model.summary()
            in_dim = summary.get("input_dimension", 3136)
            l0_neurons = summary["layers"][0]["neurons"]
            l1_neurons = summary["layers"][1]["neurons"]
            l2_neurons = summary["layers"][2]["neurons"]
            self.send_json_response({
                "status": "online",
                "service": "Hey-BIS Neural Audio Service",
                "target_wake_word": "Hey BIS",
                "sample_rate_hz": 16000,
                "wake_threshold": self.classifier.wake_threshold,
                "weights_path": self.weights_path_str or "Inicializado (sin archivo externo)",
                "weights_loaded": self.weights_path_str is not None,
                "dataset_directory": str(DATA_DIR),
                "network_architecture": {
                    "input_features": in_dim,
                    "dense1": f"{in_dim} -> {l0_neurons} (ReLU)",
                    "dense2": f"{l0_neurons} -> {l1_neurons} (ReLU)",
                    "dense3": f"{l1_neurons} -> {l2_neurons} (Softmax)",
                    "total_parameters": summary.get("total_parameters", 202947),
                },
                "network_access": {
                    "port": self.server_port,
                    "localhost_url": f"http://localhost:{self.server_port}",
                    "lan_urls": [f"http://{ip}:{self.server_port}" for ip in self.lan_ips],
                },
            })
            return

        if path == "/api/test-audio":
            query = parse_qs(parsed.query)
            sample_type = query.get("type", ["silence"])[0]
            try:
                wav_bytes = generate_synthetic_wav(sample_type=sample_type, duration_sec=1.0)
                self.send_response(HTTPStatus.OK)
                self.send_cors_headers()
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(wav_bytes)))
                self.send_header("Content-Disposition", f'inline; filename="{sample_type}.wav"')
                self.end_headers()
                self.wfile.write(wav_bytes)
            except Exception as exc:
                self.send_json_response({"error": str(exc)}, status=500)
            return

        # Servir index.html en raíz
        if path == "/" or path == "/index.html":
            self.path = "/index.html"
            return super().do_GET()

        # Si solicita archivo estático
        file_path = STATIC_DIR / path.lstrip("/")
        if file_path.is_file():
            self.path = "/" + str(file_path.relative_to(STATIC_DIR))
            return super().do_GET()

        # Fallback a index.html
        self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:
        """Enruta peticiones POST (clasificación de audio y actualización de configuración)."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/recordings":
            self.handle_save_recording()
            return

        if path == "/api/threshold":
            self.handle_set_threshold()
            return

        if path == "/api/classify" or path == "/api/predict":
            self.handle_classify()
            return

        if path == "/api/record-server-mic":
            self.handle_record_server_mic()
            return

        self.send_json_response({"error": f"Endpoint desconocido: {path}"}, status=404)

    def handle_save_recording(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 200_000:
                raise ValueError("La grabación excede el tamaño permitido.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Envía los datos de la grabación como un objeto JSON.")
            label, language = body.get("label"), body.get("language")
            speaker, split = body.get("speaker", ""), body.get("split")
            if label not in LABELS or split not in SPLITS or body.get("consent") is not True:
                raise ValueError("Revisa la categoría, partición y permiso de grabación.")
            raw = base64.b64decode(body.get("audio", ""), validate=True)
            with wave.open(io.BytesIO(raw), "rb") as audio:
                if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype(), audio.getnframes()) != (1, 2, 16000, "NONE", 48000):
                    raise ValueError("Se requiere un WAV de 3 segundos, mono, 16 kHz y 16 bits.")
                if len(audio.readframes(48000)) != 96000:
                    raise ValueError("El audio está incompleto.")
            with RECORDING_LOCK:
                rows = validate_recording(DATA_DIR, label, language, speaker, split, 3)
                initialize(DATA_DIR)
                relative = Path("audio/recorded") / f"{speaker}_{language}_{label}_{uuid4().hex}.wav"
                target = DATA_DIR / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with target.open("xb") as output:
                        output.write(raw)
                    rows.append(dict(zip(FIELDS, (relative.as_posix(), label, language, speaker, split, "yes"))))
                    write_csv(DATA_DIR / "manifest.csv", FIELDS, rows)
                except Exception:
                    target.unlink(missing_ok=True)
                    raise
            self.send_json_response({"success": True, "path": str(target), "count": len(rows)})
        except (ValueError, TypeError, wave.Error, EOFError) as error:
            self.send_json_response({"error": str(error)}, status=400)
        except OSError:
            self.send_json_response({"error": "No se pudo guardar. Intenta de nuevo."}, status=500)

    def handle_record_server_mic(self) -> None:
        """Graba desde el micrófono físico del servidor cuando el cliente entra por HTTP sin WebRTC."""
        start_time = time.perf_counter()
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            duration_sec = 1.5
            if content_length > 0:
                body = json.loads(self.rfile.read(content_length).decode("utf-8"))
                duration_sec = float(body.get("duration", 1.5))
            duration_sec = max(0.5, min(duration_sec, 5.0))

            import sounddevice as sd
            n_frames = int(round(duration_sec * 16000))
            rec = sd.rec(n_frames, samplerate=16000, channels=1, dtype="float32", blocking=True)
            samples = rec[:, 0].astype(np.float32)

            infer_start = time.perf_counter()
            if hasattr(self.classifier, "classify_clip"):
                inference_result = self.classifier.classify_clip(samples)
            else:
                inference_result = self.classifier.classify_samples(samples)
            inference_ms = round((time.perf_counter() - infer_start) * 1000.0, 2)

            # Generar WAV en base64 para reproductor del navegador
            int_signal = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
            buf = io.BytesIO()
            wavfile.write(buf, 16000, int_signal)
            wav_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            self.send_json_response({
                "success": True,
                "source": "server_microphone",
                "label": inference_result["label"],
                "is_wake_word": inference_result["is_wake_word"],
                "confidence": round(float(inference_result["confidence"]), 4),
                "wake_word_probability": round(float(inference_result["wake_word_probability"]), 4),
                "probabilities": {
                    k: round(float(v), 4) for k, v in inference_result["probabilities"].items()
                },
                "audio_info": {
                    "samples_received": int(len(samples)),
                    "duration_seconds": round(len(samples) / 16000.0, 3),
                    "target_rate_hz": 16000,
                },
                "audio_wav_base64": f"data:audio/wav;base64,{wav_b64}",
                "inference_time_ms": inference_ms,
                "total_elapsed_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
                "wake_threshold": self.classifier.wake_threshold,
            })
        except Exception as exc:
            self.send_json_response({
                "success": False,
                "error": f"No se pudo grabar desde el micrófono del servidor: {exc}",
            }, status=400)

    def handle_set_threshold(self) -> None:
        """Actualiza el umbral dinámico de activación para 'Hey BIS'."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
            new_val = float(data.get("threshold", data.get("wake_threshold", 0.60)))
            if not (0.01 <= new_val <= 0.99):
                raise ValueError("El umbral debe ser un valor entre 0.01 y 0.99")
            self.classifier.wake_threshold = new_val
            self.send_json_response({
                "success": True,
                "wake_threshold": self.classifier.wake_threshold,
                "message": f"Umbral de Wake Word actualizado a {self.classifier.wake_threshold:.2f}",
            })
        except Exception as exc:
            self.send_json_response({"error": str(exc)}, status=400)

    def handle_classify(self) -> None:
        """Recibe y clasifica una señal de audio enviada desde el navegador."""
        start_time = time.perf_counter()
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", 0))

        if content_length <= 0:
            self.send_json_response({"error": "Petición vacía (Content-Length = 0)"}, status=400)
            return

        raw_body = self.rfile.read(content_length)

        try:
            audio_bytes: bytes = b""

            if "application/json" in content_type:
                # Caso 1: JSON con audio base64 o arreglo de muestras
                json_data = json.loads(raw_body.decode("utf-8"))
                if "audio_base64" in json_data:
                    b64_str = json_data["audio_base64"]
                    if "," in b64_str:
                        b64_str = b64_str.split(",", 1)[1]
                    audio_bytes = base64.b64decode(b64_str)
                    samples = decode_audio_bytes_to_16k_mono(audio_bytes)
                elif "samples" in json_data:
                    samples = np.array(json_data["samples"], dtype=np.float32)
                else:
                    raise ValueError("El JSON debe contener 'audio_base64' o 'samples'")
            elif "multipart/form-data" in content_type:
                # Caso 2: Formulario Multipart con archivo de audio
                # Extraer cuerpo del archivo usando cabecera de mensaje MIME
                header_text = f"Content-Type: {content_type}\r\n\r\n"
                msg = email.message_from_bytes(header_text.encode("latin1") + raw_body)
                found_audio = False
                for part in msg.walk():
                    if part.get_content_maintype() == "multipart":
                        continue
                    payload = part.get_payload(decode=True)
                    if payload and len(payload) > 44:  # tamaño mínimo WAV header
                        samples = decode_audio_bytes_to_16k_mono(payload)
                        found_audio = True
                        break
                if not found_audio:
                    raise ValueError("No se encontró ningún archivo de audio válido en el formulario multipart.")
            else:
                # Caso 3: Audio WAV binario directo o PCM
                samples = decode_audio_bytes_to_16k_mono(raw_body)

            # Clasificación mediante la red neuronal (soporta tanto ventanas de 1s como clips largos)
            if hasattr(self.classifier, "classify_clip"):
                inference_result = self.classifier.classify_clip(samples)
            else:
                inference_result = self.classifier.classify_samples(samples)
            inference_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

            duration_sec = round(len(samples) / 16000.0, 3)

            response = {
                "success": True,
                "label": inference_result["label"],
                "is_wake_word": inference_result["is_wake_word"],
                "confidence": round(float(inference_result["confidence"]), 4),
                "wake_word_probability": round(float(inference_result["wake_word_probability"]), 4),
                "probabilities": {
                    k: round(float(v), 4) for k, v in inference_result["probabilities"].items()
                },
                "audio_info": {
                    "samples_received": int(len(samples)),
                    "duration_seconds": duration_sec,
                    "target_rate_hz": 16000,
                },
                "inference_time_ms": inference_ms,
                "wake_threshold": self.classifier.wake_threshold,
            }
            self.send_json_response(response, status=200)

        except Exception as exc:
            self.send_json_response({
                "success": False,
                "error": f"Error al procesar audio: {str(exc)}",
            }, status=400)


def resolve_dataset_directory(data_dir: str | Path | None = None) -> Path:
    """Usa la ruta explícita, configuración local o repositorio Dataset cercano."""
    local_config = SERVICE_DIR / "dataset.local.json"
    if data_dir is None and local_config.is_file():
        data_dir = json.loads(local_config.read_text(encoding="utf-8"))["data_dir"]
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()
    for parent in PROJECT_ROOT.parents:
        for name in ("Dataset", "DataSet"):
            candidate = parent / name / "Hey-BIS"
            if (candidate / "manifest.csv").is_file():
                return candidate.resolve()
    return PROJECT_ROOT / "data"


def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    weights_path: str | Path | None = None,
    wake_threshold: float = 0.60,
    data_dir: str | Path | None = None,
) -> None:
    """Inicia el servidor HTTP multi-hilo en la dirección y puerto especificados."""
    global DATA_DIR
    DATA_DIR = resolve_dataset_directory(data_dir)
    # Verificar si existe archivo de pesos por defecto en el proyecto si no fue especificado
    actual_weights: Path | None = None
    if weights_path:
        w_path = Path(weights_path)
        if w_path.is_file():
            actual_weights = w_path
    else:
        default_model = PROJECT_ROOT / "wake_word_model.npz"
        if default_model.is_file():
            actual_weights = default_model

    classifier = AudioNoiseClassifier(
        weights_path=actual_weights,
        wake_threshold=wake_threshold,
    )

    lan_ips = get_lan_ips()

    # Inyectar propiedades de clase al Handler
    HeyBISHandler.classifier = classifier
    HeyBISHandler.weights_path_str = str(actual_weights) if actual_weights else None
    HeyBISHandler.server_port = port
    HeyBISHandler.lan_ips = lan_ips

    # Multi-hilo para atender varios navegadores simultáneos
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer.allow_reuse_address = True

    try:
        httpd = ThreadingHTTPServer((host, port), HeyBISHandler)
    except OSError as err:
        if "Address already in use" in str(err):
            # Intentar puerto alternativo 8000 si 8080 está ocupado
            alt_port = 8000 if port != 8000 else 8081
            print(f"[!] Puerto {port} ocupado. Reintentando en puerto {alt_port}...")
            port = alt_port
            HeyBISHandler.server_port = port
            httpd = ThreadingHTTPServer((host, port), HeyBISHandler)
        else:
            raise

    weights_display = str(actual_weights.name) if actual_weights else "Pesos inicializados (He/Glorot)"

    print("=" * 64)
    print("    HEY BIS - SERVICIO WEB DE DETECCIÓN DE WAKE WORD & AUDIO    ")
    print("=" * 64)
    print(
        f"  * Modelo:          {classifier.model.input_dim} -> "
        f"{classifier.model.hidden_dim1} (ReLU) -> "
        f"{classifier.model.hidden_dim2} (ReLU) -> "
        f"{classifier.model.num_classes} (Softmax)"
    )
    print(f"  * Estado Pesos:    {weights_display}")
    print(f"  * Umbral Activo:   {wake_threshold:.2f}")
    print(f"  * Dataset:         {DATA_DIR}")
    print(f"  * Acceso Local:    http://localhost:{port}")
    for ip in lan_ips:
        print(f"  * Red Local (LAN): http://{ip}:{port}")
    print("=" * 64)
    print("  Conéctate desde cualquier navegador en tu red local o móvil.")
    print("  Presiona Ctrl + C para detener el servidor.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[+] Deteniendo servidor Hey-BIS...")
    finally:
        httpd.server_close()
        print("[+] Servidor detenido correctamente.")


def main():
    parser = argparse.ArgumentParser(
        description="Servidor Web de Hey BIS para interacción con la IA desde navegador."
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Dirección host (0.0.0.0 para red local)")
    parser.add_argument("--port", type=int, default=8080, help="Puerto HTTP (por defecto 8080)")
    parser.add_argument("--weights", type=Path, default=None, help="Ruta al archivo .npz de pesos")
    parser.add_argument("--threshold", type=float, default=0.60, help="Umbral de detección de Wake Word")
    parser.add_argument("--data-dir", type=Path, help="Carpeta del dataset con manifest.csv y audio/")
    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        weights_path=args.weights,
        wake_threshold=args.threshold,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
