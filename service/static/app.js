/**
 * Hey BIS - Interfaz Web Interactiva para Clasificación de Audio y Wake Word
 */

// Elementos del DOM
const elStatusIndicator = document.getElementById("connection-indicator");
const elStatusText = document.getElementById("connection-text");
const elIpChips = document.getElementById("ip-chips");

const elBtnRecord = document.getElementById("btn-record");
const elRecordStatus = document.getElementById("record-status");
const elDurationSelect = document.getElementById("record-duration-select");
const elWaveformCanvas = document.getElementById("waveform-canvas");
const canvasCtx = elWaveformCanvas.getContext("2d");
const elPlaybackContainer = document.getElementById("playback-container");
const elAudioPreview = document.getElementById("audio-preview");

const elDropZone = document.getElementById("drop-zone");
const elFileInput = document.getElementById("file-input");
const elFileInfo = document.getElementById("file-info");
const elFileName = document.getElementById("file-name");
const elBtnClearFile = document.getElementById("btn-clear-file");
const elFilePlayback = document.getElementById("file-playback-container");
const elFileAudioPreview = document.getElementById("file-audio-preview");
const elBtnAnalyzeFile = document.getElementById("btn-analyze-file");

const elBtnTestSilence = document.getElementById("btn-test-silence");
const elBtnTestNoise = document.getElementById("btn-test-noise");
const elBtnTestTone = document.getElementById("btn-test-tone");

const elVerdictBanner = document.getElementById("verdict-banner");
const elVerdictIcon = document.getElementById("verdict-icon");
const elVerdictTitle = document.getElementById("verdict-title");
const elVerdictSubtitle = document.getElementById("verdict-subtitle");
const elConfidencePill = document.getElementById("confidence-pill");

const elValWake = document.getElementById("val-wake");
const elBarWake = document.getElementById("bar-wake");
const elValSpeech = document.getElementById("val-speech");
const elBarSpeech = document.getElementById("bar-speech");
const elValBg = document.getElementById("val-bg");
const elBarBg = document.getElementById("bar-bg");

const elMetricLatency = document.getElementById("metric-latency");
const elMetricDuration = document.getElementById("metric-duration");
const elInferenceTimeBadge = document.getElementById("inference-time-badge");

const elThresholdSlider = document.getElementById("threshold-slider");
const elThresholdVal = document.getElementById("threshold-val");
const elModelWeightsStatus = document.getElementById("model-weights-status");
const elModelArchitectureText = document.getElementById("model-architecture-text");

// Variables globales de estado
let isRecording = false;
let audioContext = null;
let mediaStream = null;
let scriptProcessor = null;
let analyserNode = null;
let recordedSamples = [];
let animFrameId = null;
let recordTimeoutId = null;
let selectedFileBlob = null;

// =============================================================================
// Inicialización y Comprobación del Servidor
// =============================================================================

async function initServerStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();

    // Actualizar indicador de conexión
    elStatusIndicator.className = "status-pill status-online";
    elStatusText.textContent = `En línea (${data.network_access.port})`;

    // Actualizar IPs para red local
    renderLanIps(data.network_access.lan_urls, data.network_access.localhost_url);

    // Actualizar datos de configuración
    if (data.wake_threshold) {
      elThresholdSlider.value = data.wake_threshold;
      elThresholdVal.textContent = Number(data.wake_threshold).toFixed(2);
    }

    if (data.weights_path) {
      elModelWeightsStatus.textContent = data.weights_loaded
        ? `Pesos: ${data.weights_path}`
        : "Pesos inicializados (He/Glorot)";
    }

    if (data.network_architecture) {
      elModelArchitectureText.textContent =
        `Entrada: ${data.network_architecture.input_features} Mel ` +
        `→ ${data.network_architecture.dense1} ` +
        `→ ${data.network_architecture.dense2} ` +
        `→ ${data.network_architecture.dense3}`;
    }
  } catch (err) {
    elStatusIndicator.className = "status-pill status-offline";
    elStatusText.textContent = "Desconectado";
    elIpChips.innerHTML = '<span class="ip-chip">No se pudo contactar con el backend</span>';
  }
}

function renderLanIps(lanUrls, localhostUrl) {
  elIpChips.innerHTML = "";
  const allUrls = [localhostUrl, ...lanUrls];
  const uniqueUrls = [...new Set(allUrls)];

  uniqueUrls.forEach((url) => {
    const chip = document.createElement("span");
    chip.className = "ip-chip";
    chip.innerHTML = `<span>${url}</span> <span class="copy-hint">[Copiar]</span>`;
    chip.title = "Haz clic para copiar URL al portapapeles";

    chip.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(url);
        chip.classList.add("copied");
        chip.querySelector(".copy-hint").textContent = "¡Copiado!";
        setTimeout(() => {
          chip.classList.remove("copied");
          chip.querySelector(".copy-hint").textContent = "[Copiar]";
        }, 1800);
      } catch (e) {
        prompt("Copia la URL manualmente:", url);
      }
    });

    elIpChips.appendChild(chip);
  });
}

// =============================================================================
// Navegación por Pestañas
// =============================================================================

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));

    btn.classList.add("active");
    const targetId = btn.getAttribute("data-tab");
    const targetContent = document.getElementById(targetId);
    if (targetContent) targetContent.classList.add("active");
  });
});

// =============================================================================
// Grabación de Micrófono & Web Audio API
// =============================================================================

elBtnRecord.addEventListener("click", toggleRecording);

async function toggleRecording() {
  if (isRecording) {
    stopRecording();
  } else {
    await startRecording();
  }
}

async function startRecording() {
  recordedSamples = [];

  try {
    // 1. Solicitar permiso de micrófono
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: true,
      },
    });

    const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
    audioContext = new AudioCtxClass();

    const source = audioContext.createMediaStreamSource(mediaStream);
    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 512;

    // ScriptProcessor para recolectar muestras crudas de audio
    const bufferSize = 4096;
    scriptProcessor = audioContext.createScriptProcessor(bufferSize, 1, 1);

    scriptProcessor.onaudioprocess = (e) => {
      if (!isRecording) return;
      const channelData = e.inputBuffer.getChannelData(0);
      recordedSamples.push(new Float32Array(channelData));
    };

    source.connect(analyserNode);
    analyserNode.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination);

    isRecording = true;
    elBtnRecord.classList.add("recording");
    elRecordStatus.textContent = "🎙️ Grabando... di 'Hey BIS' o habla ahora";

    // Iniciar dibujo de forma de onda
    drawLiveWaveform();

    // Comprobar duración automática
    const durationSetting = elDurationSelect.value;
    if (durationSetting !== "manual") {
      const durSec = parseFloat(durationSetting);
      let remaining = durSec;
      const intervalId = setInterval(() => {
        remaining -= 0.1;
        if (remaining > 0 && isRecording) {
          elRecordStatus.textContent = `🎙️ Grabando... (${remaining.toFixed(1)}s)`;
        } else {
          clearInterval(intervalId);
        }
      }, 100);

      recordTimeoutId = setTimeout(() => {
        clearInterval(intervalId);
        if (isRecording) stopRecording();
      }, durSec * 1000);
    }
  } catch (err) {
    console.error("Error al acceder al micrófono:", err);
    alert(
      "No se pudo acceder al micrófono: " +
        err.message +
        "\n\nRecuerda que los navegadores exigen localhost o HTTPS para el micrófono. Si estás en la red local HTTP, usa la pestaña 'Archivo WAV' o agrega la excepción en tu navegador."
    );
  }
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;

  if (recordTimeoutId) {
    clearTimeout(recordTimeoutId);
    recordTimeoutId = null;
  }

  elBtnRecord.classList.remove("recording");
  elRecordStatus.textContent = "Procesando audio y enviando a la IA...";

  // Desconectar nodos de audio
  if (scriptProcessor) scriptProcessor.disconnect();
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
  }

  // Cancelar animación
  if (animFrameId) cancelAnimationFrame(animFrameId);
  clearCanvas();

  // Concatenar muestras grabadas
  const totalLength = recordedSamples.reduce((acc, curr) => acc + curr.length, 0);
  const mergedSamples = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of recordedSamples) {
    mergedSamples.set(chunk, offset);
    offset += chunk.length;
  }

  if (totalLength === 0) {
    elRecordStatus.textContent = "Haz clic para grabar 1.5s (o decir 'Hey BIS')";
    return;
  }

  // Re-muestrear a 16 kHz si el contexto del navegador trabaja a 44.1k o 48k
  const nativeSampleRate = audioContext.sampleRate;
  const resampled16k = resampleLinear(mergedSamples, nativeSampleRate, 16000);

  // Codificar a WAV binario de 16 kHz mono 16-bit
  const wavBlob = encodeWAV(resampled16k, 16000);

  // Vista previa de audio para el usuario
  const audioUrl = URL.createObjectURL(wavBlob);
  elAudioPreview.src = audioUrl;
  elPlaybackContainer.classList.remove("hidden");

  // Enviar a la IA
  sendAudioToAI(wavBlob);
  elRecordStatus.textContent = "Haz clic para grabar de nuevo";
}

// Dibujo continuo de forma de onda mientras se graba
function drawLiveWaveform() {
  if (!isRecording) return;

  const width = elWaveformCanvas.width;
  const height = elWaveformCanvas.height;
  const bufferLength = analyserNode.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  analyserNode.getByteTimeDomainData(dataArray);

  canvasCtx.fillStyle = "#0f131a";
  canvasCtx.fillRect(0, 0, width, height);

  canvasCtx.lineWidth = 2.5;
  canvasCtx.strokeStyle = "#10b981";
  canvasCtx.beginPath();

  const sliceWidth = (width * 1.0) / bufferLength;
  let x = 0;

  for (let i = 0; i < bufferLength; i++) {
    const v = dataArray[i] / 128.0;
    const y = (v * height) / 2;

    if (i === 0) {
      canvasCtx.moveTo(x, y);
    } else {
      canvasCtx.lineTo(x, y);
    }
    x += sliceWidth;
  }

  canvasCtx.lineTo(width, height / 2);
  canvasCtx.stroke();

  animFrameId = requestAnimationFrame(drawLiveWaveform);
}

function clearCanvas() {
  canvasCtx.fillStyle = "#0f131a";
  canvasCtx.fillRect(0, 0, elWaveformCanvas.width, elWaveformCanvas.height);
  canvasCtx.strokeStyle = "#273142";
  canvasCtx.beginPath();
  canvasCtx.moveTo(0, elWaveformCanvas.height / 2);
  canvasCtx.lineTo(elWaveformCanvas.width, elWaveformCanvas.height / 2);
  canvasCtx.stroke();
}
clearCanvas();

// =============================================================================
// Re-muestreo & Codificador WAV a 16 kHz Mono
// =============================================================================

function resampleLinear(samples, fromRate, toRate = 16000) {
  if (fromRate === toRate) return samples;
  const ratio = toRate / fromRate;
  const outLength = Math.round(samples.length * ratio);
  const output = new Float32Array(outLength);

  for (let i = 0; i < outLength; i++) {
    const pos = i / ratio;
    const index = Math.floor(pos);
    const nextIndex = Math.min(index + 1, samples.length - 1);
    const frac = pos - index;
    output[i] = samples[index] * (1 - frac) + samples[nextIndex] * frac;
  }
  return output;
}

function encodeWAV(samples, sampleRate = 16000) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  /* RIFF header */
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");

  /* fmt subchunk */
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
  view.setUint16(20, 1, true); // AudioFormat (1 = PCM)
  view.setUint16(22, 1, true); // NumChannels (1 = Mono)
  view.setUint32(24, sampleRate, true); // SampleRate
  view.setUint32(28, sampleRate * 2, true); // ByteRate (SampleRate * NumChannels * BitsPerSample/8)
  view.setUint16(32, 2, true); // BlockAlign (NumChannels * BitsPerSample/8)
  view.setUint16(34, 16, true); // BitsPerSample (16 bits)

  /* data subchunk */
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  /* Escribir muestras int16 */
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([view], { type: "audio/wav" });
}

// =============================================================================
// Subida y Drag & Drop de Archivos
// =============================================================================

elDropZone.addEventListener("click", () => elFileInput.click());

elDropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  elDropZone.classList.add("dragover");
});

elDropZone.addEventListener("dragleave", () => {
  elDropZone.classList.remove("dragover");
});

elDropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  elDropZone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) {
    handleSelectedFile(e.dataTransfer.files[0]);
  }
});

elFileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    handleSelectedFile(e.target.files[0]);
  }
});

function handleSelectedFile(file) {
  selectedFileBlob = file;
  elFileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  elFileInfo.classList.remove("hidden");
  elBtnAnalyzeFile.disabled = false;

  const url = URL.createObjectURL(file);
  elFileAudioPreview.src = url;
  elFilePlayback.classList.remove("hidden");
}

elBtnClearFile.addEventListener("click", () => {
  selectedFileBlob = null;
  elFileInput.value = "";
  elFileInfo.classList.add("hidden");
  elFilePlayback.classList.add("hidden");
  elBtnAnalyzeFile.disabled = true;
});

elBtnAnalyzeFile.addEventListener("click", () => {
  if (selectedFileBlob) {
    sendAudioToAI(selectedFileBlob);
  }
});

// =============================================================================
// Señales Sintéticas de Prueba Rápida
// =============================================================================

function generateClientTestSignal(type) {
  const sr = 16000;
  const dur = 1.0;
  const n = sr * dur;
  const samples = new Float32Array(n);

  if (type === "tone") {
    // Tono puro 440 Hz
    for (let i = 0; i < n; i++) {
      samples[i] = 0.5 * Math.sin((2 * Math.PI * 440 * i) / sr);
    }
  } else if (type === "noise") {
    // Ruido blanco
    for (let i = 0; i < n; i++) {
      samples[i] = (Math.random() * 2 - 1) * 0.2;
    }
  } else {
    // Silencio (ceros)
    samples.fill(0);
  }

  return encodeWAV(samples, sr);
}

elBtnTestSilence.addEventListener("click", () => {
  const blob = generateClientTestSignal("silence");
  sendAudioToAI(blob);
});

elBtnTestNoise.addEventListener("click", () => {
  const blob = generateClientTestSignal("noise");
  sendAudioToAI(blob);
});

elBtnTestTone.addEventListener("click", () => {
  const blob = generateClientTestSignal("tone");
  sendAudioToAI(blob);
});

// =============================================================================
// Envío a la API & Procesamiento de Inferencia
// =============================================================================

async function sendAudioToAI(audioBlob) {
  setVerdictLoading();

  try {
    const res = await fetch("/api/classify", {
      method: "POST",
      headers: {
        "Content-Type": "audio/wav",
      },
      body: audioBlob,
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.error || `Error del servidor HTTP ${res.status}`);
    }

    const data = await res.json();
    displayResults(data);
  } catch (err) {
    console.error("Error al clasificar audio:", err);
    displayError(err.message);
  }
}

function setVerdictLoading() {
  elVerdictBanner.className = "verdict-banner verdict-waiting";
  elVerdictIcon.textContent = "⏳";
  elVerdictTitle.textContent = "Analizando con la red neuronal...";
  elVerdictSubtitle.textContent = "Extrayendo Log-Mel Spectrogram (32 filtros × 98 frames)...";
  elConfidencePill.textContent = "...";
}

function displayResults(data) {
  const probs = data.probabilities;
  const pWake = (probs.wake_word * 100).toFixed(1);
  const pSpeech = (probs.other_speech * 100).toFixed(1);
  const pBg = (probs.background * 100).toFixed(1);

  // Actualizar barras
  elValWake.textContent = `${pWake}%`;
  elBarWake.style.width = `${pWake}%`;

  elValSpeech.textContent = `${pSpeech}%`;
  elBarSpeech.style.width = `${pSpeech}%`;

  elValBg.textContent = `${pBg}%`;
  elBarBg.style.width = `${pBg}%`;

  // Actualizar métricas técnicas
  elMetricLatency.textContent = `${data.inference_time_ms} ms`;
  elMetricDuration.textContent = `${data.audio_info.duration_seconds} s`;
  elInferenceTimeBadge.textContent = `${data.inference_time_ms} ms`;

  const confPercent = (data.confidence * 100).toFixed(1);
  elConfidencePill.textContent = `${confPercent}%`;

  // Actualizar Veredicto Principal
  if (data.is_wake_word || data.label === "wake_word") {
    elVerdictBanner.className = "verdict-banner verdict-wake";
    elVerdictIcon.textContent = "✨";
    elVerdictTitle.textContent = "¡WAKE WORD DETECTADA! (Hey BIS)";
    elVerdictSubtitle.textContent =
      `Activación confirmada (Probabilidad: ${pWake}%, Umbral: ${Number(data.wake_threshold).toFixed(2)})`;
  } else if (data.label === "other_speech") {
    elVerdictBanner.className = "verdict-banner verdict-speech";
    elVerdictIcon.textContent = "🗣️";
    elVerdictTitle.textContent = "Voz Común Detectada";
    elVerdictSubtitle.textContent = "Se detectó habla humana, pero NO coincide con la frase clave 'Hey BIS'.";
  } else {
    elVerdictBanner.className = "verdict-banner verdict-bg";
    elVerdictIcon.textContent = "🔊";
    elVerdictTitle.textContent = "Ruido de Fondo / Silencio";
    elVerdictSubtitle.textContent = "El audio corresponde a sonido ambiental sin palabras activas.";
  }
}

function displayError(message) {
  elVerdictBanner.className = "verdict-banner verdict-waiting";
  elVerdictIcon.textContent = "⚠️";
  elVerdictTitle.textContent = "Error de Procesamiento";
  elVerdictSubtitle.textContent = message;
  elConfidencePill.textContent = "Error";
}

// =============================================================================
// Control de Umbral de Activación
// =============================================================================

elThresholdSlider.addEventListener("input", (e) => {
  elThresholdVal.textContent = Number(e.target.value).toFixed(2);
});

elThresholdSlider.addEventListener("change", async (e) => {
  const newThreshold = parseFloat(e.target.value);
  try {
    const res = await fetch("/api/threshold", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ threshold: newThreshold }),
    });
    const result = await res.json();
    if (result.success) {
      console.log("Umbral actualizado:", result.wake_threshold);
    }
  } catch (err) {
    console.error("Error al actualizar umbral:", err);
  }
});

// Arrancar al cargar la página
window.addEventListener("DOMContentLoaded", () => {
  initServerStatus();
});

