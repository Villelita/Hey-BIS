(() => {
  const form = document.getElementById('dataset-form');
  const record = document.getElementById('dataset-record');
  const save = document.getElementById('dataset-save');
  const preview = document.getElementById('dataset-preview');
  const status = document.getElementById('dataset-status');
  let blob = null;
  let previewUrl = null;
  let metadata = null;
  let saving = false;

  record.addEventListener('click', async () => {
    if (!form.reportValidity()) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      status.textContent = 'Abre el dashboard en localhost o HTTPS para grabar desde este dispositivo.';
      return;
    }
    record.disabled = true;
    save.disabled = true;
    blob = null;
    preview.hidden = true;
    const fields = new FormData(form);
    metadata = Object.fromEntries(fields.entries());
    metadata.consent = fields.has('consent');
    let stream;
    let context;
    let processor;
    let source;
    try {
      status.textContent = 'Solicitando micrófono…';
      stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
      context = new AudioContext();
      await context.resume();
      source = context.createMediaStreamSource(stream);
      processor = context.createScriptProcessor(4096, 1, 1);
      const requiredSamples = Math.round(context.sampleRate * 3);
      const samples = new Float32Array(requiredSamples);
      let offset = 0;
      status.textContent = 'Grabando 3 segundos. Habla ahora…';
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('La captura se interrumpió. Vuelve a grabar.')), 8000);
        processor.onaudioprocess = event => {
          const chunk = event.inputBuffer.getChannelData(0);
          const count = Math.min(chunk.length, requiredSamples - offset);
          samples.set(chunk.subarray(0, count), offset);
          offset += count;
          if (offset === requiredSamples) {
            processor.onaudioprocess = null;
            clearTimeout(timeout);
            resolve();
          }
        };
        source.connect(processor);
        processor.connect(context.destination);
      });
      blob = encodeWAV(resampleLinear(samples, context.sampleRate, 16000), 16000);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      previewUrl = URL.createObjectURL(blob);
      preview.src = previewUrl;
      preview.hidden = false;
      save.disabled = false;
      status.textContent = 'Escucha el audio. Puedes guardarlo o volver a grabar para reemplazarlo.';
    } catch (error) {
      status.textContent = `No se pudo grabar: ${error.message}`;
    } finally {
      if (processor) processor.disconnect();
      if (source) source.disconnect();
      if (stream) stream.getTracks().forEach(track => track.stop());
      if (context) await context.close();
      record.disabled = false;
    }
  });

  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (!blob || saving) return;
    saving = true;
    save.disabled = true;
    record.disabled = true;
    status.textContent = 'Guardando audio y etiqueta…';
    try {
      const bytes = new Uint8Array(await blob.arrayBuffer());
      let binary = '';
      for (const byte of bytes) binary += String.fromCharCode(byte);
      const response = await fetch('/api/recordings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...metadata, audio: btoa(binary) })
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'No se pudo guardar.');
      blob = null;
      status.textContent = `Guardado en ${result.path}. Disponible para ${metadata.split === 'train' ? 'el próximo entrenamiento' : 'la evaluación'}.`;
    } catch (error) {
      status.textContent = error.message;
      save.disabled = false;
    } finally {
      saving = false;
      record.disabled = false;
    }
  });
})();
