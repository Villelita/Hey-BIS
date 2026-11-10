# Hey BIS

Detector de la frase **Hey BIS** en español e inglés. Implementa una red pequeña desde cero en NumPy: Log-Mel de un segundo (3,136 características), capas 64 y 32 con ReLU y tres clases: `background`, `other_speech` y `wake_word`.

La computadora prepara los datos y entrena. **La Raspberry Pi ejecuta la detección desde un micrófono USB**, con los pesos entrenados. También se puede probar la inferencia en la computadora.

## Estado real

Los 2,000 audios ambientales ESC-50 locales ya tienen copias PCM mono de 16 kHz y 16 bits registradas en `data/manifest.csv`. Los originales a 44.1 kHz se conservan. El fondo registrado ocupa 320,088,000 bytes y dura 2 h 46 min 40 s.

**Faltan grabaciones propias de Hey BIS y otras palabras en ambos idiomas.** El programa bloquea el entrenamiento hasta que haya ejemplos de las tres clases y de ambos idiomas en train, validation y test. No incluye un modelo de voz preentrenado ni afirma exactitud sin medirla. Las pruebas sintéticas verifican el software; sus resultados no representan reconocimiento de voz real.

Los 10 GB de `config.json` son una meta local opcional, no un requisito del PDF. Los 4 GB de RAM son una referencia configurable, no una comprobación del equipo conectado.

## Instalación

Python 3.10 o posterior. Desde la raíz del proyecto, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-audio.txt
.\.venv\Scripts\Activate.ps1
```

En esta carpeta ya se preparó `.venv` con la dependencia de micrófono. Si PowerShell no permite activar el entorno, sustituye `python` en los comandos siguientes por `.\.venv\Scripts\python.exe`.

Para preparar datos, entrenar y clasificar archivos sin micrófono basta con `pip install -r requirements.txt`. La captura utiliza [python-sounddevice](https://python-sounddevice.readthedocs.io/); en Linux requiere PortAudio.

## 1. Preparar y comprobar audios

```powershell
python main.py init --data-dir data
python prepare_audio.py --data-dir data
python main.py inspect --data-dir data
python main.py inspect --data-dir data --require-training-ready
```

`prepare_audio.py` genera copias en una subcarpeta `pcm16k` de cada ubicación original y actualiza el manifiesto al terminar todas las conversiones. Guarda el manifiesto previo y la trazabilidad en `data/sources/preparation/`. No modifica ni sobrescribe audios originales. Repetir el comando no vuelve a convertir archivos ya preparados. Acepta PCM mono de 16 bits; otros formatos deben convertirse primero. Para limitar memoria, rechaza archivos por convertir de más de cinco minutos.

Para importar una descarga nueva de ESC-50 en un dataset nuevo:

```powershell
python import_esc50.py --source-dir C:\datasets\ESC-50-master --data-dir data-nuevo
```

La importación conserva atribuciones y agrupa por fuente. No la repitas sobre los originales sin convertir de este dataset: el importador rechaza sobrescribir un archivo diferente.

El manifiesto requiere estas columnas:

```csv
path,label,language,speaker_id,split,consent
```

Audio: WAV PCM, mono, 16,000 Hz, 16 bits. Voz: idioma `es` o `en`; fondo: `none`. Grabaciones propias: `consent=yes`; ESC-50: `licensed`, con su licencia. Cada hablante pertenece a una sola partición. No copies una grabación entre particiones.

## 2. Grabar las clases faltantes

Primero identifica tu micrófono:

```powershell
python microphone.py devices
```

Graba únicamente con permiso claro de todas las personas que puedan escucharse. `--consent` confirma ese permiso, sin sustituir el registro de consentimiento que lleve el equipo.

Ejemplos para un hablante de entrenamiento:

```powershell
python microphone.py record --label wake_word --language es --speaker p01 --split train --seconds 1 --consent
python microphone.py record --label wake_word --language en --speaker p01 --split train --seconds 1 --consent
python microphone.py record --label other_speech --language es --speaker p01 --split train --seconds 5 --consent
python microphone.py record --label other_speech --language en --speaker p01 --split train --seconds 5 --consent
```

Puedes seleccionar el micrófono con `--device 1` o su nombre. Cada comando crea un WAV único y lo registra automáticamente. Pronuncia la frase completa al iniciar la captura de un segundo; revisa que no quede cortada. Las frases positivas de más de un segundo se rechazan en entrenamiento para evitar recortarlas sin avisar. Si necesitas frases más largas, tendrás que cambiar de forma consistente la ventana, las características y la arquitectura.

Repite con distintas voces, distancias y ruidos. Reserva otros hablantes, por ejemplo `p02` para `validation` y `p03` para `test`; el programa impide registrar al mismo identificador en dos particiones. Cada partición debe contener ambas clases de voz en español e inglés y fondo. El mínimo validado solo comprueba cobertura, no que el dataset sea suficiente para generalizar.

Graba aparte conversación normal sin Hey BIS, con hablantes reservados para test, para medir falsas activaciones por hora:

```powershell
python microphone.py record --label other_speech --language es --speaker p03 --split test --seconds 60 --consent
```

Conserva el nombre del WAV que imprime el comando. Usa suficientes minutos de conversación para que la tasa sea informativa. Durante recolección, evita ejecutar dos grabadores a la vez sobre el mismo manifiesto.

## 3. Entrenar

```powershell
python main.py inspect --data-dir data --require-training-ready
python train_wake_word.py --data-dir data --epochs 15 --batch-size 32 --lr 0.001 --seed 42 --save-path output/wake_word_model.npz
```

- No se omiten archivos inválidos en silencio: un dataset incompleto o incorrecto produce un error y código de salida distinto de cero.
- Los negativos largos se segmentan en ventanas de un segundo, incluida la cola. Las frases positivas cortas se rellenan con silencio.
- Solo train se equilibra por sobremuestreo de clases. Validation y test mantienen sus distribuciones.
- Se usa la época con menor pérdida de validación. El umbral se elige en validation, maximizando sensibilidad con una tasa de falsos positivos por ventana de hasta `--max-false-positive-rate` (0.01 por defecto). Esa tasa no equivale a falsas activaciones por hora.
- No hay un límite oculto de 500 ejemplos. `--max-per-class N` limita opcionalmente solo train para experimentos rápidos, tomando los primeros N por clase según el manifiesto.
- El entrenamiento guarda pesos, arquitectura, etiquetas, versión de características, umbral y huellas del manifiesto y del audio. El historial, pérdidas y configuración quedan en `output/wake_word_model.training.json`.
- Test no se usa para elegir época o umbral. Mantén el manifiesto y sus audios fijos durante el experimento; no edites ni reemplaces WAV después de entrenar.

El cargador mantiene las características en RAM; no carga el audio completo del dataset. Su consumo crece con el número de ventanas, por lo que el entrenamiento se realiza en la computadora. Para datasets mucho mayores, usa el límite explícito de train durante experimentos o adapta el cargador a lotes desde disco.

Para generar la curva de pérdida para el informe:

```powershell
python -m pip install -r requirements-plots.txt
python plot_training.py output/wake_word_model.training.json --output output/loss.svg
```

## 4. Clasificar y evaluar

Clip de hasta un segundo:

```powershell
python audio_classifier.py --weights output/wake_word_model.npz --wav data/audio/recorded/CLIP.wav
```

Archivo largo completo:

```powershell
python audio_classifier.py --weights output/wake_word_model.npz --wav CONVERSACION.wav --scan
```

Evaluación final en test:

```powershell
python evaluate_wake_word.py --data-dir data --weights output/wake_word_model.npz --output output/evaluation.json
```

Para incluir la métrica del PDF, indica uno o varios WAV de conversación sin la frase, que nunca se hayan usado para entrenar o ajustar el modelo:

```powershell
python evaluate_wake_word.py --data-dir data --weights output/wake_word_model.npz --ordinary-talk data/audio/recorded/CONVERSACION_TEST.wav --confirm-no-wake-word --output output/evaluation.json
```

Se guardan matriz de confusión, precisión y sensibilidad de activación, sensibilidad por idioma y eventos falsos por hora. Los resultados de fondo y de otras palabras se separan de la conversación confirmada. Si no proporcionas conversación, la métrica correspondiente queda pendiente, no se sustituye por ruido ambiental.

La política predeterminada usa ventanas de un segundo cada 0.25 s. Una racha positiva cuenta como un solo evento; para un nuevo evento debe haber una ventana negativa y haber pasado el tiempo de espera, un segundo. `--hop-size` y `--cooldown` se comparten entre análisis de archivos, evaluación de eventos y escucha en vivo. La duración de la métrica es la del audio real, sin sumar ventanas solapadas. No mezcles resultados de políticas distintas.

## 5. Demo y mediciones en Raspberry Pi

Copia el código y el modelo entrenado a la Pi. El dataset completo no es necesario para escuchar en vivo. En Raspberry Pi OS:

```bash
sudo apt install python3-venv libportaudio2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-audio.txt
python microphone.py devices
python microphone.py listen --weights output/wake_word_model.npz --log output/pi-live.jsonl
```

Para terminar pulsa Ctrl+C. También puedes limitar la duración con `--seconds 60`. Si el micrófono no acepta 16 kHz mono, selecciona un dispositivo compatible; se informa el error, no se cambia la frecuencia sin remuestrear. La escucha no guarda las voces, solo predicciones, eventos, configuración y tiempos de procesamiento. Un desbordamiento de captura detiene la sesión porque sus tiempos y resultados estarían incompletos.

Los registros contienen el promedio de procesamiento y el percentil 95 de las últimas 10,000 ventanas. Incluyen extracción de características y red; excluyen la espera del micrófono y la acumulación inicial de un segundo. Un tiempo bajo de procesamiento no equivale a una respuesta instantánea.

Para medir recursos ejecuta en la Pi, con el dataset disponible:

```bash
python profile_model.py --weights output/wake_word_model.npz --data-dir data --output output/pi-profile.json
```

Sin `--weights`, el perfil se identifica como arquitectura sin entrenar. Distingue peso de parámetros, arreglos asignados y memoria del proceso, y cronometra tanto la red como características más red con audio sintético. El hardware de cada resultado se registra: medir en PC no demuestra rendimiento en Pi. El perfil de almacenamiento cuenta solo WAV válidos registrados y muestra aparte el espacio físico, incluidos los originales.

## Pruebas

```powershell
python -m unittest discover -s tests -v
```

Incluyen regresiones de umbral, ausencia de pesos, conversión sin pérdida de originales, cola temporal, conteo de eventos, validación, captura simulada y un ciclo sintético de entrenamiento y evaluación. Las pruebas nunca necesitan encender el micrófono.

Si el entorno de ejecución restringe su carpeta temporal, usa una carpeta del proyecto:

```powershell
New-Item -ItemType Directory -Force tmp | Out-Null
$env:TEMP = (Resolve-Path tmp).Path
$env:TMP = $env:TEMP
python -m unittest discover -s tests -v
```

## Entrega del curso

Faltan las grabaciones propias, las mediciones reales del modelo final y la verificación física en la Pi. Después prepara la demo, el dataset con permisos y atribuciones, el código y el informe técnico de dos páginas con integrantes, método, resultados, tamaños de muestra y limitaciones. Cada integrante debe llevar su bitácora semanal. Una segunda palabra de comando es opcional.

`data/`, modelos, resultados, `.env` y el PDF se excluyen de Git. Entrega el dataset por un medio acordado con el profesor; no basta con compartir el repositorio. Nunca publiques grabaciones sin el permiso correspondiente.
