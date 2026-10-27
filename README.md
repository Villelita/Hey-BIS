# Hey BIS

Proyecto 1 del documento local `nanoAI.pdf`: detectar la frase **Hey BIS** en español
e inglés y distinguirla de otras palabras y del ruido de fondo.

Empezamos por la **etapa 1: preparar y validar los datos**. Todavía no hay
modelo, entrenamiento ni detector en vivo.

## Dónde corre cada cosa

```text
Raspberry Pi                         Computadora
audio WAV + manifest.csv  ------->   validación y lectura por bloques
almacenamiento por red               después: entrenamiento y detección
```

La Raspberry solo almacena archivos y tiene **4 GB de RAM**. La meta del
proyecto es almacenar allí **10 GB de datos de audio**. Se supone una carpeta compartida ya
montada en la computadora mediante SMB o una unidad de red. El programa se
ejecuta en la computadora y recibe la ruta de esa carpeta; no copia todo el
dataset al disco local. Aún no se ha configurado ni probado la conexión a la Pi.

El PDF propone ejecutar el modelo en la Raspberry. Aquí seguimos el cambio
solicitado: procesamiento en la computadora. Conviene dejar esta adaptación
explicada en la entrega.

## Tamaño y memoria

- Meta del dataset: **10 GB = 10,000,000,000 bytes** de WAV válidos incluidos
  en el manifiesto, configurada en `dataset_target_bytes`. Se cuenta cada ruta
  una vez, con su encabezado WAV. Es una meta que debemos alcanzar; se permite
  superarla. `bytes_remaining` muestra cuánto falta y `progress_percent` el
  avance. `storage_bytes` informa aparte el tamaño de toda la carpeta, incluidas
  licencias, metadatos y archivos que no cuentan para la meta.
- RAM de la Pi: **4 GB = 4,000,000,000 bytes**, registrada en `pi_ram_bytes`.
  Es el recurso de la Raspberry, que sirve archivos. No limita la RAM del
  entrenamiento en la computadora ni significa que haya que consumir 4 GB.
- La validación en la computadora lee como máximo **16 MB de audio por bloque**,
  configurados en `read_block_bytes`; no equivale a la RAM total del proceso.
- La Raspberry necesita espacio para esos 10 GB de audio, metadatos y sistema
  operativo. Su capacidad real de almacenamiento aún no se ha comprobado.
  La configuración documenta la meta y la RAM prevista; falta medir espacio
  libre y uso de RAM en la Raspberry real.
- A 16 kHz, mono y 16 bits, 10 GB equivalen aproximadamente a **86.8 horas
  de audio**, sin contar encabezados. Se sumarán grabaciones variadas de la
  frase, otras palabras y fondos. Las copias del mismo contenido no aportan
  nuevos ejemplos y el validador todavía no detecta esas copias por contenido.

## Audios de ESC-50 incorporados

El workspace del editor incluye dos carpetas: `Hey BIS` y `ESC-50-master`.
Los audios originales están en la subcarpeta del dataset:

```text
ESC-50-master/audio/
```

Se incorporan **2,000 clips de fondo**, con 5 segundos cada uno, al directorio
`data/audio/esc50/`. Sirven como ejemplos negativos: el detector debe ignorarlos.
No contienen ejemplos etiquetados de “Hey BIS” ni sustituyen grabaciones de
conversación en español e inglés. Las categorías de origen son sonidos
ambientales y humanos sin habla; conviene escuchar una muestra para revisar
que no haya voces incidentales ni etiquetas dudosas.

La importación genera copias WAV PCM mono a 16 kHz y 16 bits, unos **320 MB**
en total (2 h 46 min 40 s de audio). Los originales de 44.1 kHz se conservan.
Se usa [remuestreo con filtro de SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html).

Los folds 1-3 se asignan a entrenamiento, el 4 a validación y el 5 a prueba.
Todos los fragmentos con el mismo `src_file` permanecen juntos, incluso si los
metadatos originales los colocan en distintos folds. En el manifiesto,
`esc50_<src_file>` representa un grupo de grabación, no una persona identificada.
No mezcles fondos de prueba con grabaciones de entrenamiento al aumentar datos.

La procedencia se conserva en `data/sources/esc50/`:

- `esc50.csv`: categorías y folds originales.
- `provenance.csv`: correspondencia entre archivos y particiones del proyecto.
- `LICENSE`: condiciones del dataset y atribuciones por clip.
- `ATTRIBUTION.md`: autor, fuente y transformaciones realizadas.

Para reproducir la importación desde la computadora:

```powershell
python -m pip install -r requirements.txt
python import_esc50.py --source-dir "$env:USERPROFILE\Downloads\ESC-50-master\ESC-50-master" --data-dir .\data
python main.py inspect --data-dir .\data
```

Repetir la importación conserva las grabaciones propias del manifiesto y no
duplica los registros ESC-50. Si un archivo de destino tiene otro contenido,
se detiene sin sobrescribirlo. Las dependencias de conversión se instalan
solo en la computadora; no hacen falta en la Raspberry para compartir archivos.

## Primeros pasos

Requiere Python 3.11 o superior. El validador usa la biblioteca estándar;
la importación ESC-50 usa NumPy y SciPy, incluidas en `requirements.txt`.

```powershell
python main.py plan
```

Para conocer el flujo con una carpeta local de prueba:

```powershell
python main.py init --data-dir .\data
```

Esto crea `data/audio/` y un `data/manifest.csv` con sus columnas si no existe.
Ejecutarlo otra vez conserva el manifiesto y los audios existentes.
Guarda tus primeras grabaciones WAV en `data/audio/` y añade sus filas:

```csv
path,label,language,speaker_id,split,consent
audio/p001_es_001.wav,wake_word,es,p001,train,yes
audio/p002_en_001.wav,other_speech,en,p002,validation,yes
audio/p003_es_001.wav,wake_word,es,p003,test,yes
audio/salon_001.wav,background,none,salon01,train,yes
```

Los nombres son ejemplos: deben corresponder a archivos que hayas grabado.
Formato obligatorio: **WAV PCM, mono, 16,000 Hz, 16 bits**. `other_speech`
incluye conversación, otras frases y palabras parecidas a “Hey BIS”. Graba
ambos idiomas y distintas personas, distancias y condiciones de ruido.

Usa identificadores anónimos de participantes y registra `yes` únicamente si
cuentas con su permiso. En audio de fondo propio con voces también hace falta
permiso. Para ESC-50 se registra `licensed`, conservando su licencia; este valor
no afirma que se obtuvo consentimiento de personas del dataset externo.
Cada persona debe permanecer en una sola partición: `train`, `validation` o
`test`. Para ruido sin voces usa un ID de sesión y conserva esa sesión en una
sola partición. El validador detecta cruces de ID y rutas repetidas; no detecta
copias del mismo contenido con otro nombre ni voces mal etiquetadas.

```powershell
python main.py inspect --data-dir .\data
```

Lee el audio por bloques, detecta archivos truncados y reporta cantidad de
archivos, bytes válidos, horas y distribución por etiqueta y partición.
`missing_labels` indica las clases que todavía faltan. El validador comprueba
integridad y avance hacia 10 GB; aprobarlo no significa que ya se pueda entrenar.
Las grabaciones propias se añaden manualmente al manifiesto. Un error de datos
produce código de salida 1. Durante la recolección se permite estar por debajo
de la meta; para exigir que esté cumplida:

```powershell
python main.py inspect --data-dir .\data --require-target
```

Este comando devuelve código de salida 1 si todavía faltan datos, aunque los
audios existentes sean válidos. Con los 2,000 clips ESC-50 actuales hay
320,088,000 bytes de WAV: **3.2009 %** de la meta. Faltan **9,679,912,000 bytes**
(unos **9.68 GB**) de audio válido registrado en el manifiesto.

Cuando esté montada la carpeta de la Raspberry, usa su ruta en lugar de
`.\data`. Por ejemplo, **solo si ese recurso existe en tu red**:

```powershell
python main.py inspect --data-dir "\\raspberrypi\hey-bis"
```

La revisión completa lee todos los WAV registrados a través de la red.
La carpeta completa `data/` (incluidos manifiesto y atribuciones) es la que se
copiará a la Raspberry. Todavía no se ha realizado esa transferencia.

## Siguientes etapas

1. Grabar la frase “Hey BIS” (`wake_word`) y otras frases (`other_speech`), en
   ambos idiomas, con distintas personas. Para el piloto, comenzar con unas
   20-30 tomas por persona y clase, en clips de 1-2 segundos. Mantener cada
   persona en una sola partición y cubrir las clases en las tres particiones.
   Es un punto de partida para probar el flujo, no una garantía de precisión.
2. Añadir las filas al manifiesto y ejecutar `python main.py inspect --data-dir .\data`.
   Continuar la recolección hasta alcanzar 10 GB, revisando el avance por etapas.
   Conservar conversación continua aparte para medir falsas activaciones por hora.
3. Configurar una carpeta compartida en la Raspberry, copiar `data/` y comprobar
   acceso desde la computadora. En la Pi, revisar `df -h` para espacio y `free -h`
   para RAM; verificar también consumo durante una transferencia. Sus 4 GB incluyen
   el sistema operativo, el servicio de archivos y su caché.
4. Preparar ventanas y características de audio en la computadora y entrenar una
   red pequeña desde cero. La Raspberry seguirá sirviendo archivos.
5. Evaluar con personas no vistas: proporción de activaciones correctas y falsas
   activaciones por hora de conversación continua. Guardar mediciones para la demo
   y el reporte de dos páginas que pide el PDF.

La revisión de datos no demuestra que el detector funcione. No se han medido
precisión, falsas activaciones, consumo de RAM ni rendimiento en la Pi.

## Archivos locales y privacidad

El PDF del curso, los audios, el dataset, las salidas, los temporales y los
archivos de secretos o credenciales se mantienen fuera de Git mediante
`.gitignore`. Las rutas de ejemplo usan variables de entorno, sin nombres de
usuario locales. Los ejemplos `.env.example` y `.env.template` pueden versionarse
si contienen únicamente valores ficticios. El PDF sigue disponible localmente.

Pruebas de esta etapa:

```powershell
python -m unittest discover -s tests -v
```
