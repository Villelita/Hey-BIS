# Hey BIS - Servicio Web & Frontend en Red Local (LAN)

Servicio web ligero para interactuar con la red neuronal y clasificador de audio de **Hey BIS** directamente desde cualquier navegador conectado a la red local (computadora, laptop, tablet o teléfono celular).

---

## 🚀 Inicio Rápido

Para iniciar el servidor en tu red local:

```bash
# Opción 1: Ejecutar directamente con Python
python3 service/server.py

# Opción 2: Usar el script de ejecución
./service/run.sh
```

El servidor se enlaza automáticamente a `0.0.0.0` (todas las interfaces de red) en el puerto `8080` (o `8000` si está ocupado).

Al iniciar, verás en consola las URLs exactas para conectarte:
```text
================================================================
    HEY BIS - SERVICIO WEB DE DETECCIÓN DE WAKE WORD & AUDIO    
================================================================
  * Modelo:          3136 -> 64 (ReLU) -> 32 (ReLU) -> 3 (Softmax)
  * Estado Pesos:    Pesos inicializados (He/Glorot)
  * Umbral Activo:   0.60
  * Acceso Local:    http://localhost:8080
  * Red Local (LAN): http://192.168.1.XX:8080
================================================================
```

---

## 🌐 Conexión desde Cualquier Navegador en la Red Local

1. Asegúrate de que el dispositivo (celular, laptop, tablet) esté conectado a la **misma red Wi-Fi o red LAN** que el servidor.
2. Abre cualquier navegador (Chrome, Safari, Firefox, Edge).
3. Ingresa la dirección IP que mostró la consola:
   ```text
   http://<TU_IP_LOCAL>:8080
   ```
   *(Ejemplo: `http://10.100.0.122:8080`)*

---

## 🎙️ Funcionalidades del Frontend

- **Micrófono en Vivo**: Graba tu voz directamente desde el navegador (1.0s, 1.5s, 2.0s o manual) y visualiza la forma de onda en tiempo real mediante un osciloscopio en HTML5 Canvas.
- **Subida de Archivos WAV**: Arrastra y suelta grabaciones `.wav` para clasificarlas al instante con reproductor integrado.
- **Pruebas Rápidas con Señales Sintéticas**: Botones para generar y probar de inmediato Silencio puro, Ruido blanco ambiental y Tonos a 440 Hz sin necesidad de micrófono.
- **Resultados en Tiempo Real**:
  - Veredicto visual destacado con animación si se detecta `"Hey BIS"`.
  - Barras porcentuales de probabilidad para cada una de las 3 clases (`wake_word`, `other_speech`, `background`).
  - Métricas de latencia de inferencia en milisegundos y duración del clip.
- **Ajuste Dinámico de Umbral**: Control deslizante para variar el umbral de activación (0.10 a 0.95) en caliente.

> [!NOTE]
> **Permisos de Micrófono en HTTP:**
> Los navegadores modernos restringen el acceso al micrófono vía Web Audio API en sitios que no sean `localhost` o `HTTPS`. Si accedes desde un teléfono móvil por `http://<IP_LOCAL>:8080`:
> - Puedes usar la pestaña **"Archivo WAV"** o **"Pruebas Rápidas"** para probar sin restricciones.
> - O en Chrome de Android/PC, puedes habilitar temporalmente la IP en `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.

---

## ⚙️ Argumentos de Línea de Comandos

```bash
python3 service/server.py --help
```

| Parámetro | Por Defecto | Descripción |
|---|---|---|
| `--host` | `0.0.0.0` | Dirección de escucha (`0.0.0.0` para acceso LAN completo) |
| `--port` | `8080` | Puerto HTTP del servidor |
| `--weights` | `None` | Ruta a archivo `.npz` de pesos entrenados (ej. `wake_word_model.npz`) |
| `--threshold` | `0.60` | Umbral mínimo de confianza para activar el Wake Word |

---

## 📡 Endpoints de la API REST

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/` | Interfaz Web (Frontend) |
| `GET` | `/api/status` | Estado del servidor, arquitectura del modelo y URLs de red |
| `POST` | `/api/classify` | Envío de audio (WAV binario, Multipart o JSON base64) para inferencia |
| `POST` | `/api/threshold` | Actualización dinámica del umbral de detección |
| `GET` | `/api/test-audio?type=...` | Genera y descarga un WAV sintético (`silence`, `noise`, `tone`) |

