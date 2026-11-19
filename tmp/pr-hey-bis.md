El flujo anterior podía descartar todos los audios incompatibles sin fallar y realizar inferencia con pesos aleatorios. Este cambio valida los datos antes de entrenar, exige pesos entrenados y añade el flujo de captura y evaluación necesario para probar Hey BIS en computadora y Raspberry Pi.

### Cambios
- Preparación de copias WAV a 16 kHz con respaldo del manifiesto y conservación de originales.
- Validación de clases, idiomas y particiones; detección de contenido duplicado entre particiones y huellas para comprobar que los datos no cambien tras entrenar.
- Entrenamiento reproducible con historial, selección de época y umbral mediante validación, metadatos del modelo y test reservado para evaluación final.
- Clasificación coherente con el umbral, cobertura de la cola del audio y agrupación de ventanas positivas en eventos.
- Grabación con consentimiento, escucha desde micrófono y registro de detecciones.
- Evaluación de sensibilidad por idioma y falsas activaciones por hora, distinguiendo conversación normal de ruido ambiental.
- Perfil de memoria y tiempos, generación opcional de curvas e instrucciones de instalación y ejecución.
- Exclusión de credenciales, instrucciones locales de asistentes, audios, modelos y resultados mediante `.gitignore`. Los TXT añadidos solo declaran dependencias opcionales.

### Validación
- 38 pruebas automatizadas aprobadas, incluido un ciclo completo con datos sintéticos y captura simulada.
- Los 2,000 WAV ambientales locales fueron convertidos y pasan la validación; originales conservados. Los audios y respaldos permanecen fuera del repositorio.
- Verificada la compatibilidad del micrófono local con mono a 16 kHz, sin iniciar captura.
- `git diff --check` y comprobación de dependencias sin errores.
- Revisión de archivos candidatos sin encontrar tokens, claves privadas ni rutas personales.

### Pendiente para demostrar reconocimiento real
Faltan grabaciones propias de Hey BIS y otras palabras en español e inglés, el entrenamiento con esas voces y las mediciones físicas en Raspberry Pi. Las pruebas sintéticas verifican el software y no representan exactitud de reconocimiento de voz.
