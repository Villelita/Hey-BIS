El proyecto partía de un script vacío. Esta primera etapa permite importar ESC-50 como audio de fondo, validar los WAV y el manifiesto, y medir el avance hacia una meta de 10 GB de audio almacenado en la Raspberry Pi.

La Raspberry almacena los archivos y tiene 4 GB de RAM; la computadora realiza la validación y, en etapas posteriores, el entrenamiento y la detección. La importación convierte las copias a PCM mono de 16 kHz y 16 bits, conserva las atribuciones y agrupa fragmentos de una misma fuente en una sola partición.

Incluye código, configuración, instrucciones y pruebas. El PDF del curso, audios, dataset, salidas, temporales, credenciales y claves quedan excluidos de Git; los ejemplos usan variables de entorno en lugar de rutas con nombres de usuario.

Validación: `python -m unittest discover -s tests -v` (14 pruebas aprobadas), `git diff --check` y revisión local de 2,000 clips con `python main.py inspect --data-dir .\data` (sin errores). El dataset actual contiene 320,088,000 bytes de WAV, un 3.2009 % de la meta; todavía faltan ejemplos de wake_word y other_speech.

El alcance se limita a preparar datos. Aún no hay modelo entrenado, detección en vivo ni conexión comprobada a la Raspberry.
