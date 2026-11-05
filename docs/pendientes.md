# Acuerdos y pendientes de Hey BIS

Actualizado el 4 de noviembre de 2026. Este documento conserva las decisiones acordadas con el equipo para futuras revisiones de lo que falta.

## Decisiones vigentes

- La red detecta la frase Hey BIS en español e inglés y clasifica también otras voces y ruido ambiental.
- Ventana provisional de **3 segundos**, a 16 kHz mono y 16 bits: 48,000 muestras y 9,536 características Log-Mel. Grabación, entrenamiento, inferencia y evaluación utilizan esa misma duración.
- El equipo recolectará los positivos: personas distintas, ambientes distintos y ruido de fondo real que acompañe la frase.
- El equipo también recolectará negativos de voz y ejemplos difíciles, como palabras o frases parecidas. Los audios ambientales existentes conservan su utilidad como negativos de fondo.
- Cada hablante debe quedar en una sola partición. Validation se usa para ajustar el modelo y el umbral; test se reserva para evaluar el resultado final.

## Pendientes diferidos hasta tener los audios

| Pendiente | Responsable o condición |
| --- | --- |
| Recolectar y etiquetar positivos variados en ambos idiomas | Equipo |
| Recolectar otras voces, conversación y negativos difíciles | Equipo |
| Comprobar que tres segundos contienen la frase completa, sin cortes | Después de recibir los audios reales; la suficiencia todavía no está validada |
| Revisar cobertura de hablantes, ambientes, idiomas y clases por partición | Con el primer dataset de voz |
| Comprobar activaciones sobre frases con ruido de fondo | Con positivos reales; todavía no hay una medición de robustez |
| Entrenar y ajustar con train y validation; guardar modelo e historial | Después de completar y validar los datos |
| Medir sensibilidad y falsas activaciones por hora con datos reservados | Con el modelo entrenado y conversación de prueba |
| Verificar escucha, tiempos y memoria en Raspberry Pi | Cuando estén disponibles el modelo y el hardware |
| Completar demo, tabla de resultados, informe y bitácoras | Después de obtener mediciones reales |

Estos pendientes siguen abiertos aunque las pruebas sintéticas del código pasen. No hay que presentarlos como errores del software ya corregidos ni como trabajo completado antes de comprobarlo.
