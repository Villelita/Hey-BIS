"""Red neuronal para detección de Wake Word ("Hey BIS") y clasificación de ruido.

Implementación en NumPy puro de las neuronas artificiales (capas densas, ReLU, Softmax,
propagación hacia adelante, retropropagación y optimizador Adam).

Diseñada para ser ligera y ejecutable tanto en PC como directamente en una Raspberry Pi
sin requerir dependencias pesadas de GPU o librerías de gran tamaño.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np

import audio_features


# Mapeo oficial de etiquetas según el manifiesto del proyecto Hey BIS
LABELS: List[str] = ["background", "other_speech", "wake_word"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(LABELS)}


class DenseLayer:
    """Capa de neuronas densamente conectadas (Fully Connected).
    
    Cada neurona realiza la operación: z = X @ W + b
    donde:
      X = vector de entrada
      W = matriz de pesos sinápticos
      b = vector de sesgo (bias)
    """

    def __init__(self, input_dim: int, output_dim: int, name: str = "dense"):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Inicialización He (Kaiming Normal) adecuada para activaciones ReLU
        std = np.sqrt(2.0 / input_dim)
        self.weights = (np.random.randn(input_dim, output_dim) * std).astype(np.float32)
        self.bias = np.zeros((1, output_dim), dtype=np.float32)

        # Gradientes acumulados en backpropagation
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

        # Variables temporales para backpropagation y optimizador Adam
        self.last_input: np.ndarray | None = None
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.bias)
        self.v_b = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Paso hacia adelante: cálculo lineal de las neuronas."""
        self.last_input = x
        return np.matmul(x, self.weights) + self.bias

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Paso hacia atrás: cálculo de gradientes respecto a entradas y pesos."""
        if self.last_input is None:
            raise RuntimeError("Se debe ejecutar forward antes de backward.")
        
        # dL/dW = X.T @ grad_output
        self.grad_weights = np.matmul(self.last_input.T, grad_output)
        # dL/db = suma sobre los ejemplos del batch
        self.grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        # dL/dX = grad_output @ W.T
        grad_input = np.matmul(grad_output, self.weights.T)
        return grad_input


class ReLU:
    """Función de activación Rectified Linear Unit: f(x) = max(0, x)."""

    def __init__(self):
        self.last_input: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_input = x
        return np.maximum(0.0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.last_input is None:
            raise RuntimeError("Se debe ejecutar forward antes de backward.")
        return grad_output * (self.last_input > 0).astype(np.float32)


class SoftmaxCrossEntropy:
    """Combina función Softmax y pérdida de entropía cruzada con estabilidad numérica."""

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        # Se resta el máximo por fila para evitar overflow exponencial
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

    @staticmethod
    def loss(probs: np.ndarray, targets: np.ndarray) -> float:
        """Calcula la pérdida promedio sobre el batch.
        
        targets puede ser índices enteros de clase (N,) o matriz one-hot (N, C).
        """
        batch_size = len(probs)
        clipped_probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
        if targets.ndim == 1:
            log_likelihood = -np.log(clipped_probs[np.arange(batch_size), targets])
        else:
            log_likelihood = -np.sum(targets * np.log(clipped_probs), axis=-1)
        return float(np.mean(log_likelihood))

    @staticmethod
    def gradient(probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Gradiente de la pérdida respecto a los logits de entrada: (probs - targets) / N."""
        batch_size = len(probs)
        grad = probs.copy()
        if targets.ndim == 1:
            grad[np.arange(batch_size), targets] -= 1.0
        else:
            grad -= targets
        return grad / batch_size


class AdamOptimizer:
    """Optimizador Adam para actualizar pesos de las neuronas."""

    def __init__(
        self,
        layers: List[DenseLayer],
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.layers = layers
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.t = 0

    def step(self):
        self.t += 1
        for layer in self.layers:
            # Actualización de pesos W
            layer.m_w = self.beta1 * layer.m_w + (1.0 - self.beta1) * layer.grad_weights
            layer.v_w = self.beta2 * layer.v_w + (1.0 - self.beta2) * (layer.grad_weights ** 2)
            m_hat_w = layer.m_w / (1.0 - (self.beta1 ** self.t))
            v_hat_w = layer.v_w / (1.0 - (self.beta2 ** self.t))
            layer.weights -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.eps)

            # Actualización de sesgo b
            layer.m_b = self.beta1 * layer.m_b + (1.0 - self.beta1) * layer.grad_bias
            layer.v_b = self.beta2 * layer.v_b + (1.0 - self.beta2) * (layer.grad_bias ** 2)
            m_hat_b = layer.m_b / (1.0 - (self.beta1 ** self.t))
            v_hat_b = layer.v_b / (1.0 - (self.beta2 ** self.t))
            layer.bias -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)


class WakeWordNeuralNetwork:
    """Red Neuronal Artificial para la detección de 'Hey BIS' y ruidos ambientales.
    
    Arquitectura por defecto:
      - Entrada: 3136 características (Espectrograma Log-Mel: 32 bandas x 98 frames)
      - Capa 1: 64 neuronas + activación ReLU
      - Capa 2: 32 neuronas + activación ReLU
      - Capa de Salida: 3 neuronas (background, other_speech, wake_word) + Softmax
    """

    def __init__(
        self,
        input_dim: int = 3136,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        num_classes: int = 3,
    ):
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.num_classes = num_classes

        # Capas del modelo
        self.dense1 = DenseLayer(input_dim, hidden_dim1, name="capa_oculta_1")
        self.relu1 = ReLU()
        self.dense2 = DenseLayer(hidden_dim1, hidden_dim2, name="capa_oculta_2")
        self.relu2 = ReLU()
        self.dense3 = DenseLayer(hidden_dim2, num_classes, name="capa_salida")

        self.dense_layers: List[DenseLayer] = [self.dense1, self.dense2, self.dense3]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Calcula los logits sin normalizar."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        z1 = self.dense1.forward(x)
        a1 = self.relu1.forward(z1)
        z2 = self.dense2.forward(a1)
        a2 = self.relu2.forward(z2)
        logits = self.dense3.forward(a2)
        return logits

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Devuelve las probabilidades de cada clase (Softmax)."""
        logits = self.forward(x)
        return SoftmaxCrossEntropy.softmax(logits)

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Devuelve las clases predichas (índice 0, 1 o 2) y las probabilidades máximas."""
        probs = self.predict_proba(x)
        class_indices = np.argmax(probs, axis=-1)
        confidences = np.max(probs, axis=-1)
        return class_indices, confidences

    def classify_audio_samples(
        self,
        samples: np.ndarray,
        wake_threshold: float = 0.60,
    ) -> dict:
        """Analiza un arreglo de muestras de audio (16 kHz) y clasifica si es ruido o Wake Word."""
        features = audio_features.extract_features(samples)
        probs = self.predict_proba(features)[0]

        pred_idx = int(np.argmax(probs))
        prob_dict = {
            LABELS[0]: float(probs[0]),
            LABELS[1]: float(probs[1]),
            LABELS[2]: float(probs[2]),
        }

        # Regla de seguridad: para despertar ("wake_word"), se exige superar el umbral
        wake_word_prob = float(probs[LABEL_TO_IDX["wake_word"]])
        is_wake = wake_word_prob >= wake_threshold

        detected_label = "wake_word" if is_wake else LABELS[pred_idx]

        return {
            "label": detected_label,
            "is_wake_word": is_wake,
            "confidence": float(probs[pred_idx]),
            "wake_word_probability": wake_word_prob,
            "probabilities": prob_dict,
        }

    def train_step(
        self,
        batch_x: np.ndarray,
        batch_y: np.ndarray,
        optimizer: AdamOptimizer,
    ) -> float:
        """Ejecuta un ciclo completo de entrenamiento (Forward -> Loss -> Backward -> Update)."""
        # 1. Forward pass
        logits = self.forward(batch_x)
        probs = SoftmaxCrossEntropy.softmax(logits)

        # 2. Cálculo de pérdida
        loss = SoftmaxCrossEntropy.loss(probs, batch_y)

        # 3. Backward pass (cálculo de gradientes)
        grad_logits = SoftmaxCrossEntropy.gradient(probs, batch_y)
        grad_a2 = self.dense3.backward(grad_logits)
        grad_z2 = self.relu2.backward(grad_a2)
        grad_a1 = self.dense2.backward(grad_z2)
        grad_z1 = self.relu1.backward(grad_a1)
        _ = self.dense1.backward(grad_z1)

        # 4. Actualización de pesos
        optimizer.step()

        return loss

    def get_parameter_count(self) -> int:
        """Calcula el total de pesos y sesgos entrenables en la red."""
        total = 0
        for layer in self.dense_layers:
            total += layer.weights.size + layer.bias.size
        return total

    def summary(self) -> dict:
        """Resumen técnico de la arquitectura y uso de memoria para Raspberry Pi."""
        params = self.get_parameter_count()
        # Tamaño en memoria asumiendo float32 (4 bytes por parámetro)
        weights_bytes = params * 4
        return {
            "model_name": "HeyBIS_WakeWord_MLP",
            "input_dimension": self.input_dim,
            "layers": [
                {"name": self.dense1.name, "neurons": self.hidden_dim1, "params": self.dense1.weights.size + self.dense1.bias.size},
                {"name": self.dense2.name, "neurons": self.hidden_dim2, "params": self.dense2.weights.size + self.dense2.bias.size},
                {"name": self.dense3.name, "neurons": self.num_classes, "params": self.dense3.weights.size + self.dense3.bias.size},
            ],
            "total_parameters": params,
            "weights_memory_bytes": weights_bytes,
            "weights_memory_kb": round(weights_bytes / 1024, 2),
            "classes": LABELS,
        }

    def save_weights(self, filepath: str | Path):
        """Guarda los pesos entrenados en un archivo comprimido .npz."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            w1=self.dense1.weights,
            b1=self.dense1.bias,
            w2=self.dense2.weights,
            b2=self.dense2.bias,
            w3=self.dense3.weights,
            b3=self.dense3.bias,
        )

    def load_weights(self, filepath: str | Path):
        """Carga los pesos guardados desde un archivo .npz."""
        data = np.load(str(filepath))
        self.dense1.weights = data["w1"]
        self.dense1.bias = data["b1"]
        self.dense2.weights = data["w2"]
        self.dense2.bias = data["b2"]
        self.dense3.weights = data["w3"]
        self.dense3.bias = data["b3"]
