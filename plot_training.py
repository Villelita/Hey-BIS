"""Generar una curva de perdida desde un historial real de entrenamiento."""

import argparse
import json
from pathlib import Path


def plot_history(history_path: Path, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = json.loads(history_path.read_text(encoding="utf-8"))
    history = report["history"]
    if not history:
        raise ValueError("El historial no contiene epocas.")
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = [r["epoch"] for r in history]
    ax.plot(epochs, [r["train_loss"] for r in history], label="Entrenamiento")
    ax.plot(epochs, [r["validation_loss"] for r in history], label="Validacion")
    ax.set(xlabel="Epoca", ylabel="Entropia cruzada", title="Hey BIS: perdida por epoca")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/loss.svg"))
    args = parser.parse_args()
    try:
        plot_history(args.history, args.output)
    except ImportError:
        parser.exit(1, "Instala la dependencia opcional: python -m pip install -r requirements-plots.txt\n")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
