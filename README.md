# Hey BIS

Project 1 from the local course specification `nanoAI.pdf`: detecting the wake word phrase **"Hey BIS"** in both Spanish and English, distinguishing it from other spoken words and environmental background noise.

---

## System Architecture: Where Each Component Runs

```text
Raspberry Pi (Storage / SMB)                Host Computer (Processing & AI)
WAV Audio + manifest.csv       ------->    Block-by-block data validation
Network-attached storage                   Log-Mel Spectrogram feature extraction
                                           Neural network training & live inference
```

- **Raspberry Pi**: Acts as file storage and has **4 GB of RAM**. The project target is to store **10 GB of audio data** on the Pi. It is assumed to be accessible as a shared folder mounted on the computer via SMB or network drive. The program runs on the host computer and accesses that folder path directly without copying the full dataset to the local drive.
- **Host Computer**: Handles data validation, feature extraction (Log-Mel Spectrograms), pure NumPy neural network training, inference, and profiling.

---

## Target Size, Memory, and Storage Limits

- **Dataset Storage Target**: **10 GB = 10,000,000,000 bytes** of valid WAV audio registered in the manifest, configured in `dataset_target_bytes`. Each unique path counts towards the goal with its WAV header. `bytes_remaining` and `progress_percent` track progress.
- **Raspberry Pi RAM**: **4 GB = 4,000,000,000 bytes**, tracked in `pi_ram_bytes`.
- **Block Validation Memory**: Reads audio in maximum blocks of **16 MB**, configured in `read_block_bytes` (streaming read to avoid memory exhaustion).
- **Audio Duration Equivalent**: At 16 kHz, mono, 16-bit PCM, 10 GB corresponds to approximately **86.8 hours of audio**.

---

## ESC-50 Background Audio Integration

The workspace includes 2,000 environmental background sound clips (5 seconds each) from ESC-50 in `data/audio/esc50/`. They serve as negative samples (`background` label) that the wake word detector must ignore.

Conversion produces 16 kHz, 16-bit mono PCM copies (~320 MB total, 2h 46m 40s of audio), keeping original 44.1 kHz files intact.

Provenance is tracked in `data/sources/esc50/`:
- `esc50.csv`: Original categories and folds.
- `provenance.csv`: Mapping between files and project splits (`train`, `validation`, `test`).
- `LICENSE` & `ATTRIBUTION.md`: Licensing and clip attributions.

---

## Neural Network Architecture (Lightweight Wake Word Model)

Built in pure NumPy without heavy external ML framework dependencies, making it directly runnable on standard CPUs and lightweight single-board computers (Raspberry Pi):

- **Input Layer**: 3,136 features (Log-Mel Spectrogram: 32 frequency bands × 98 temporal frames from 1-second 16 kHz mono audio).
- **Hidden Layer 1**: 64 neurons + ReLU activation.
- **Hidden Layer 2**: 32 neurons + ReLU activation.
- **Output Layer**: 3 neurons + Softmax Cross-Entropy:
  - Class 0: `background`
  - Class 1: `other_speech`
  - Class 2: `wake_word`
- **Optimizer**: Adam Optimizer implemented from scratch (first and second moment estimation $m$ and $v$ with bias correction).

---

## How to Check Weight and Storage Footprint (`profile_model.py`)

A dedicated benchmarking and profiling tool is available to inspect model parameters, RAM consumption, disk footprint, inference latency, and dataset progress toward the 10 GB goal.

### Running the Profile Tool:

From the project root:
```powershell
python profile_model.py
```

Or if your terminal is inside the `tests/` directory:
```powershell
python ../profile_model.py
```

### What the Report Analyzes:

1. **Trainable Parameters**:
   - `capa_oculta_1`: 200,768 parameters (784.25 KB)
   - `capa_oculta_2`: 2,080 parameters (8.12 KB)
   - `capa_salida`: 99 parameters (396 B)
   - **Total parameters**: **202,947 parameters**
2. **RAM Footprint (Inference)**:
   - Float32 precision: **~792.76 KB** (811,788 bytes).
   - Activations buffer: ~12.64 KB per sample.
   - Less than **0.02% of the Raspberry Pi's 4 GB RAM**.
3. **RAM Footprint (Training)**:
   - Weights + Gradients + Adam moment buffers ($m$, $v$): **~3.097 MB**.
4. **Disk Model Weight**:
   - Compressed `.npz` binary: **~737 KB** (`np.savez_compressed`).
   - Uncompressed `.npz`: ~794 KB.
5. **CPU Inference Latency**:
   - **~0.02 ms** per 1-second audio window (~45,000+ inferences/second on CPU).
6. **Storage Audit vs 10 GB Target**:
   - Current audio WAV files count and size in MB.
   - Total `data/` folder size.
   - Progress percentage towards `10,000,000,000 bytes` (10 GB).
   - Remaining storage required and estimated audio hours left to record.

---

## Training the Wake Word Model

To train the neural network using the dataset registered in `data/manifest.csv`:

```powershell
python train_wake_word.py --epochs 15 --batch-size 32 --lr 0.001 --save-path wake_word_model.npz
```

Available arguments:
- `--data-dir`: Path to dataset folder (default: `data`).
- `--epochs`: Number of training epochs (default: `10`).
- `--batch-size`: Mini-batch size (default: `32`).
- `--lr`: Learning rate for Adam optimizer (default: `0.001`).
- `--save-path`: Output path for saved weights (default: `wake_word_model.npz`).

---

## Data Management & Inspection CLI (`main.py`)

### 1. Initialize dataset structure:
```powershell
python main.py init --data-dir .\data
```
Creates `data/audio/` and `data/manifest.csv` with standard columns:
```csv
path,label,language,speaker_id,split,consent
audio/p001_es_001.wav,wake_word,es,p001,train,yes
audio/p002_en_001.wav,other_speech,en,p002,validation,yes
audio/salon_001.wav,background,none,salon01,train,yes
```

Format requirements: **WAV PCM, mono, 16,000 Hz, 16 bits**.

### 2. Inspect dataset integrity and progress:
```powershell
python main.py inspect --data-dir .\data
```
Validates chunk by chunk, checks for corrupted/truncated files, cross-partition speaker leaks, and reports current byte count towards the 10 GB target.

To enforce that the target is met before proceeding:
```powershell
python main.py inspect --data-dir .\data --require-target
```

### 3. When mounted on Raspberry Pi:
```powershell
python main.py inspect --data-dir "\\raspberrypi\hey-bis"
```

---

## Running Automated Tests

Run the full unit test suite (including neural network math, audio features, model serialization, evaluation, and data validation):

```powershell
python -m unittest discover tests
```

Or run individual test modules directly:
```powershell
python tests/test_neural_network.py
python tests/test_data.py
python tests/test_import_esc50.py
```

---

## Privacy and Ignored Files

All heavy data, raw audio WAVs, local models, `.env` credentials, and the course PDF (`nanoAI.pdf`) are excluded from Git via `.gitignore`. Sample templates (`.env.example`) may be versioned if they contain only dummy placeholders.
