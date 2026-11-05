"""Perfil de recursos en el equipo actual; pesos y memoria del proceso se distinguen."""

import argparse
import json
from pathlib import Path
import platform
import tempfile
import time

import numpy as np

import audio_features
from main import inspect_dataset, load_config
from neural_network import WakeWordNeuralNetwork

ROOT_DIR = Path(__file__).resolve().parent


def process_memory():
    if platform.system() == 'Windows':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {'rss_bytes': counters.WorkingSetSize, 'peak_rss_bytes': counters.PeakWorkingSetSize}
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {'peak_rss_bytes': int(peak * (1 if platform.system() == 'Darwin' else 1024))}
    status = Path('/proc/self/status')
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith('VmRSS:'):
                result['rss_bytes'] = int(line.split()[1]) * 1024
    return result


def inspect_model_size(input_dim=audio_features.FEATURE_DIM, hidden_dim1=64, hidden_dim2=32, num_classes=3,
                       pi_ram_bytes=4_000_000_000, weights_path=None):
    model = (WakeWordNeuralNetwork.from_weights(weights_path) if weights_path else
             WakeWordNeuralNetwork(input_dim, hidden_dim1, hidden_dim2, num_classes, trainable=False))
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 0.05, audio_features.WINDOW_SAMPLES).astype(np.float32)
    features = audio_features.extract_features(samples)
    for _ in range(10):
        model.classify_audio_samples(samples)
    forward_times, full_times = [], []
    for _ in range(100):
        start = time.perf_counter()
        model.predict_proba(features)
        forward_times.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        model.classify_audio_samples(samples)
        full_times.append((time.perf_counter() - start) * 1000)
    params = model.get_parameter_count()
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'profile.npz'
        model.save_weights(path)
        disk_bytes = path.stat().st_size
    arrays = [a for layer in model.dense_layers for a in (
        layer.weights, layer.bias, layer.grad_weights, layer.grad_bias,
        layer.m_w, layer.v_w, layer.m_b, layer.v_b) if a is not None]
    return {'hardware': platform.platform(), 'processor': platform.processor(),
            'python': platform.python_version(), 'numpy': np.__version__,
            'model_status': 'trained' if weights_path else 'untrained_architecture_only',
            'architecture': model.summary(), 'total_parameters': params,
            'weights_bytes': params * 4, 'allocated_layer_arrays_bytes': sum(a.nbytes for a in arrays),
            'process_memory_including_python_libraries_and_audio': process_memory(),
            'estimated_training_arrays_bytes': params * 16,
            'compressed_model_bytes': disk_bytes,
            'reference_ram_bytes_assumed': pi_ram_bytes,
            'weights_percent_of_assumed_ram': params * 4 / pi_ram_bytes * 100,
            'timing_input': f'synthetic_audio_{audio_features.WINDOW_SECONDS}_seconds_not_microphone_capture',
            'runs': 100, 'network_only_mean_ms': float(np.mean(forward_times)),
            'features_and_network_mean_ms': float(np.mean(full_times)),
            'features_and_network_p95_ms': float(np.percentile(full_times, 95)),
            'timing_excludes': 'microphone capture, window accumulation and device buffering'}


def inspect_storage_progress(data_dir=None):
    data_dir = Path(data_dir) if data_dir is not None else ROOT_DIR / 'data'
    report = inspect_dataset(data_dir, load_config())
    return {'registered_valid_files': report['files'], 'registered_valid_audio_bytes': report['dataset_bytes'],
            'audio_hours': report['audio_hours'], 'physical_storage_bytes_including_originals': report['storage_bytes'],
            'target_bytes_optional_local_goal': report['dataset_target_bytes'],
            'progress_percent': report['progress_percent'], 'bytes_remaining': report['bytes_remaining'],
            'validation_error_count': len(report['errors']), 'validation_errors_first_five': report['errors'][:5],
            'training_missing': report['training_missing']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--data-dir', type=Path, default=ROOT_DIR / 'data')
    parser.add_argument('--output', type=Path, default=Path('output/profile.json'))
    args = parser.parse_args()
    try:
        report = {'model': inspect_model_size(weights_path=args.weights),
                  'storage': inspect_storage_progress(args.data_dir)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
