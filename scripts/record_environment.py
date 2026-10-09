"""Record the actual local environment separately from the reference environment."""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def environment() -> dict:
    packages = {}
    for name in ('jax', 'jaxlib', 'numpy', 'matplotlib', 'scikit-learn'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    data = {'recorded_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'python': platform.python_version(), 'python_executable': sys.executable,
            'system': platform.system(), 'release': platform.release(),
            'machine': platform.machine(), 'processor': platform.processor(),
            'cpu_count': os.cpu_count(), 'packages': packages,
            'environment_variables': {key: os.environ.get(key) for key in
                ('JAX_PLATFORMS', 'JAX_ENABLE_X64', 'OMP_NUM_THREADS',
                 'OPENBLAS_NUM_THREADS', 'XLA_FLAGS', 'MPLBACKEND')}}
    if platform.system() == 'Darwin':
        data['macos_version'] = platform.mac_ver()[0]
        for key in ('machdep.cpu.brand_string', 'hw.memsize'):
            result = subprocess.run(['sysctl', '-n', key], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                data[key] = result.stdout.strip()
    # No JAX import: precision is selected separately in each experiment process.
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-reference', action='store_true')
    args = parser.parse_args()
    data = environment()
    if args.check_reference:
        ref = json.loads((ROOT / 'environment.json').read_text())
        differences = []
        if data['python'] != ref['python']:
            differences.append(f"Python: local={data['python']}, reference={ref['python']}")
        for name, version in ref['packages'].items():
            if data['packages'][name] != version:
                differences.append(f"{name}: local={data['packages'][name]}, reference={version}")
        if data['system'] != 'Darwin' or data['machine'] != 'arm64':
            differences.append('The reference machine is an arm64 Mac.')
        data['reference_differences'] = differences
    text = json.dumps(data, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            handle.write(text)
        print(f'Actual environment written to {args.output}')
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
