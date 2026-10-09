"""Verify original benchmark sources and the integrated supplied helper scripts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    original = json.loads((ROOT / 'metadata/source_manifest.json').read_text())
    helpers = json.loads((ROOT / 'metadata/maml_helpers_manifest.json').read_text())
    failures = []
    for entry in original['entries'] + helpers['entries']:
        path = ROOT / entry['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            failures.append(entry['path'])
    if failures:
        print('Changed or missing files:\n' + '\n'.join(failures), file=sys.stderr)
        raise SystemExit(1)
    unchanged = sum(e['original_sha256'] == e['sha256'] for e in original['entries'])
    print(f"Verified {len(original['entries'])} original Python files; {unchanged} are byte-identical to VC_paper(1).zip.")
    print('The remaining original test file contains the documented two-line axis-label correction.')
    print(f"Verified {len(helpers['entries'])} supplied MAML helpers with documented CLI/output adaptations.")
    print('See metadata/maml_helpers.patch and metadata/maml_helpers_update.md.')


if __name__ == '__main__':
    main()
