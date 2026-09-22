#!/usr/bin/env python3
"""Refresh experimental-input checksums after an intentional input change."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]


def input_files():
    paths = set()
    for pattern in ('data/UD_English-EWT/*.conllu', 'outputs/*/*/*.json.gz', 'prompts/*.txt'):
        paths.update(ROOT.glob(pattern))
    paths.update(ROOT / name for name in (
        'docs/input_identity.json', 'docs/expected_numeric_hashes.json'))
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Expected a regular input file: {path.relative_to(ROOT)}')
        yield path


def main():
    paths = list(input_files())
    folder = ROOT / 'checksums'
    folder.mkdir(exist_ok=True)
    content = ''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}\n' for p in paths)
    (folder / 'INPUT_SHA256SUMS.txt').write_text(content, encoding='utf-8')
    print(f'INPUT_SHA256SUMS.txt: {len(paths)} files')


if __name__ == '__main__':
    main()
