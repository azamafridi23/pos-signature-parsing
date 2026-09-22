#!/usr/bin/env python3
"""Refresh release manifests after an intentional, verified publication edit."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.idea', '.vscode', 'runs', 'processed', 'build', 'dist', 'htmlcov'}
MANIFESTS = {'checksums/SHA256SUMS.txt', 'checksums/INPUT_SHA256SUMS.txt'}


def release_files():
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f'Symlink is not a release file: {relative}')
        if (not path.is_file() or relative.as_posix() in MANIFESTS
                or path.name in {'.DS_Store', '.coverage', '.env'}
                or (path.name.startswith('.env.') and path.name != '.env.example')
                or path.suffix in {'.pyc', '.pyo', '.tmp', '.bak'}):
            continue
        yield path


def main():
    files = list(release_files())
    inputs = [p for p in files if p.relative_to(ROOT).parts[0] in {'data', 'outputs', 'prompts'}
              or p.relative_to(ROOT).as_posix() in {'docs/input_identity.json', 'docs/expected_numeric_hashes.json'}]
    folder = ROOT / 'checksums'
    folder.mkdir(exist_ok=True)
    for name, paths in [('INPUT_SHA256SUMS.txt', inputs), ('SHA256SUMS.txt', files)]:
        content = ''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}\n' for p in paths)
        (folder / name).write_text(content, encoding='utf-8')
        print(f'{name}: {len(paths)} files')


if __name__ == '__main__':
    main()
