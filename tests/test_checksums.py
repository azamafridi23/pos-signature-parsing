"""Protect scientific inputs without freezing routine repository edits."""
import pytest
from scripts import update_checksums, verify


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(update_checksums, 'ROOT', tmp_path)
    monkeypatch.setattr(verify, 'ROOT', tmp_path)
    for name in ('data/UD_English-EWT/test.conllu', 'outputs/model/condition/model_responses.json.gz',
                 'prompts/task.txt', 'docs/input_identity.json', 'docs/expected_numeric_hashes.json'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original')
    update_checksums.main()
    return tmp_path


def test_routine_edits_do_not_change_input_manifest(inputs):
    manifest = inputs / 'checksums/INPUT_SHA256SUMS.txt'
    before = manifest.read_bytes()
    for name in ('README.md', 'src/example.py', 'pyproject.toml', 'figures/example.png', 'data/README.md'):
        path = inputs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('edited')
    verify.verify_input_checksums()
    update_checksums.main()
    assert manifest.read_bytes() == before
    assert not (inputs / 'checksums/SHA256SUMS.txt').exists()
    assert (inputs / 'docs/expected_numeric_hashes.json').read_text() == 'original'


def test_changed_input_requires_review(inputs):
    (inputs / 'prompts/task.txt').write_text('changed')
    with pytest.raises(AssertionError, match='prompts/task.txt'):
        verify.verify_input_checksums()


def test_missing_input_fails_verification(inputs):
    (inputs / 'data/UD_English-EWT/test.conllu').unlink()
    with pytest.raises(AssertionError, match='test.conllu'):
        verify.verify_input_checksums()
