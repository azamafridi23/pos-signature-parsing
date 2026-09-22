"""Check compressed archive support and safe setup of optional new runs."""
import csv
import gzip
import json
import pytest
from src import analyze_results, extended_error_analysis
from scripts.init_tracker import init_tracker


@pytest.mark.parametrize('module', [analyze_results, extended_error_analysis])
@pytest.mark.parametrize('compressed', [False, True])
def test_diagnostic_loader_accepts_both_archive_formats(tmp_path, module, compressed):
    records = [{'sent_id': 'example', 'llm_parse': None}]
    requested = tmp_path / 'model_evaluation.json'
    payload = json.dumps(records).encode()
    if compressed:
        (tmp_path / 'model_evaluation.json.gz').write_bytes(gzip.compress(payload))
    else:
        requested.write_bytes(payload)
    assert module.load_experiment(str(requested)) == records


def test_tracker_matches_collector_columns_and_preserves_existing_work(tmp_path):
    source = tmp_path / 'input.conllu'
    source.write_text('# sent_id = a\n\n# sent_id = b\n')
    target = tmp_path / 'runs/tracker.csv'
    assert init_tracker(source, target) == 2
    rows = list(csv.DictReader(target.open()))
    assert [r['sent_id'] for r in rows] == ['a', 'b']
    assert all(r[f'experiment{i}_status'] == 'PENDING' for r in rows for i in range(1, 7))
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        init_tracker(source, target)
    assert target.read_bytes() == before


@pytest.mark.parametrize('content', ['', '# sent_id = a\n# sent_id = a\n'])
def test_tracker_rejects_invalid_identifiers_before_writing(tmp_path, content):
    source = tmp_path / 'input.conllu'
    source.write_text(content)
    target = tmp_path / 'tracker.csv'
    with pytest.raises(ValueError):
        init_tracker(source, target)
    assert not target.exists()
