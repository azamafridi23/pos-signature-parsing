#!/usr/bin/env python3
"""Verify package integrity, final raw/evaluation agreement, and numeric outputs."""
import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
import reproduce_paper_results as analysis


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def gold_reference():
    records = {}
    sent_id, tokens = None, []
    text = (ROOT / 'data/UD_English-EWT/en_ewt-ud-test.conllu').read_text()
    for line in (text + '\n').splitlines():
        if not line:
            if sent_id is not None:
                assert sent_id not in records
                records[sent_id] = tokens
            sent_id, tokens = None, []
        elif line.startswith('# sent_id = '):
            sent_id = line.split(' = ',1)[1]
        elif not line.startswith('#'):
            fields = line.split('\t')
            if fields[0].isdigit():
                tokens.append({'id':int(fields[0]), 'form':fields[1], 'upos':fields[3], 'head':int(fields[6]), 'deprel':fields[7]})
    assert len(records) == 2077
    assert sum(t['upos']!='PUNCT' for tokens in records.values() for t in tokens)==21998
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--inputs-only', action='store_true')
    mode.add_argument('--results-only', action='store_true')
    args=parser.parse_args()
    if not args.results_only:
        manifest = ROOT / 'checksums' / ('INPUT_SHA256SUMS.txt' if args.inputs_only else 'SHA256SUMS.txt')
        for line in manifest.read_text().splitlines():
            expected, relative = line.split('  ',1)
            path = ROOT / relative
            assert path.is_file() and not path.is_symlink(), relative
            assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, relative
        identities=json.loads((ROOT/'docs/input_identity.json').read_text())
        for relative,identity in identities.items():
            data=gzip.decompress((ROOT/relative).read_bytes())
            assert hashlib.sha256(data).hexdigest()==identity['uncompressed_sha256'],relative
        print('File checksums and all 24 archive identities verified.')
    if args.inputs_only:
        return
    reference = gold_reference()
    checked = 0
    for model in analysis.MODELS:
        for condition,_ in analysis.CONDITIONS:
            base=ROOT/f'outputs/{model}/{condition}'
            raw=analysis.load_json(base/'model_responses.json.gz')
            evaluated=analysis.load_json(base/'model_evaluation.json.gz')
            raw_map={r['sent_id']:r for r in raw}; eval_map={r['sent_id']:r for r in evaluated}
            assert len(raw_map)==len(raw) and len(eval_map)==len(evaluated)
            assert set(raw_map)==set(eval_map) and set(raw_map)<=set(reference)
            for sid,row in raw_map.items():
                gold=reference[sid]
                for archived in [row['gold'],eval_map[sid]['gold']]:
                    assert [{k:t[k] for k in ['id','form','upos','head','deprel']} for t in archived]==gold,(model,condition,sid)
                predicted=analysis.parse_raw_response(row.get('raw_response'))
                stored=eval_map[sid].get('llm_parse')
                assert (predicted is None)==(stored is None),(model,condition,sid,'schema')
                assert analysis.score_tokens(gold,predicted)[:3]==analysis.score_tokens(gold,stored)[:3],(model,condition,sid,'scores')
                checked+=1
    assert checked==24921
    print('All 24,921 raw responses agree with evaluations and CoNLL-U gold.')
    result=json.loads((ROOT/'results/paper_results.json').read_text())
    expected=json.loads((ROOT/'docs/expected_numeric_hashes.json').read_text())
    actual={key:digest(result[key]) for key in ['dataset','pairwise','critique_churn','qualitative_selection']}
    actual['scoring_without_protocol']=digest({k:v for k,v in result['scoring'].items() if k!='protocol'})
    for name in ['qualitative_examples','per_relation','length_analysis']:
        actual[name]=digest(json.loads((ROOT/f'results/{name}.json').read_text()))
    assert actual==expected,'Numerical output differs from reference.'
    assert result['budget']['total']['total_tokens']==66003325
    print('All numerical reference checks passed; final-record usage is 66,003,325 tokens.')


if __name__=='__main__':
    main()
