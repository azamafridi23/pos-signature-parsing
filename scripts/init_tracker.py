#!/usr/bin/env python3
"""Create a fresh six-condition tracker for optional new inference runs."""
import argparse
import csv
from pathlib import Path


def init_tracker(input_path: Path, output_path: Path) -> int:
    sentence_ids = [line.split(' = ', 1)[1].strip()
                    for line in input_path.read_text(encoding='utf-8').splitlines()
                    if line.startswith('# sent_id = ')]
    if not sentence_ids or any(not sid for sid in sentence_ids) or len(set(sentence_ids)) != len(sentence_ids):
        raise ValueError('Input must contain nonempty, unique sentence identifiers.')
    fields = ['sent_id'] + [f'experiment{i}_status' for i in range(1, 7)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Never replace a tracker that may already contain completed work.
    with output_path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sid in sentence_ids:
            writer.writerow({'sent_id': sid, **{field: 'PENDING' for field in fields[1:]}})
    return len(sentence_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/UD_English-EWT/en_ewt-ud-test.conllu'))
    parser.add_argument('--output', type=Path, default=Path('runs/experiment_tracker_gpt_oss_120b.csv'))
    args = parser.parse_args()
    count = init_tracker(args.input, args.output)
    print(f'Created {count} tracker rows in {args.output}')


if __name__ == '__main__':
    main()
