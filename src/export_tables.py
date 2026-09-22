"""Export the final-result objects as CSV and LaTeX tables."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export_all(result):
    directory = ROOT / 'results/tables'
    directory.mkdir(exist_ok=True)
    def table(name, header, rows):
        with (directory / (name + '.csv')).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        def tex(value):
            text = str(value)
            for old, new in [('&', r'\&'), ('%', r'\%'), ('_', r'\_'), ('#', r'\#')]:
                text = text.replace(old, new)
            return text
        lines = [r'\begin{tabular}{' + 'l' * len(header) + '}', r'\hline']
        lines.append(' & '.join(map(tex, header)) + r' \\')
        lines.append(r'\hline')
        lines.extend(' & '.join(map(tex, row)) + r' \\' for row in rows)
        lines.extend([r'\hline', r'\end{tabular}'])
        (directory / (name + '.tex')).write_text('\n'.join(lines) + '\n')
    models = result['scoring']['corpus']
    table('corpus', ['Model', 'Condition', 'UAS (%)', 'LAS (%)', 'Scoring tokens'], [
        [model, condition, f"{100*row['uas']:.4f}", f"{100*row['las']:.4f}", row['total_tokens']]
        for model, conditions in models.items() for condition, row in conditions.items()])
    table('pairwise', ['Model', 'First', 'Second', 'Status', 'LAS diff (pp)', 'CI lower', 'CI upper', 'Raw p', 'Holm p (15)', 'Primary Holm p (4)'], [
        [model, row['first'], row['second'], row['status'], 100*row['difference'], 100*row['ci_95'][0], 100*row['ci_95'][1], row['p_raw'], row['p_holm_15'], row.get('p_holm_primary_4','')]
        for model, comparisons in result['pairwise']['models'].items() for row in comparisons.values()])
    for name in ['reliability','length','relations']:
        rows = []
        def flatten(value, path):
            if isinstance(value, dict):
                for key, subvalue in value.items(): flatten(subvalue, path+[key])
            elif isinstance(value, list):
                for key, subvalue in enumerate(value): flatten(subvalue, path+[str(key)])
            else: rows.append(['/'.join(path), value])
        flatten(result['scoring'][name], [])
        table(name, ['Metric path', 'Value'], rows)
    table('token_usage', ['Model','Records','With usage','Prompt tokens','Completion tokens','Total tokens'], [
        [model]+[row[key] for key in ['records','records_with_usage','prompt_tokens','completion_tokens','total_tokens']]
        for model,row in result['budget'].items()])
    table('critique_churn', ['Model','Metric','Value'], [
        [model,key,value] for model,row in result['critique_churn'].items() for key,value in row.items() if not isinstance(value,(dict,list))])
