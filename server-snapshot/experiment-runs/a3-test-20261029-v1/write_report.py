"""Summarize completed A3 results alongside identically recomputed A2 metrics."""
import csv
import json
from pathlib import Path


def main(folder):
    folder = Path(folder)
    summary = json.loads((folder / 'summary.json').read_text())
    protocol = json.loads((folder / 'protocol.json').read_text())
    baseline = summary['baseline']
    fields = ['id', 'a2_valid', 'a3_valid', 'a2_iou_raw', 'a3_iou_raw',
              'a2_iou_translation_aligned', 'a3_iou_translation_aligned', 'a3_bbox_max_relative', 'a3_error']
    with (folder / 'results.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fields); writer.writeheader()
        for ident in protocol['evaluated_ids']:
            a = json.loads((folder / 'baseline-first-pass' / (ident + '.json')).read_text())
            b = json.loads((folder / 'samples' / (ident + '.json')).read_text())
            row = {'id': ident, 'a2_valid': a['valid_solid'], 'a3_valid': b['valid_solid'],
                   'a3_bbox_max_relative': max(b.get('metrics', {}).get('bbox_relative', []), default=None),
                   'a3_error': b['attempts'][0].get('error', b.get('metric_error', ''))}
            for key in ['iou_raw', 'iou_translation_aligned']:
                row['a2_' + key] = a.get('metrics', {}).get(key)
                row['a3_' + key] = b.get('metrics', {}).get(key)
            writer.writerow(row)
    def fmt(v):
        return 'unavailable' if v is None else f'{v:.6f}'
    lines = ['# A3 fixed-test evaluation', '', '| Metric | A2 | A3 |', '|---|---:|---:|',
             f"| Valid solids | {baseline['valid_solid_count']}/44 | {summary['valid_solid_count']}/44 |"]
    for label, key in [('Raw IoU, all 44', 'mean_iou_raw_all'),
                       ('Translation-aligned IoU, all 44', 'mean_iou_translation_aligned_all'),
                       ('Raw IoU, valid outputs', 'mean_iou_raw_successful_comparable'),
                       ('Translation-aligned IoU, valid outputs', 'mean_iou_translation_aligned_successful_comparable')]:
        lines.append(f'| {label} | {fmt(baseline.get(key))} | {fmt(summary.get(key))} |')
    for tolerance in [1, 5]:
        key = f'bbox_xyz_within_{tolerance}pct_count'
        lines.append(f'| XYZ bounding dimensions within {tolerance}% | {baseline[key]} | {summary[key]} |')
    lines += ['', 'All-44 means assign zero to invalid CAD; unresolved metric errors leave the mean unavailable.',
              'Exact serial OCC Boolean IoU, also measured after bbox-center translation; no rotation or scale fitting.',
              '50 fixed test cases, 6 excluded for invalid ground truth, 44 evaluated. This previously diagnosed set is a regression test.',
              'A3 checkpoint1225. Same input and generation settings as A2: 16384 new tokens, 600 seconds, greedy, no repairs.',
              'GPU admission is stricter (<15% utilization, >42 GiB free); model weights are CPU-merged in memory and cast to BF16.',
              'All four projectors are checked against the saved A3 tensors on every model load.',
              'Dimension pass checks bounding extents, not every engineering dimension or feature topology.', '',
              'Newly valid relative to A2: ' + ', '.join(summary['newly_valid_ids']),
              'Regressed relative to A2: ' + ', '.join(summary['regressed_ids']),
              'Unresolved metric errors: ' + ', '.join(summary['metric_error_ids']),
              'Original source hashes unchanged: ' + str(summary['sources_unchanged'])]
    (folder / 'REPORT.md').write_text('\n'.join(lines) + '\n')
