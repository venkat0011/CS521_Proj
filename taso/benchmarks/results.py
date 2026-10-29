"""Export every paired trial and describe its distribution without hiding failures."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


def quantile(sorted_values, p):
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def distribution(samples):
    """Keep trial IDs and values in trial order; quartiles use linear interpolation."""
    samples = [{'trial': trial, 'value': float(value)} for trial, value in samples
               if isinstance(value, (int, float)) and math.isfinite(value)]
    values = [item['value'] for item in samples]
    ordered = sorted(values)
    q1, q3 = quantile(ordered, .25), quantile(ordered, .75)
    return {'n': len(values), 'samples': samples, 'sorted_values': ordered,
            'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'sample_stddev': statistics.stdev(values) if len(values) > 1 else None,
            'min': min(values) if values else None, 'max': max(values) if values else None,
            'q1': q1, 'q3': q3, 'iqr': q3 - q1 if values else None}


def flatten(report):
    row = {k: report.get(k) for k in ('network', 'model', 'trial', 'status', 'error', 'budget', 'alpha', 'model_calls')}
    inference = report.get('inference') or {}
    row['inference_seconds'] = inference.get('inference_seconds')
    row['provider'] = inference.get('provider')
    row['returned_model'] = inference.get('returned_model')
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cost'):
        row['api_' + key] = (inference.get('usage') or {}).get(key)
    worker = report.get('evaluation') or {}
    row['warmup_search_seconds'] = worker.get('warmup_search_seconds')
    row['evaluation_order'] = ','.join(worker.get('evaluation_order', []))
    for name, values in worker.get('variants', {}).items():
        for key in ('estimated_ms', 'measured_ms', 'search_seconds', 'expansions', 'stop_reason', 'queue_remaining'):
            row[name + '_' + key] = values.get(key)
        for key, value in values.get('counters', {}).items():
            row[name + '_' + key] = value
    if report.get('status') == 'completed':
        for metric in ('estimated_ms', 'measured_ms', 'search_seconds', 'expansions', 'admitted', 'generated'):
            baseline = row.get('baseline_' + metric)
            llm = row.get('llm_' + metric)
            row['llm_over_baseline_' + metric] = llm / baseline if baseline and llm is not None else None
        seconds = row.get('inference_seconds')
        if seconds is not None:
            row['llm_inference_plus_search_seconds'] = seconds + row['llm_search_seconds']
    return row


def write_exports(directory, reports, plots=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = [flatten(report) for report in reports]
    columns = sorted({key for row in rows for key in row})
    with (directory / 'trials.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (directory / 'trials.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in reports))
    groups = defaultdict(list)
    for row in rows:
        groups[(row['network'], row['model'])].append(row)
    summaries = []
    for (network, model), group in groups.items():
        metrics = {}
        for key in columns:
            if key in ('trial', 'budget', 'alpha'):
                continue
            samples = [(r['trial'], r[key]) for r in group
                       if isinstance(r.get(key), (int, float)) and not isinstance(r[key], bool)]
            if samples:
                metrics[key] = distribution(samples)
        paired = [r for r in group if r['status'] == 'completed']
        ratios = [r.get('llm_over_baseline_estimated_ms') for r in paired]
        ratios = [v for v in ratios if v is not None]
        summaries.append({'network': network, 'model': model, 'planned_trials': len(group),
                          'completed': len(paired),
                          'failed': sum(r['status'].startswith('failed') for r in group),
                          'statuses': {r['status']: sum(x['status'] == r['status'] for x in group) for r in group},
                          'estimated_cost_wins': sum(v < 1 - 1e-6 for v in ratios),
                          'estimated_cost_ties': sum(abs(v - 1) <= 1e-6 for v in ratios),
                          'estimated_cost_losses': sum(v > 1 + 1e-6 for v in ratios),
                          'metrics': metrics})
    stats = {'schema_version': 1, 'quartile_method': 'linear interpolation at (n-1)*p',
             'stddev_method': 'sample standard deviation (n-1); null when n<2',
             'notes': ['All planned trial slots appear in trials.csv and trials.jsonl, including failures.',
                       'Ratios are paired LLM/baseline; below 1 is lower. Budgets are upper bounds.',
                       'Five trials describe observed variation; they do not establish significance.',
                       'Each trial requests a new one-shot strategy. Trial variation includes GPU cost-model noise.',
                       'Inference plus search excludes setup, graph construction and experimental warm-up.'],
             'groups': summaries}
    (directory / 'summary.json').write_text(json.dumps(stats, indent=2, allow_nan=False) + '\n')
    lines = ['# Trial distributions', '', 'Each value below is a separate trial, in trial order.', '',
             '| Network | LLM | Complete/planned | Estimated cost LLM/baseline | Median | Sample SD |',
             '|---|---|---:|---|---:|---:|']
    for group in summaries:
        metric = group['metrics'].get('llm_over_baseline_estimated_ms', {})
        values = ', '.join(f"t{s['trial']}={s['value']:.4f}" for s in metric.get('samples', [])) or '—'
        def fmt(value): return f'{value:.4f}' if value is not None else '—'
        lines.append(f"| {group['network']} | {group['model']} | {group['completed']}/{group['planned_trials']} | "
                     f"{values} | {fmt(metric.get('median'))} | {fmt(metric.get('sample_stddev'))} |")
    lines.extend(['', 'Below 1 means lower cost. Missing/failed trials are retained in the raw exports.', '',
                  'summary.json contains the complete distributions for every numeric metric: trial values,',
                  'mean, median, sample SD, min, max, Q1, Q3 and IQR.', ''])
    (directory / 'summary.md').write_text('\n'.join(lines))
    if plots and any(group['completed'] for group in summaries):
        plot_distributions(directory, summaries)
    return stats


def plot_distributions(directory, groups):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        (directory / 'plots_unavailable.txt').write_text('Install taso/benchmarks/requirements.txt, then rerun --summarize.\n')
        return
    (directory / 'plots_unavailable.txt').unlink(missing_ok=True)
    metrics = [('llm_over_baseline_estimated_ms', 'Estimated cost (LLM / baseline)'),
               ('llm_over_baseline_measured_ms', 'GPU latency (LLM / baseline)'),
               ('llm_over_baseline_search_seconds', 'Warmed search time (LLM / baseline)'),
               ('llm_over_baseline_expansions', 'Expanded graphs (LLM / baseline)')]
    fig, axes = plt.subplots(2, 2, figsize=(max(11, len(groups) * 1.8), 10))
    for axis, (metric, title) in zip(axes.flat, metrics):
        labels = []
        for index, group in enumerate(groups, 1):
            values = [s['value'] for s in group['metrics'].get(metric, {}).get('samples', [])]
            labels.append(f"{group['network']}\n{group['model']}\nn={len(values)}/{group['planned_trials']}")
            if values:
                axis.boxplot([values], positions=[index], widths=.45, showfliers=False,
                             medianprops={'color': 'black'})
                offsets = [(j - (len(values)-1)/2) * .055 for j in range(len(values))]
                axis.scatter([index + offset for offset in offsets], values,
                             color='#276fbf', s=32, zorder=3)
        axis.axhline(1, color='#ae3434', linestyle='--', linewidth=1)
        axis.set_title(title)
        axis.set_xticks(range(1, len(labels) + 1), labels, fontsize=8, rotation=20)
        axis.set_xlim(.4, len(labels) + .6)
        axis.set_ylim(bottom=0)
        axis.grid(axis='y', alpha=.2)
    fig.suptitle('Every completed trial is a dot; paired ratios below 1 are lower')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(directory / f'distributions.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)
