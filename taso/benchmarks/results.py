"""Export every paired trial and describe its distribution without hiding failures."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


def feedback_mode(report):
    worker = report.get('evaluation') or {}
    return (report.get('feedback_mode') or worker.get('feedback_mode') or
            ('checkpoint_gpu' if report.get('gpu_feedback', worker.get('gpu_feedback', False)) else
             'checkpoint' if report.get('search_mode', worker.get('search_mode')) == 'checkpoint' else 'one_shot'))


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
    row = {k: report.get(k) for k in ('network', 'model', 'trial', 'status', 'error', 'budget', 'alpha', 'model_calls', 'search_mode', 'checkpoint_interval')}
    inference = report.get('inference') or {}
    row['feedback_mode'] = feedback_mode(report)
    row['gpu_feedback'] = row['feedback_mode'] == 'checkpoint_gpu'
    row['reused_result'] = report.get('reused_result', False)
    row['reuse_source_trial'] = report.get('reuse_source_trial')
    row['model_calls_this_run'] = 0 if row['reused_result'] else report.get('model_calls', 0)
    row['inference_seconds'] = inference.get('inference_seconds')
    row['prompt_version'] = inference.get('prompt_version')
    row['prompt_bytes_total'] = inference.get('prompt_bytes_total')
    row['max_tokens'] = inference.get('max_tokens')
    row['output_token_limit_mode'] = inference.get('output_token_limit_mode')
    row['api_visible_completion_tokens'] = inference.get('visible_completion_tokens')
    reasons = inference.get('finish_reason_counts')
    if reasons is None:
        reason = inference.get('finish_reason')
        reasons = {reason: 1} if reason else {}
    row['api_finish_reason_counts'] = json.dumps(reasons, sort_keys=True)
    row['api_length_completions'] = reasons.get('length', 0) if reasons else None
    for key in ('completion_policy', 'max_retries', 'checkpoints_reached',
                'continuation_requests', 'provider_retries', 'strategy_retries', 'provider_routing', 'retry_wait_seconds',
                'usage_observations', 'usage_missing_attempts'):
        row[key] = inference.get(key)
    decisions = (report.get('evaluation') or {}).get('checkpoint_decisions', report.get('checkpoint_decisions', []))
    row['recovered_checkpoints'] = sum(d.get('status') == 'completed' and len(d.get('attempts', [])) > 1
                                        for d in decisions)
    row['first_attempt_checkpoint_successes'] = sum(d.get('status') == 'completed' and len(d.get('attempts', [])) == 1
                                                    for d in decisions)
    row['failed_checkpoint_decisions'] = sum(d.get('status') == 'failed' for d in decisions)
    for key in ('requested_temperature', 'temperature', 'requested_reasoning_effort',
                'reasoning_effort', 'reasoning_effort_override_applied', 'capabilities_checked_on'):
        row[key] = inference.get(key)
    row['omitted_parameters'] = json.dumps(inference.get('omitted_parameters', {}), sort_keys=True)
    row['api_reasoning_tokens'] = ((inference.get('usage') or {}).get(
        'completion_tokens_details') or {}).get('reasoning_tokens')
    row['provider'] = inference.get('provider')
    row['returned_model'] = inference.get('returned_model')
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cost'):
        row['api_' + key] = (inference.get('usage') or {}).get(key)
    row['api_cost_this_run'] = 0 if row['reused_result'] else row['api_cost']
    worker = report.get('evaluation') or {}
    row['gpu'] = worker.get('environment', {}).get('gpu')
    row['evaluation_python'] = worker.get('environment', {}).get('python')
    for key in ('gpu_feedback_seconds', 'gpu_feedback_samples'):
        row[key] = worker.get(key, 0)
    row['warmup_search_seconds'] = worker.get('warmup_search_seconds')
    row['evaluation_order'] = ','.join(worker.get('evaluation_order', []))
    for name, values in worker.get('variants', {}).items():
        for key in ('estimated_ms', 'measured_ms', 'search_seconds', 'expansions', 'stop_reason', 'queue_remaining',
                    'optimization_wall_seconds', 'checkpoint_callback_seconds'):
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
            if row.get('retry_wait_seconds') is not None:
                row['llm_inference_search_and_retry_wait_seconds'] = (
                    seconds + row['llm_search_seconds'] + row['retry_wait_seconds'])
    return row


def model_call_rows(reports):
    """One row per observed request, including failed checkpoint completions."""
    rows = []
    for report in reports:
        evaluation = report.get('evaluation') or {}
        decisions = evaluation.get('checkpoint_decisions', report.get('checkpoint_decisions', []))
        is_checkpoint = report.get('search_mode', evaluation.get('search_mode')) == 'checkpoint'
        calls = decisions if is_checkpoint else ([report] if report.get('model_calls') else [])
        expanded = [(call, attempt) for call in calls for attempt in (call.get('attempts') or [call])]
        for checkpoint, call in expanded:
            if not call.get('model_calls'):
                continue
            info = call.get('inference') or {}
            usage = info.get('usage') or {}
            row = {k: report.get(k) for k in ('network', 'model', 'trial')}
            row.update(feedback_mode=feedback_mode(report), checkpoint_interval=report.get('checkpoint_interval'),
                       reused_result=report.get('reused_result', False), reuse_source_trial=report.get('reuse_source_trial'),
                       model_calls_this_run=0 if report.get('reused_result', False) else 1)
            row.update(trial_status=report.get('status'), status=call.get('status'),
                       expansion=checkpoint.get('expansions'), checkpoint_status=checkpoint.get('status'),
                       attempt=call.get('attempt', 1), action=call.get('action', 'initial'),
                       replay_mode=call.get('replay_mode'), estimated_input_tokens=call.get('estimated_input_tokens'),
                       retry_reason=call.get('retry_reason'), failure_kind=call.get('failure_kind'),
                       error=call.get('error'),
                       prompt_bytes=call.get('prompt_bytes', checkpoint.get('prompt_compaction', {}).get('messages_utf8_bytes')))
            for key in ('request_id', 'provider', 'returned_model', 'inference_seconds', 'max_tokens',
                        'output_token_limit_mode', 'finish_reason', 'native_finish_reason',
                        'visible_completion_tokens', 'reasoning_effort', 'temperature'):
                row[key] = info.get(key)
            if call.get('max_tokens') is not None:
                row['max_tokens'] = call['max_tokens']
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cost'):
                row[key] = usage.get(key)
            row['cost_this_run'] = 0 if report.get('reused_result', False) else usage.get('cost')
            row['reasoning_tokens'] = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
            rows.append(row)
    return rows


def write_exports(directory, reports, plots=True, min_successful_trials=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    reports = list(reports)
    calls = model_call_rows(reports)
    call_columns = sorted({key for row in calls for key in row}) or ['network', 'model', 'trial', 'status']
    with (directory / 'model_calls.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=call_columns)
        writer.writeheader()
        writer.writerows(calls)
    rows = [flatten(report) for report in reports]
    gpu_rows = []
    for report in reports:
        worker = report.get('evaluation') or {}
        for obs in worker.get('gpu_feedback_observations', []):
            gpu_rows.append({**{k: report.get(k) for k in ('network', 'model', 'trial', 'checkpoint_interval')},
                'feedback_mode': feedback_mode(report), 'reused_result': report.get('reused_result', False),
                **{k: v for k, v in obs.items() if k != 'samples_ms'},
                'samples_ms': json.dumps(obs['samples_ms'])})
    with (directory / 'gpu_feedback.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=sorted({k for row in gpu_rows for k in row}) or ['network','model','trial','median_ms'])
        writer.writeheader()
        writer.writerows(gpu_rows)
    columns = sorted({key for row in rows for key in row})
    with (directory / 'trials.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (directory / 'trial.csv').write_bytes((directory / 'trials.csv').read_bytes())
    (directory / 'trials.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in reports))
    groups = defaultdict(list)
    for row in rows:
        groups[(row['network'], row['model'], row['feedback_mode'], row['checkpoint_interval'])].append(row)
    summaries = []
    for (network, model, mode, interval), group in groups.items():
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
        summaries.append({'network': network, 'model': model, 'feedback_mode': mode,
                          'checkpoint_interval': interval, 'planned_trials': len(group),
                          'completed': len(paired),
                          'minimum_successful_trials': min_successful_trials,
                          'meets_minimum': len(paired) >= min_successful_trials if min_successful_trials is not None else None,
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
                       'Trial samples describe observed variation; they do not establish statistical significance.',
                       'Each trial obtains independent policies. Checkpoint mode can make multiple attempts per decision; all returned usage is included.',
                       'model_calls.csv records per-request usage and finish reasons, including failed checkpoint calls.',
                       'A provider_default output limit omits our cap; provider defaults and model limits still apply.',
                       'Conditions and checkpoint intervals are grouped separately; reused results retain historical usage and have zero new calls/cost.',
                       'Inference plus search excludes setup, graph construction and experimental warm-up.'],
             'groups': summaries}
    (directory / 'summary.json').write_text(json.dumps(stats, indent=2, allow_nan=False) + '\n')
    lines = ['# Trial distributions', '', 'Each value below is a separate trial, in trial order.', '',
             '| Network | LLM | Feedback | Interval | Complete/planned | Final GPU latency LLM/baseline | Median | Sample SD |',
             '|---|---|---|---:|---:|---|---:|---:|']
    for group in summaries:
        metric = group['metrics'].get('llm_over_baseline_measured_ms', {})
        values = ', '.join(f"t{s['trial']}={s['value']:.4f}" for s in metric.get('samples', [])) or '—'
        def fmt(value): return f'{value:.4f}' if value is not None else '—'
        lines.append(f"| {group['network']} | {group['model']} | {group['feedback_mode']} | {group['checkpoint_interval']} | {group['completed']}/{group['planned_trials']} | "
                     f"{values} | {fmt(metric.get('median'))} | {fmt(metric.get('sample_stddev'))} |")
    lines.extend(['', 'Below 1 means lower cost. Missing/failed trials are retained in the raw exports.', '',
                  'summary.json contains the complete distributions for every numeric metric: trial values,',
                  'mean, median, sample SD, min, max, Q1, Q3 and IQR.', ''])
    if calls:
        lines.extend(['', '## LLM token usage', '',
                      'Includes returned usage from successful and failed requests, including historical reused results. model_calls_this_run/api_cost_this_run identify new work; missing usage stays blank.',
                      '', '| Network | LLM | Feedback | Interval | Requests observed | Length finishes | Prompt tokens | Output tokens | Reasoning tokens | Cost (USD) |',
                      '|---|---|---|---:|---:|---:|---:|---:|---:|---:|'])
        for network, model, mode, interval in groups:
            selected = [r for r in calls if r['network'] == network and r['model'] == model and r['feedback_mode'] == mode and r['checkpoint_interval'] == interval]
            def total(key):
                values = [r[key] for r in selected if isinstance(r.get(key), (int, float))]
                return sum(values) if values else None
            def show(value): return str(value) if value is not None else '—'
            costs = total('cost')
            count = sum(r['finish_reason'] == 'length' for r in selected)
            cost_text = f'{costs:.6f}' if costs is not None else '—'
            lines.append(f'| {network} | {model} | {mode} | {interval} | {len(selected)} | {count} | '
                         f'{show(total("prompt_tokens"))} | {show(total("completion_tokens"))} | '
                         f'{show(total("reasoning_tokens"))} | {cost_text} |')
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
            labels.append(f"{group['network']}\n{group['model']}\n{group.get('feedback_mode','')} I={group.get('checkpoint_interval')}\nn={len(values)}/{group['planned_trials']}")
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
