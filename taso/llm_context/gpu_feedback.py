"""Measured latency of the estimated-best live graph; no GPU observations when disabled."""
import math
import statistics
import time

GPU_FEEDBACK_VERSION = 'best_graph_gpu_v1'


def measure_best_graph(graph, expansions, history, repeats=3):
    method = getattr(graph, 'run_time_for_feedback', None)
    if not callable(method):
        raise RuntimeError('Rebuild TASO: run_time_for_feedback snapshot binding is missing')
    if type(repeats) is not int or not 1 <= repeats <= 10:
        raise ValueError('gpu_feedback_repeats must be between 1 and 10')
    started = time.perf_counter()
    values = [float(method()) for _ in range(repeats)]
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise RuntimeError('GPU feedback returned a nonpositive or nonfinite latency')
    value = statistics.median(values)
    previous = history[-1]['median_ms'] if history else None
    initial = history[0]['median_ms'] if history else value
    return {'version': GPU_FEEDBACK_VERSION, 'expansions': expansions,
        'target': 'current best by TASO estimated cost; not GPU-selected or the full frontier',
        'samples_ms': values, 'median_ms': value, 'initial_ms': initial,
        'previous_ms': previous,
        'improvement_since_previous_ms': previous - value if previous is not None else None,
        'speedup_vs_initial': initial / value,
        'measurement_wall_seconds': time.perf_counter() - started,
        'method': 'snapshot inference only; constant-only preparation excluded; each sample uses upstream 100 warm-up and 100 timed executions',
        'scope': 'Latency observation only; no numerical equivalence check or queue reordering'}
