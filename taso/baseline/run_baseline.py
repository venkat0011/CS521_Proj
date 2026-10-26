"""Tiny TASO smoke run using the all-rules policy. This does not identify internal bottlenecks."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1] / 'TASO')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ['TASO_HOME'] = str(args.repo.resolve())
    report = {'status': 'started', 'platform': platform.platform(), 'python': sys.version,
              'graph': 'Input[1,3,16,16] -> Conv[8,3,3,3] -> ReLU',
              'alpha': 1.05, 'budget_argument': 5,
              'notes': ['Upstream budget counts expansions and has an off-by-one condition.',
                        'This is a smoke run, not a speedup or bottleneck claim.',
                        'Native run_time uses backend timing, not numerical equivalence verification.']}
    try:
        import taso as ts
        if not hasattr(ts, 'new_graph'):
            raise RuntimeError('Python found a folder named taso, not the compiled TASO package. Build/install TASO first.')
        report['taso_module'] = ts.__file__
        try:
            report['git_commit'] = subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD'],text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            report['git_commit'] = 'unavailable (e.g. source archive)'
        start = time.perf_counter()
        graph = ts.new_graph()
        x = graph.new_input(dims=(1,3,16,16))
        w = graph.new_weight(dims=(8,3,3,3))
        y = graph.conv2d(input=x,weight=w,strides=(1,1),padding='SAME')
        graph.relu(y)
        report['graph_construction_seconds'] = time.perf_counter() - start
        report['before_operators'] = dict(Counter(graph.get_operator_type(op) for op in graph.get_operator_list()))
        report['before_estimated_ms'] = graph.cost()
        report['before_measured_ms'] = graph.run_time()
        start = time.perf_counter()
        optimized = ts.optimize(graph,alpha=1.05,budget=5,print_subst=True)
        report['optimize_wall_seconds'] = time.perf_counter() - start
        # Upstream optimize may delete the original graph: never access graph again.
        report['after_operators'] = dict(Counter(optimized.get_operator_type(op) for op in optimized.get_operator_list()))
        report['after_estimated_ms'] = optimized.cost()
        report['after_measured_ms'] = optimized.run_time()
        report['status'] = 'completed'
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    (args.out / 'baseline.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return 0 if report['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
