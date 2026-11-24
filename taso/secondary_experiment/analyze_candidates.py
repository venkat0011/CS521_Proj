"""Exact candidate-overlap, checkpoint lineage, numeric validity, and GPU evidence."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import statistics
from candidate_identity import Journal, equivalent


def unique(journal, ids):
    buckets=defaultdict(list);unknown=0
    for gid in sorted(ids):
        graph=journal.get(gid)
        if not graph['complete']:unknown+=1;continue
        duplicate=False;uncertain=False
        for previous in buckets[graph['token']]:
            result=equivalent(graph,journal.get(previous))
            if result is True:duplicate=True;break
            uncertain |= result is None
        if not duplicate:
            if uncertain:unknown+=1
            else:buckets[graph['token']].append(gid)
    return dict(established_classes=sum(map(len,buckets.values())),unresolved_records=unknown)


def compare_trial(kimi_dir, reference_dir, cfg):
    kr=json.loads((kimi_dir/'worker_result.json').read_text());br=json.loads((reference_dir/'worker_result.json').read_text())
    print(f'Indexing reference journal: {reference_dir.name}',flush=True)
    baseline=Journal(reference_dir/br['journal'])
    print(f'Indexing guided journal: {kimi_dir.name}',flush=True)
    guided=Journal(kimi_dir/kr['journal'])
    for journal,worker_report in [(baseline,br),(guided,kr)]:
        expected=worker_report['search']
        for event,counter in [('generated','generated'),('admitted','admitted')]:
            if sum(e['event']==event for e in journal.events)!=expected['counters'][counter]:
                raise ValueError('Candidate journal counters do not reconcile with native search trace')
        if sum(e['event']=='expanded' for e in journal.events)!=expected['expansions']:
            raise ValueError('Candidate expansion journal is incomplete')
    result=dict(network=kr.get('network','squeezenet'),kimi_trial=str(kimi_dir),reference_trial=str(reference_dir),
                comparison_scope='Independent searches; exact identity can be compared, causal effects of checkpoints are not isolated',
                counts={},overlap={},checkpoints=[],numerically_valid=kr.get('numerically_valid',False),
                model_calls=kr.get('model_calls'),winner_lineage=[],gpu_evidence={})
    for label,event in [('generated','generated'),('admitted','admitted'),('expanded','expanded')]:
        ids=guided.ids(event)
        result['counts'][label]=dict(records=len(ids),unique_graphs=unique(guided,ids))
        result['overlap'][label]={}
        for budget in cfg['reference_points']:
            allowed=baseline.ids(event,budget)
            counts=Counter(baseline.find(guided.get(gid),allowed)['status'] for gid in sorted(ids))
            result['overlap'][label][str(budget)]=dict(counts)
    checkpoints=[e for e in guided.events if e['event']=='checkpoint']+[guided.last('returned_search')]
    trace=json.loads((kimi_dir/'search_trace.json').read_text())
    catalog=json.loads((kimi_dir/'prepared/static_context.json').read_text())['catalog']
    rule_ids={r['taso_xfer_index']:r['rule_id'] for r in catalog['rules']} if isinstance(catalog,dict) and 'rules' in catalog else {}
    previous=None;previous_rules=None
    for e in checkpoints:
        graph=guided.get(e['graph'])
        path=[dict(x,generation_expansion_number=x['expansion']+1,rule_id=rule_ids.get(x['rule'])) for x in guided.lineage(e['graph'])]
        entry=dict(expansion=e['expansion'],native_graph=e['graph'],
                   same_graph_as_previous=None if previous is None else equivalent(previous,graph),
                   estimated_ms=guided.graphs[e['graph']]['estimated_ms'],
                   lineage=path,
                   baseline_occurrence={str(b):baseline.find(graph,baseline.ids('generated',b)|baseline.ids('initial'))
                                        for b in cfg['reference_points']})
        if e['event']=='checkpoint':
            active={r['rule_index'] for r in trace['phases'][e['phase']]['rules'] if r['attempts']>0}
            entry.update(enabled_native_rules=sorted(active),
                         added_rules=None if previous_rules is None else sorted(active-previous_rules),
                         removed_rules=None if previous_rules is None else sorted(previous_rules-active))
            previous_rules=active
        if path:
            entry['winner_generated_before_last_search_block']=path[-1]['expansion']<max(0,e['expansion']-25)
            entry['winning_graph_generated_at_expansion']=path[-1]['expansion']
            entry['winning_graph_generated_in_phase']=path[-1]['phase']
        result['checkpoints'].append(entry);previous=graph
    winner=guided.last('returned_search')['graph'];result['winner_lineage']=guided.lineage(winner)
    result['baseline_final_winner_in_guided_generation']=guided.find(
        baseline.get(baseline.last('returned_search')['graph']),guided.ids('generated')|guided.ids('initial'))
    # Compare both mappings' medians of fixed graphs measured in this Kimi worker.
    # This is a descriptive robustness check, not a significance test or a proof of causation.
    reference=kr.get('restored_reference_timing')
    final=next((o for o in reversed(kr['observations']) if o['role']=='final_execution'),None)
    if reference and final:
        r=[x['gpu']['median_ms'] for x in reference['timing']]
        g=[x['gpu']['median_ms'] for x in final['timing']]
        valid=(final['numerically_valid'] and kr.get('reference_restoration',{}).get('exact_graph_restoration',False)
               and kr.get('restored_reference_timing',{}).get('numerically_valid',False))
        result['gpu_evidence']=dict(valid_for_comparison=valid,
            reference_backend_plan_preserved=kr['reference_restoration'].get('exact_backend_plan_restoration'),reference_medians_ms=r,guided_medians_ms=g,
            median_ratio=statistics.median(g)/statistics.median(r),
            every_guided_mapping_median_below_every_reference_mapping_median=valid and max(g)<min(r),
            every_guided_mapping_median_above_every_reference_mapping_median=valid and min(g)>max(r),
            reference_timing_concerns=reference['timing_concerns'],guided_timing_concerns=final['timing_concerns'],
            limitation='Fixed baseline-first timing, variable hardware state and only two mapping medians; not a statistical or causal proof of speedup')
    result['reference_budget_quality']=[dict(expansion=o['expansion'],role=o['role'],
        numerically_valid=o['numerically_valid'],estimated_search_ms=o['estimated_search_ms'],gpu_medians_ms=[t['gpu']['median_ms'] for t in o['timing']],
        timing_concerns=o['timing_concerns']) for o in br['observations'] if o['expansion'] in cfg['reference_points']]
    return result


def analyze(root):
    cfg=json.loads((root/'config.json').read_text());summary=json.loads((root/'summary.json').read_text())
    report=dict(scope='Best found under specified budgets, not global optimality',trials=[],
                limitations=['Exact structural identity is distinct from mathematical equivalence.',
                    'Numerical agreement is tested on three documented inputs, not universally proven.',
                    'Generated/admitted/expanded overlap counts refer to recorded candidates; unique counts are reported separately.',
                    'Audit logging can affect profiling conditions; search time includes instrumentation.',
                    'GPU variability established in the earlier smoke test remains a competing explanation.'])
    for item in summary['variants'].get('kimi',{}).get('attempts',[]):
        if item['status']!='completed':continue
        directory=Path(item['directory']);record=json.loads((directory/'worker_result.json').read_text())
        reference=Path(record['reference_restoration']['source']).parents[2]
        trial=compare_trial(directory,reference,cfg);report['trials'].append(trial)
        (root/'candidate_analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['Candidate study: exact graph comparisons, numerical checks, and larger-budget references.',
           'Larger-budget results are best-found references, not proofs of global optimality.']
    for i,t in enumerate(report['trials'],1):
        lines.append(f'\n{t["network"]} trial {i}: numerical checks passed={t["numerically_valid"]}')
        for cp in t['checkpoints']:
            lines.append(f'  Expansion {cp["expansion"]}: same graph as previous={cp["same_graph_as_previous"]}; estimated cost={cp["estimated_ms"]:.6f} ms')
        lines.append('  Generated-candidate overlap: '+json.dumps(t['overlap']['generated']))
        lines.append('  GPU comparison: '+json.dumps(t['gpu_evidence']))
    lines.extend(['\nLimitations:']+report['limitations'])
    (root/'candidate_report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    analyze(p.parse_args().root)
