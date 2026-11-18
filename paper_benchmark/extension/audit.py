"""Offline failure and billed-usage audit. Never retranslate or alter saved rows."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from resume_matrix import completed


def usage_of(attempt):
    usage = attempt.get('usage')
    if not isinstance(usage, dict):
        usage = (attempt.get('provider_response') or {}).get('usage')
    return usage if isinstance(usage, dict) else None


def finish_of(attempt):
    if attempt.get('finish_reason'):
        return attempt['finish_reason']
    body = attempt.get('provider_response')
    if not body:
        try:
            body = json.loads(attempt.get('response') or '{}')
        except (TypeError, ValueError):
            body = {}
    return ((body.get('choices') or [{}])[0]).get('finish_reason')


def failure_bucket(row):
    """Observable categories, not unverified causal attribution."""
    if row['infrastructure_error']:
        return 'infrastructure_not_measured'
    assessment = row['assessment']
    if assessment['score'] == 'Correct':
        return 'correct'
    if row.get('explicit_rejection'):
        return 'refused_formula_expected' if row['reference_ltl'] else 'refusal_requires_review'
    if not row.get('formula'):
        if any(finish_of(a) in ('length', 'max_tokens') for a in row.get('attempts', [])):
            return 'no_formula_output_limit'
        if any(a.get('code_error') for a in row.get('attempts', [])):
            return 'no_formula_code_interpretation_failed'
        return 'no_formula_parser_or_output_failure'
    if row['reference_ltl'] is None:
        return 'formula_invented_for_ambiguous_or_unsupported_intent'
    if not assessment.get('syntax_valid'):
        return 'invalid_syntax'
    if assessment.get('unsupported_predicates'):
        return 'unsupported_predicates'
    if not assessment.get('compilable'):
        return 'compilation_failure_or_limit'
    comparison = assessment.get('comparison', {})
    if comparison.get('left_only') is None and comparison.get('right_only') is not None:
        return 'strictly_stronger_than_reference'
    if comparison.get('right_only') is None and comparison.get('left_only') is not None:
        return 'strictly_weaker_than_reference'
    return 'different_meaning_or_inconclusive'


def write_csv(path, rows):
    if not rows:
        return
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(roots, output):
    output.mkdir(parents=True, exist_ok=False)
    chosen = completed(roots)
    failure_counts = Counter()
    reviews = []
    for (model, trial, method, item_id), entry in chosen.items():
        row = entry['row']
        bucket = failure_bucket(row)
        failure_counts[(model, method, bucket)] += 1
        if bucket != 'correct':
            reviews.append(dict(model=model, method=method, trial=trial, item_id=item_id,
                instruction=row['raw_input'], reference=row['reference_ltl'], formula=row.get('formula'),
                observed_failure=bucket, error=row.get('error'),
                source=entry['source'], line=entry['line'], reviewer_cause='', reviewer_notes=''))
    # Include every actual recorded attempt, not only the eventually retained rows.
    # Skipped placeholder rows have zero attempts and do not inflate request counts.
    ledger, hashes, seen = [], {}, set()
    for root in roots:
        for path in sorted(root.rglob('raw.jsonl')):
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            if not (path.parent / 'source_snapshot_sha256.json').exists():
                continue  # Exclude consolidated copies of already counted attempts.
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                for index, attempt in enumerate(row.get('attempts', [])):
                    usage = usage_of(attempt)
                    details = (usage or {}).get('completion_tokens_details') or {}
                    ledger.append(dict(model=row['requested_model'], method=row['method'],
                        smoke=path.parent.name == 'smoke', source=str(path), line=line_number,
                        attempt=index, is_repair=index > 0, http_status=attempt.get('http_status'),
                        infrastructure_outcome=bool(row['infrastructure_error']),
                        usage_present=usage is not None, prompt_tokens=(usage or {}).get('prompt_tokens'),
                        completion_tokens=(usage or {}).get('completion_tokens'),
                        reasoning_tokens=details.get('reasoning_tokens'),
                        total_tokens=(usage or {}).get('total_tokens'),
                        reported_cost_usd=(usage or {}).get('cost'), finish_reason=finish_of(attempt)))
    aggregates = []
    for model, method in sorted({(r['model'], r['method']) for r in ledger}):
        group = [r for r in ledger if r['model'] == model and r['method'] == method and not r['smoke']]
        aggregates.append(dict(model=model, method=method, provider_attempts=len(group),
            repair_attempts=sum(r['is_repair'] for r in group),
            infrastructure_outcome_attempts=sum(r['infrastructure_outcome'] for r in group),
            attempts_with_usage=sum(r['usage_present'] for r in group),
            prompt_tokens_reported=sum(r['prompt_tokens'] or 0 for r in group),
            completion_tokens_reported=sum(r['completion_tokens'] or 0 for r in group),
            reasoning_tokens_reported=sum(r['reasoning_tokens'] or 0 for r in group),
            total_tokens_reported=sum(r['total_tokens'] or 0 for r in group),
            attempts_with_reported_cost=sum(r['reported_cost_usd'] is not None for r in group),
            reported_cost_usd=(sum(r['reported_cost_usd'] or 0 for r in group)
                               if any(r['reported_cost_usd'] is not None for r in group) else None)))
    counts = [dict(model=m, method=t, observed_failure=f, outcomes=n)
              for (m, t, f), n in sorted(failure_counts.items())]
    write_csv(output / 'usage_attempts.csv', ledger)
    write_csv(output / 'usage_summary.csv', aggregates)
    write_csv(output / 'failure_counts.csv', counts)
    write_csv(output / 'failure_review.csv', reviews)
    manifest = dict(timestamp=datetime.now(timezone.utc).isoformat(),
        source_roots=[str(p) for p in roots], input_hashes=hashes,
        measured_outcomes=len(chosen), failure_counts=counts, usage=aggregates,
        limits=['Missing usage/cost is unknown, never inferred as zero.',
                'Reasoning tokens are normally included in completion tokens; do not add twice.',
                'All actual recorded attempts count, including repairs and failed/recovered requests.',
                'Smoke attempts are in the ledger but excluded from main aggregates.',
                'Failure buckets are observed symptoms, not proven causal diagnoses.'])
    (output / 'audit.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    report = ['# Translator failure and usage audit', '',
              'Offline analysis of preserved outputs. No new model calls.', '',
              '## Observed outcomes', '', '| Model | Method | Failure category | Count |',
              '|---|---|---|---:|']
    report += [f"| {r['model']} | {r['method']} | {r['observed_failure']} | {r['outcomes']} |" for r in counts]
    report += ['', '## Interpretation', '',
        'IBM: refusal on a formula-reference case can reflect the ONE-template capability restriction. '
        'It is not by itself a parsing failure or a wrong choice of temporal operator. '
        'Output-limit cases are counted separately from explicit refusal and malformed output.',
        'LTLCodeGen: inspect code_error in every attempt to distinguish code construction failures '
        'from formulas that parse successfully but strengthen, weaken or otherwise change the reference.',
        'The semantic labels measure unconstrained Boolean LTLf agreement, not physical game equivalence. '
        'Use failure_review.csv for human causal annotations before making attribution claims.', '',
        '## Token and cost accounting', '', *manifest['limits']]
    (output / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervisor', type=Path, default=ROOT / 'runs/w-1106-085330/status.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    roots = [Path(p) for p in json.loads(args.supervisor.read_text())['roots']]
    result = audit(roots, args.output)
    print(json.dumps({'output': str(args.output), 'measured': result['measured_outcomes']}))
