"""Prepare and check a reviewable dataset; never calls a model or approves a reviewer."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(PAPER / 'reference_run/source'))
from candidates import FAMILIES
from maritime_nsrl.temporal import compile_ltlf
from maritime_nsrl.translator_benchmark.scoring import equivalence

INVARIANT = 'G((teammate_safe -> !tagged) & (!tagged -> teammate_safe))'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def same_meaning(a, b):
    return equivalence(f'({INVARIANT}) & ({a})', f'({INVARIANT}) & ({b})')['equivalent']


def build(output):
    output.mkdir(parents=True, exist_ok=False)
    historical = read(PAPER / 'reference_run/dataset.json')
    items = []
    for old in historical['items']:
        item = {k: old[k] for k in ('category', 'instruction', 'reference_ltl', 'expected', 'rationale', 'ordering_sensitive')}
        item.update(id='dev-' + old['id'], semantic_family='history-' + old['paraphrase_group'],
                    split='development', historical_item_id=old['id'], source_ids=['historical_pilot'])
        item['expected'] = 'formula' if old['reference_ltl'] else 'clarification_or_rejection'
        items.append(item)
    for category, family, formula, a, b, rationale, source in FAMILIES:
        for index, instruction in enumerate((a, b)):
            items.append(dict(id=f'test-{family}-{index}', category=category,
                              semantic_family=family, split='test_candidate', instruction=instruction,
                              reference_ltl=formula,
                              expected='formula' if formula else 'clarification_or_rejection',
                              rationale=rationale, ordering_sensitive=category in {'ordered', 'difficult'},
                              source_ids=[source]))
    data = dict(version='mctf-translation-candidate-2.0', title='Maritime CTF Translation Benchmark',
                release_stage='candidate', review_complete=False, glossary=historical['glossary'],
                semantics='LTLf; nonempty finite traces; inclusive F; strong X',
                proposed_grounding_invariant=INVARIANT,
                invariant_review_required=True, items=items)
    save(output / 'dataset.json', data)
    save(output / 'splits.json', {name: [i['id'] for i in items if i['split'] == name]
                                 for name in ('development', 'test_candidate')})
    for reviewer in ('A', 'B'):
        with (output / f'reviewer_{reviewer}.csv').open('x', encoding='utf-8', newline='') as stream:
            fields = ['item_id', 'instruction', 'reviewer_id', 'decision', 'proposed_ltl', 'comments']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for item in items:
                writer.writerow({'item_id': item['id'], 'instruction': item['instruction']})
    with (output / 'reference_review.csv').open('x', encoding='utf-8', newline='') as stream:
        fields = ['item_id', 'instruction', 'candidate_expected', 'candidate_ltl', 'rationale',
                  'abstract_satisfying_trace', 'abstract_violating_trace', 'source_ids']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in items:
            witness = violation = None
            if item['reference_ltl']:
                constrained = f'({INVARIANT}) & ({item["reference_ltl"]})'
                witness = compile_ltlf(constrained).report()['witness']
                violation = equivalence(constrained, INVARIANT)['right_only']
            writer.writerow(dict(item_id=item['id'], instruction=item['instruction'],
                                 candidate_expected=item['expected'], candidate_ltl=item['reference_ltl'],
                                 rationale=item['rationale'], abstract_satisfying_trace=json.dumps(witness),
                                 abstract_violating_trace=json.dumps(violation),
                                 source_ids=';'.join(item['source_ids'])))
    with (output / 'adjudication.csv').open('x', encoding='utf-8', newline='') as stream:
        fields = ['item_id', 'adjudicator_id', 'final_decision', 'final_ltl', 'rationale']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({'item_id': i['id']} for i in items)
    with (output / 'context_review.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['reviewer', 'reviewer_id', 'glossary_scope_approved',
                                                    'grounding_invariant_approved', 'comments'])
        writer.writeheader()
        writer.writerows({'reviewer': name} for name in ('A', 'B'))
    save(output / 'draft_manifest.json', {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                        for name in ('dataset.json', 'splits.json', 'reference_review.csv')})
    return check(output)


def check(output):
    data = read(output / 'dataset.json')
    items = data['items']
    ids = [i['id'] for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate item IDs')
    for name, digest in read(output / 'draft_manifest.json').items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Draft was changed without versioning: {name}')
    splits = read(output / 'splits.json')
    if sorted(ids) != sorted(splits['development'] + splits['test_candidate']):
        raise ValueError('Split membership mismatch')
    families = {}
    for item in items:
        families.setdefault(item['semantic_family'], []).append(item)
    for family, rows in families.items():
        if len(rows) != 2 or len({i['split'] for i in rows}) != 1:
            raise ValueError(f'Paraphrase split leakage or malformed family: {family}')
        if rows[0]['reference_ltl'] != rows[1]['reference_ltl']:
            raise ValueError(f'Inconsistent candidate references: {family}')
    old_formulas = {i['reference_ltl'] for i in items if i['split'] == 'development' and i['reference_ltl']}
    new_formulas = [rows[0]['reference_ltl'] for rows in families.values()
                    if rows[0]['split'] == 'test_candidate' and rows[0]['reference_ltl']]
    for formula in old_formulas | set(new_formulas):
        compile_ltlf(f'({INVARIANT}) & ({formula})')
    for index, formula in enumerate(new_formulas):
        if any(same_meaning(formula, old) for old in old_formulas):
            raise ValueError('Test reference duplicates an exposed historical development meaning: ' + formula)
        if any(same_meaning(formula, other) for other in new_formulas[:index]):
            raise ValueError('Equivalent test meanings must share a family: ' + formula)
    return dict(version=data['version'], total_items=len(items), semantic_families=len(families),
                splits=dict(Counter(i['split'] for i in items)),
                expected_outputs=dict(Counter(i['expected'] for i in items)),
                test_formula_families=len(new_formulas), historical_equivalence_overlap=0,
                review_complete=data['review_complete'], new_model_requests=0)


def review_status(output):
    data = read(output / 'dataset.json')
    items = {i['id']: i for i in data['items']}
    reviews = []
    with (output / 'context_review.csv').open(encoding='utf-8-sig', newline='') as stream:
        context = list(csv.DictReader(stream))
    context_by_label = {r['reviewer']: r for r in context}
    context_ready = (len(context) == 2 and set(context_by_label) == {'A', 'B'} and all(
        r['reviewer_id'].strip() and r['glossary_scope_approved'].strip().lower() == 'yes'
        and r['grounding_invariant_approved'].strip().lower() == 'yes' and r['comments'].strip()
        for r in context) and len({r['reviewer_id'].strip().casefold() for r in context}) == 2)
    for label in ('A', 'B'):
        with (output / f'reviewer_{label}.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        mapped = {r['item_id']: r for r in rows}
        if len(mapped) != len(rows) or set(mapped) != set(items):
            raise ValueError(f'Reviewer {label} has missing, duplicate, or unknown item IDs')
        reviews.append(mapped)
    counts = Counter()
    details = []
    for item_id, item in items.items():
        a, b = [r[item_id] for r in reviews]
        if not all(r['reviewer_id'].strip() and r['decision'].strip() and r['comments'].strip() for r in (a, b)):
            status = 'pending'
        elif a['reviewer_id'].strip().casefold() == b['reviewer_id'].strip().casefold():
            status = 'same_reviewer'
        elif context_ready and any(r['reviewer_id'].strip().casefold() !=
                 context_by_label[label]['reviewer_id'].strip().casefold()
                 for label, r in zip(('A', 'B'), (a, b))):
            status = 'reviewer_identity_mismatch'
        elif any(r['decision'] not in {'formula', 'ambiguous', 'unsupported'} for r in (a, b)):
            status = 'invalid_decision'
        elif a['decision'] != b['decision']:
            status = 'needs_adjudication'
        elif a['decision'] == 'formula':
            try:
                status = 'agreement' if (item['reference_ltl'] and
                         same_meaning(a['proposed_ltl'], b['proposed_ltl']) and
                         same_meaning(a['proposed_ltl'], item['reference_ltl'])) else 'needs_adjudication'
            except ValueError:
                status = 'invalid_or_inconclusive_formula'
        elif a['proposed_ltl'].strip() or b['proposed_ltl'].strip():
            status = 'invalid_decision'
        else:
            status = 'agreement' if (item['reference_ltl'] is None and
                                     a['decision'] == item['category']) else 'needs_adjudication'
        counts[status] += 1
        if status != 'agreement':
            details.append({'item_id': item_id, 'status': status})
    return {'counts': dict(counts), 'unresolved': details,
            'context_review_ready': context_ready,
            'ready_for_adjudication_or_release_review': context_ready and counts['agreement'] == len(items),
            'note': 'Checks declarations only; does not authenticate reviewers or approve a release.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'check', 'review-status'))
    parser.add_argument('--directory', type=Path, default=ROOT / 'candidate_v1')
    args = parser.parse_args()
    result = {'build': build, 'check': check, 'review-status': review_status}[args.command](args.directory)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
