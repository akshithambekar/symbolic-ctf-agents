"""Fresh-directory benchmark with per-call records, review sheet, and coverage."""
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import itertools
import json
import os
from pathlib import Path
import shutil
import statistics
from .adapters import call_backend, UPSTREAM_COMMIT
from .dataset import dataset
from .scoring import score, equivalence


def save(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")


def run(output, model=None, endpoint="https://openrouter.ai/api/v1",
        trials=3, timeout=20, logic_backend="dfa"):
    from ..provider_config import api_key
    if not model: raise ValueError('Set OPENROUTER_MODEL in .env or supply --model (same model for both translators)')
    selected_key=api_key(endpoint)
    if not selected_key: raise ValueError('Missing API key for the selected provider')
    if trials<1: raise ValueError("trials must be positive")
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    data=dataset(); save(output/"dataset.json",data)
    source=Path(__file__).resolve().parents[1]
    shutil.copytree(source,output/"source/maritime_nsrl",ignore=shutil.ignore_patterns("__pycache__"))
    hashes={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (output/"source").rglob('*') if p.is_file()}
    hashes["dataset.json"]=hashlib.sha256((output/"dataset.json").read_bytes()).hexdigest()
    save(output/"manifest.json",hashes)
    save(output/"dependencies.json",{d.metadata['Name']:d.version for d in distributions()})
    save(output/"config.json",dict(benchmark_version=data["version"],trials=trials,
        model=model,endpoint=endpoint,logic_backend=logic_backend,timeout_seconds=timeout,
        upstream_commit=UPSTREAM_COMMIT, timestamp=datetime.now(timezone.utc).isoformat(),
        semantics="LTLf nonempty finite traces; not upstream infinite-trace Buchi semantics",
        retry_budget={"current":3,"ltlcodegen":4},sampling="Provider defaults; independent repeated calls",
        gold_review="Project-authored interpretations, human approval pending",
        credential_present=bool(selected_key)))
    old_key=os.environ.get("LTL_API_KEY")
    os.environ['LTL_API_KEY']=selected_key
    rows=[]; unavailable={}
    try:
        with (output/"raw.jsonl").open('x',encoding='utf-8') as raw:
            for trial in range(trials):
                for index,item in enumerate(data["items"]):
                    # Alternate call order to reduce deterministic temporal ordering bias.
                    methods=("current","ltlcodegen") if (trial+index)%2==0 else ("ltlcodegen","current")
                    for method in methods:
                        if method in unavailable:
                            result=dict(method=method,formula=None,attempts=[],repair_attempts=0,
                                latency_seconds=None,infrastructure_error=True,
                                error="Skipped after backend infrastructure failure: "+unavailable[method],
                                explicit_rejection=False,requested_model=model,endpoint=endpoint)
                        else:
                            result=call_backend(method,item,model,endpoint,timeout,logic_backend)
                            if result['infrastructure_error']: unavailable[method]=result['error']
                        assessment=score(item,result['formula'],result['explicit_rejection'],result['infrastructure_error'])
                        record=dict(item_id=item['id'],category=item['category'],trial=trial,
                            raw_input=item['instruction'],reference_ltl=item['reference_ltl'],
                            paraphrase_group=item['paraphrase_group'],**result,assessment=assessment)
                        rows.append(record); raw.write(json.dumps(record)+"\n"); raw.flush()
                        print(f"{method} trial={trial} {item['id']}: {assessment['score']}",flush=True)
    finally:
        if old_key is None: os.environ.pop('LTL_API_KEY',None)
        else: os.environ['LTL_API_KEY']=old_key
    summarize(output,rows)
    return output


def summarize(output, rows):
    summaries=[]; pairs=[]
    for method in ('current','ltlcodegen'):
        selected=[r for r in rows if r['method']==method]
        measured=[r for r in selected if not r['infrastructure_error']]
        mean=lambda values: statistics.mean(values) if values else None
        formula_rows=[r for r in measured if r['reference_ltl'] is not None]
        for trial in sorted({r['trial'] for r in measured}):
            for group in sorted({r['paraphrase_group'] for r in measured}):
                group_rows=[r for r in measured if r['trial']==trial and r['paraphrase_group']==group]
                for a,b in itertools.combinations(group_rows,2):
                    pair=dict(method=method,trial=trial,group=group,equivalent=None,error=None)
                    if a['formula'] and b['formula']:
                        try: pair.update(equivalence(a['formula'],b['formula']))
                        except Exception as exc: pair['error']=str(exc)
                    else: pair['error']='Missing formula; not counted as semantic agreement'
                    pairs.append(pair)
        valid_pairs=[p for p in pairs if p['method']==method and p['equivalent'] is not None]
        summaries.append(dict(method=method,scheduled=len(selected),measured=len(measured),
            not_measured=len(selected)-len(measured),
            syntax_valid_rate=mean([float(r['assessment']['syntax_valid']) for r in formula_rows]),
            compilable_rate=mean([float(r['assessment']['compilable']) for r in formula_rows]),
            semantic_correct_rate=mean([float(r['assessment']['score']=='Correct') for r in formula_rows]),
            unsupported_predicate_rate=mean([float(bool(r['assessment']['unsupported_predicates'])) for r in formula_rows]),
            repair_rate=mean([float(r['repair_attempts']>0) for r in measured]),
            mean_latency_seconds=mean([r['latency_seconds'] for r in measured]),
            paraphrase_equivalence_rate=mean([float(p['equivalent']) for p in valid_pairs]),
            paraphrase_pairs_compared=len(valid_pairs)))
    save(output/'summary.json',summaries); save(output/'paraphrases.json',pairs)
    with (output/'summary.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    with (output/'human_review.csv').open('x',newline='',encoding='utf-8') as f:
        fields=['item_id','method','trial','raw_input','reference_ltl','formula','automatic_score',
                'counterexample','reviewer_score','reviewer_rationale','reference_approved']
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({**{k:row[k] for k in fields[:6]},'automatic_score':row['assessment']['score'],
                'counterexample':json.dumps(row['assessment'].get('comparison'))})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar([s['method'] for s in summaries],[s['measured'] for s in summaries],label='Measured')
    ax.bar([s['method'] for s in summaries],[s['not_measured'] for s in summaries],
           bottom=[s['measured'] for s in summaries],label='Not measured',color='#bfc6cd')
    ax.set(ylabel='Scheduled item/trial outcomes',title='Benchmark coverage (not translator accuracy)')
    ax.legend(); fig.tight_layout(); fig.savefig(output/'coverage.png',dpi=160); plt.close(fig)
    measured=sum(s['measured'] for s in summaries)
    status=("No live translator outputs were measured: backend requests were blocked or failed. "
            "Infrastructure failures are NOT translator accuracy failures. All quality rates are null."
            if not measured else "Live outcomes are recorded; see summary.json and review counterexamples before making claims.")
    (output/'REPORT.md').write_text(f'''# CTF translator benchmark

{status}

40 statements, 20 paraphrase pairs, nine categories. The same instruction and
domain glossary are sent to both methods. Project-authored gold interpretations
must be reviewed independently. Gold formulas are never sent to the model.

Validity, semantic correctness, unsupported predicates, repairs, latency and
paraphrase consistency are in summary.csv/json. Missing measurements remain null.
Semantic correctness uses exhaustive reachable-product DFA comparison over all
nonempty finite Boolean traces, with a resource cap and counterexamples. This is
formal equivalence to the authored reference, not proof that the reference captures
every human interpretation. Physically impossible Boolean traces are not filtered.
Ordered-item correctness is full reference equivalence, a conservative ordering check.
Strict strengthening/weakening is provisionally Partially correct; reviewers may
change that grade. Syntax alone never establishes meaning. Ambiguous/unsupported
items require explicit rejection and human review. Latencies include native repairs;
current has at most 3 calls, upstream code generation 4. Provider sampling defaults
are used; request and resolved model identifiers are retained per attempt.

## Evaluated LTLCodeGen portion

Pinned commit: {UPSTREAM_COMMIT}. The upstream few-shot Python prompt and logical
operator implementations are reused verbatim (see source/upstream files and MIT
license). A shared CTF glossary maps reach(object_i) to grounded Boolean predicates.
The prompt receives an appended CTF mapping/rejection instruction and ltl_next note.
LangChain transport is replaced by the same HTTP transport as the local adapter.
Generated question() bodies are interpreted with a restricted Python AST subset;
arbitrary generated code is not executed. Imports, loops and arbitrary functions
are rejected and logged. Upstream normally writes and imports generated Python.
This restriction can affect acceptance rates and is an explicit integration tradeoff.
ROS/Gazebo, speech recognition, YOLO/object extraction, semantic mapping, Buchi
construction and motion planning are excluded. The common scorer uses LTLf, whereas
the original robotics system targets infinite-trace LTL. This is a CTF-adapted
translation-only comparison, not a reproduction of full LTLCodeGen or its paper.

## Engineering comparison

| Property | Current pipeline | LTLCodeGen translation adapter |
|---|---|---|
| LLM output | JSON formula | Python question() with logical operators |
| Core dependencies here | requests, Lark, existing DFA | same, plus AST interpreter and upstream prompt/operators |
| Repair trigger | parse/grounding/compilation failure | generated-code/normalization failure |
| Game integration | native APs, existing validated artifact | explicit object mapping then common validator |
| Full robotics stack needed here | no | no; excluded rather than installed |
| Runtime stability / latency | see measurements | see measurements |

## Architecture decision

No measured dominance is established. Retain the existing translator operationally
because it is already integrated; this is an engineering default, not an accuracy
finding. Do not adopt LTLCodeGen on unmeasured quality. A hybrid using code-generated
formulas followed by the existing grounding/finite-trace validator is feasible,
but should be evaluated as a third, separately labelled arm after this comparison.
Live accuracy, meaning preservation, unsupported-predicate frequency, repair rates,
and paraphrase stability cannot be ranked when measurements are unavailable.

Sources: https://existentialrobotics.org/LTLCodeGen/ and
https://github.com/ExistentialRobotics/LTLCodeGen/tree/{UPSTREAM_COMMIT}/speech_to_ltl
''',encoding='utf-8')
