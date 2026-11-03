# CTF translator benchmark

Live outcomes are recorded; see summary.json and review counterexamples before making claims.

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

Pinned commit: b5eb6a19769c3b06c5757136bef3bd444533e328. The upstream few-shot Python prompt and logical
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
https://github.com/ExistentialRobotics/LTLCodeGen/tree/b5eb6a19769c3b06c5757136bef3bd444533e328/speech_to_ltl
