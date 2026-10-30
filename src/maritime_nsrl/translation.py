"""Offline controlled English and an OpenAI-compatible HTTP translator.

The model returns declarative JSON, never executable Python or game actions.
"""

import json
import os
import requests
from .logic import validate
from .probe_lure import INSTRUCTION, FORMULA

CLAUSES = {
    "always stay within the arena": "G(in_bounds)",
    "never get tagged": "G(!tagged)",
    "eventually capture the enemy flag": "F(captured)",
    "pick up the enemy flag and then capture it": "F(has_flag & F(captured))",
    "eventually return home": "F(home)",
    "keep the defender behind the centerline": "G(defender_behind_center)",
    "keep the designated defender behind the centerline while the team is safe": "G(teammate_safe -> defender_behind_center)",
}
CLAUSES[INSTRUCTION.rstrip(".").lower()] = FORMULA
SYSTEM = """Translate Maritime Capture the Flag instructions into JSON {"ltl": "..."}.
No code, no actions, no Markdown. Only these grounded propositions exist:
in_bounds: boat center inside arena; tagged: currently tagged;
has_flag: currently carrying enemy flag; captured: successful score event;
home: within catch_radius of own flag home; defender_behind_center: the designated
defender is behind the centerline; attacker_forward: an attacker is beyond it;
teammate_safe: no controlled teammate is tagged.
Use only conjunctions of G(propositional boolean predicates) and at most one
mission F(p) or F(p & F(q & F(r ...))). Boolean operators: ! & | ->.
F permits same-step satisfaction. Semantics: finite episode, all eventualities
must be fulfilled by its end. Do not invent extra constraints.
Capture requires returning to own territory in this PyQuaticus version, not
necessarily to the flag stand. Example: "Always stay within the arena. Pick up
the enemy flag and then capture it." -> {"ltl":"G(in_bounds) & F(has_flag & F(captured))"}.
If the requested meaning cannot be expressed exactly in this fragment, return
{"error":"explanation"}. Never silently weaken an unsupported instruction."""


def translate(text, backend="offline", model=None, base_url=None, logic_backend="fragment"):
    if logic_backend == "dfa":
        from .temporal import compile_ltlf

        validate_formula = compile_ltlf
        system = """Translate Maritime Capture the Flag instructions to JSON {"ltl":"..."}.
Return data only, no code or Markdown. Allowed propositions: in_bounds (all controlled
boats inside arena), tagged (any controlled boat tagged), has_flag (any controlled
boat carries enemy flag), captured (team scores on this transition), home (any
controlled boat within catch radius of own flag). Use ! & | -> X U F G with arbitrary
nesting. Strong X requires a next state; U is strong until. Finite-trace semantics:
eventual obligations must finish by the episode deadline. Do not invent constraints.
Example: Whenever carrying a flag, eventually capture it: G(has_flag -> F(captured)).
If intent is ambiguous or cannot be expressed, return {"error":"explanation"}."""
    elif logic_backend == "fragment":
        validate_formula, system = validate, SYSTEM
    else:
        raise ValueError("Unknown logic backend")
    attempts = []
    if backend == "offline":
        clauses = [x.strip().lower() for x in text.strip().split(".") if x.strip()]
        vocabulary = {**CLAUSES}
        if logic_backend == "dfa":
            vocabulary.update(
                {
                    "whenever carrying a flag, eventually capture it": "G(has_flag -> F(captured))",
                    "stay untagged until capturing the flag": "(!tagged U captured)",
                    "on the next step stay inside the arena": "X(in_bounds)",
                }
            )
        unknown = [x for x in clauses if x not in vocabulary]
        if unknown:
            raise ValueError(
                f"Unsupported controlled English: {unknown}. Use LLM backend or listed examples."
            )
        formula = " & ".join(vocabulary[x] for x in clauses)
        spec = validate_formula(formula)
    elif backend == "llm":
        if not model:
            raise ValueError("Specify --model for the LLM backend")
        url = (base_url or os.environ.get("LTL_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        key = os.environ.get("LTL_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
        for attempt in range(3):
            response = requests.post(
                url + "/chat/completions",
                headers=headers,
                json={"model": model, "messages": messages},
                timeout=90,
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            try:
                data = json.loads(raw)
                if "error" in data:
                    raise ValueError(data["error"])
                spec = validate_formula(data["ltl"])
                attempts.append({"response": raw, "valid": True})
                break
            except (ValueError, KeyError, TypeError) as exc:
                attempts.append({"response": raw, "valid": False, "error": str(exc)})
                if attempt == 2:
                    raise ValueError(f"Translation failed validation: {attempts}") from exc
                messages += [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f"Validation failed: {exc}. Correct the JSON/formula without changing the instruction.",
                    },
                ]
    else:
        raise ValueError(f"Unknown translator: {backend}")
    return {
        "instruction": text,
        "backend": backend,
        "model": model,
        "logic_backend": logic_backend,
        "ltl": spec.formula,
        "validation": spec.report(),
        "attempts": attempts,
    }


def load_spec(path):
    # Always revalidate the formula; do not trust a saved validation flag.
    from .specifications import read_artifact

    return validate(read_artifact(path)["ltl"])
