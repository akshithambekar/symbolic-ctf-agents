"""DFA language comparison, not exact-string or finite-sample equality."""
import ast
from collections import deque
import re
from ..logic import AP, PARSER, atoms
from ..temporal import compile_ltlf

OPS = {"AND":("&",2), "OR":("|",2), "IMPLY":("->",2),
       "NEGATION":("!",1), "NEXT":("X",1), "UNTIL":("U",2),
       "EVENTUALLY":("F",1), "ALWAYS":("G",1)}


def normalize_output(value, mapping=None):
    mapping = mapping or {}
    if not isinstance(value, str) or len(value)>16000: raise ValueError("Invalid formula size/type")
    value=value.strip()
    if value.startswith("["):
        tokens=ast.literal_eval(value)
        if not isinstance(tokens,list) or len(tokens)>1000: raise ValueError("Invalid prefix output")
        position=0
        def parse():
            nonlocal position
            if position>=len(tokens): raise ValueError("Truncated prefix formula")
            token=tokens[position]; position+=1
            if token in OPS:
                op,n=OPS[token]; a=parse()
                return f"{op}({a})" if n==1 else f"({a}) {op} ({parse()})"
            match=re.fullmatch(r"reach\((object_\d+)\)",token)
            if match:
                if match[1] not in mapping: raise ValueError(f"Unknown object mapping: {token}")
                return mapping[match[1]]
            return token
        value=parse()
        if position!=len(tokens): raise ValueError("Unused prefix tokens")
    # Only lexical operator aliases; never replace AP meaning or drop clauses.
    value=value.replace("&&","&").replace("||","|").replace("¬","!").replace("∧","&").replace("∨","|").replace("→","->")
    PARSER.parse(value)
    return value


def equivalence(left, right, max_pairs=65536):
    a,b=compile_ltlf(left),compile_ltlf(right)
    names=sorted(set(a.propositions)|set(b.propositions))
    alphabet=[frozenset(p for i,p in enumerate(names) if mask & (1<<i)) for mask in range(1<<len(names))]
    queue=deque([(0,0,[])])
    seen={(0,0)}; left_only=right_only=None
    while queue:
        x,y,path=queue.popleft()
        for labels in alphabet:
            nx,ny=a.advance(x,labels),b.advance(y,labels)
            witness=path+[sorted(labels)]
            accept_a,accept_b=nx in a.accepting,ny in b.accepting
            if accept_a and not accept_b and left_only is None: left_only=witness
            if accept_b and not accept_a and right_only is None: right_only=witness
            if left_only is not None and right_only is not None:
                return dict(equivalent=False,left_only=left_only,right_only=right_only,exhaustive=False)
            if (nx,ny) not in seen:
                if len(seen)>=max_pairs: raise ValueError("Equivalence resource limit; inconclusive")
                seen.add((nx,ny)); queue.append((nx,ny,witness))
    return dict(equivalent=left_only is None and right_only is None,
                left_only=left_only,right_only=right_only,exhaustive=True)


def score(item, formula=None, explicit_rejection=False, infrastructure_error=False):
    result=dict(syntax_valid=False, supported_predicates=None, unsupported_predicates=[],
                compilable=False, semantic_correctness=None, ordering_preserved=None,
                score="Invalid/unusable", human_review_required=True)
    if infrastructure_error:
        return {**result,"score":"Not measured","reason":"Infrastructure failure; excluded from translator quality metrics"}
    if formula is None:
        if explicit_rejection and item["reference_ltl"] is None:
            return {**result,"score":"Correct","semantic_correctness":True,
                    "reason":"Appropriate explicit rejection/clarification; review wording"}
        return {**result,"reason":"No usable formula"}
    try:
        normalized=normalize_output(formula)
        result.update(normalized_ltl=normalized,syntax_valid=True)
        unsupported=sorted(atoms(PARSER.parse(normalized))-AP)
        result.update(supported_predicates=not unsupported,unsupported_predicates=unsupported)
        if unsupported: return {**result,"reason":"Unsupported predicates"}
        result["compilation"]=compile_ltlf(normalized).report(); result["compilable"]=True
        if item["reference_ltl"] is None:
            return {**result,"score":"Incorrect","semantic_correctness":False,
                    "reason":"Invented a precise formula for unsupported/ambiguous intent"}
        comparison=equivalence(normalized,item["reference_ltl"])
        result["comparison"]=comparison
        result["semantic_correctness"]=comparison["equivalent"]
        if item.get("ordering_sensitive"): result["ordering_preserved"]=comparison["equivalent"]
        if comparison["equivalent"]:
            result.update(score="Correct",human_review_required=False)
        elif comparison["left_only"] is None or comparison["right_only"] is None:
            result.update(score="Partially correct",reason="Strict strengthening/weakening; provisional human review")
        else: result.update(score="Incorrect",reason="Incomparable trace languages")
    except Exception as exc:
        result["reason"]=str(exc)
    return result
