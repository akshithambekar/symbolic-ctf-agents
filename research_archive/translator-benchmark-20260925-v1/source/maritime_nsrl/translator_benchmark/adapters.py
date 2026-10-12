"""Existing adapter capture and translation-only LTLCodeGen integration.

LTLCodeGen commit b5eb6a19769c3b06c5757136bef3bd444533e328.
Upstream prompts/operators preserved; generated Python is interpreted, never exec'd.
"""
import ast
import json
import os
from pathlib import Path
import time
from unittest.mock import patch
import requests
from ..translation import translate
from .dataset import GLOSSARY
from .scoring import normalize_output
from .upstream import ltl_operators

UPSTREAM_COMMIT="b5eb6a19769c3b06c5757136bef3bd444533e328"
NAMES=["in_bounds","tagged","has_flag","captured","home","defender_behind_center","attacker_forward","teammate_safe"]
MAPPING={f"object_{i}":p for i,p in enumerate(NAMES)}
CONTEXT=GLOSSARY+"\nFor LTLCodeGen reach(object_i) denotes the truth of the mapped CTF predicate, not physical reach:\n"+json.dumps(MAPPING)


def interpret(code):
    """Bounded expression subset of upstream question(): strings, assignments, calls."""
    code=code.strip()
    if code.startswith("```"):
        lines=code.splitlines()
        if lines[-1].strip()!="```": raise ValueError("Unclosed code fence")
        code="\n".join(lines[1:-1])
    if len(code)>16000: raise ValueError("Generated code too long")
    tree=ast.parse(code)
    if sum(1 for _ in ast.walk(tree))>1200: raise ValueError("Generated code exceeds AST budget")
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef): raise ValueError("Expected only def question()")
    fn=tree.body[0]
    if fn.name!="question" or fn.decorator_list or fn.args.args or fn.args.defaults or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg:
        raise ValueError("Expected undecorated zero-argument question")
    operators={name:getattr(ltl_operators,name) for name in dir(ltl_operators) if name.startswith("ltl_") or name=="ap"}
    values={"reach":"reach"}
    def expr(node):
        if isinstance(node,ast.Constant) and isinstance(node.value,str): return node.value
        if isinstance(node,ast.Name) and node.id in values: return values[node.id]
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in operators and not node.keywords:
            result=operators[node.func.id](*(expr(arg) for arg in node.args))
            if len(result)>16000: raise ValueError("Generated formula too long")
            return result
        raise ValueError(f"Unsupported generated expression: {type(node).__name__}")
    for statement in fn.body:
        if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Constant) and isinstance(statement.value.value,str): continue
        if isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name):
            values[statement.targets[0].id]=expr(statement.value)
        elif isinstance(statement,ast.Return):
            return "["+expr(statement.value)+"]"
        else: raise ValueError(f"Unsupported generated statement: {type(statement).__name__}")
    raise ValueError("Missing return")


def call_backend(method, item, model, endpoint, timeout=30, logic_backend="dfa"):
    attempts=[]; started=time.perf_counter(); original=requests.post
    def captured_post(url, **kwargs):
        payload=kwargs["json"]
        attempt={"request":payload,"latency_seconds":None,"response":None,"error":None}
        attempts.append(attempt); tick=time.perf_counter()
        try:
            response=original(url,**{**kwargs,"timeout":timeout})
            attempt["http_status"]=response.status_code
            attempt["response"]=response.text
            response.raise_for_status()
            body=response.json()
            attempt["resolved_model"]=body.get("model")
            attempt["usage"]=body.get("usage")
            attempt["raw_output"]=body["choices"][0]["message"]["content"]
            return response
        except Exception as exc:
            attempt["error"]=type(exc).__name__+": "+str(exc)
            raise
        finally: attempt["latency_seconds"]=time.perf_counter()-tick
    result=dict(method=method,requested_model=model,endpoint=endpoint,attempts=attempts,
                formula=None,explicit_rejection=False,infrastructure_error=False,error=None,
                integration="existing-adapter" if method=="current" else "LTLCodeGen-translation-only-CTF-adapted")
    instruction=CONTEXT+"\nInstruction:\n"+item["instruction"]
    try:
        if method=="current":
            with patch("maritime_nsrl.translation.requests.post",side_effect=captured_post):
                artifact=translate(instruction,"llm",model,endpoint,logic_backend)
            result.update(formula=artifact["ltl"],translation_artifact=artifact)
        elif method=="ltlcodegen":
            template=(Path(__file__).parent/"upstream/code_ltl_prompt_template.py").read_text()
            previous=reason=""
            # Upstream: initial request plus max_retries=3. Repairs on code failure only.
            for _ in range(4):
                prompt=template.format(instruction=instruction,previous_answer=previous,failure_reason=reason)
                prompt += "\nCTF adaptation: use the supplied object mapping for all eight predicates. If unsupported, return JSON {\"error\":\"reason\"}. Strong-next operator ltl_next is also available."
                key=os.environ.get("LTL_API_KEY") or os.environ.get("OPENAI_API_KEY")
                response=captured_post(endpoint.rstrip('/')+'/chat/completions',
                    headers={"Authorization":f"Bearer {key}"} if key else {},
                    json={"model":model,"messages":[{"role":"user","content":prompt}]})
                raw=response.json()["choices"][0]["message"]["content"]
                try:
                    refusal=json.loads(raw)
                    if isinstance(refusal,dict) and "error" in refusal:
                        result["explicit_rejection"]=True; break
                except (ValueError,TypeError): pass
                try:
                    prefix=interpret(raw)
                    result["raw_prefix_ltl"]=prefix
                    # No validator feedback added to upstream repair loop.
                    result["formula"]=normalize_output(prefix,MAPPING)
                    break
                except Exception as exc:
                    attempts[-1]["code_error"]=str(exc)
                    previous="\nHere is your previous answer:\n"+raw
                    reason="\nBut it failed with the following error:\n"+str(exc)
            result["upstream_commit"]=UPSTREAM_COMMIT
        else: raise ValueError(method)
    except requests.RequestException as exc:
        result.update(infrastructure_error=True,error=type(exc).__name__+": "+str(exc))
    except Exception as exc: result["error"]=type(exc).__name__+": "+str(exc)
    for attempt in reversed(attempts):
        try:
            raw=json.loads(attempt.get("raw_output") or "null")
            if isinstance(raw,dict) and "error" in raw: result["explicit_rejection"]=True
        except ValueError: pass
    result.update(latency_seconds=time.perf_counter()-started,repair_attempts=max(len(attempts)-1,0))
    return result
