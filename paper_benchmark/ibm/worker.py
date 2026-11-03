"""Offline IBM engine worker: official pattern parsing and formula construction.

No provider credential is accepted. HTTP transport is supplied by the parent.
"""
import importlib.metadata
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from nl2ltl import translate
from nl2ltl.engines.gpt import core
from nl2ltl.engines.gpt.output import parse_gpt_output
from pylogics.syntax.base import AbstractAtomic, And, Or, Not, Implies
from pylogics.syntax.ltl import Always, Eventually, Next, Until

PROMPT = Path(__file__).resolve().parent / 'upstream/prompt.json'
PATTERNS = ('Existence', 'ExistenceTwo', 'Absence', 'RespondedExistence',
            'Response', 'Precedence', 'ChainResponse', 'NotCoExistence')


def infix(node):
    # Structural conversion preserves strong Next; no string replacement of X.
    if isinstance(node, AbstractAtomic):
        return str(node.name)
    for cls, token in ((Not, '!'), (Always, 'G'), (Eventually, 'F'), (Next, 'X')):
        if isinstance(node, cls):
            return f'{token}({infix(node.argument)})'
    for cls, token in ((And, '&'), (Or, '|'), (Implies, '->'), (Until, 'U')):
        if isinstance(node, cls):
            return f' {token} '.join(f'({infix(child)})' for child in node.operands)
    raise ValueError(f'Unsupported IBM formula node: {type(node).__name__}')


class CTFEngine(core.GPTEngine):
    def _check_consistency(self):
        # Adapt only the provider/model gate. The inherited translate path and
        # official output parser/grounding/formula constructors remain in use.
        if importlib.metadata.version('openai') != '1.12.0':
            raise ValueError('IBM worker requires OpenAI 1.12.0')
        if self.operation_mode != core.OperationModes.CHAT.value:
            raise ValueError('Only the upstream chat mode is adapted')


def process(data):
    if importlib.metadata.version('nl2ltl') != '0.0.6':
        raise ValueError('IBM worker requires nl2ltl 0.0.6')
    content = data.get('response', 'PATTERN: existence\nSYMBOLS: captured')
    if not isinstance(content, str) or len(content) > 65536:
        raise ValueError('Invalid model output size/type')
    payloads = []

    def completion(**payload):
        payloads.append(payload)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    core.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
    engine = CTFEngine(model=data['model'], prompt=PROMPT)
    engine._prompt += ('\n\nCTF DOMAIN ADAPTATION:\n' + data['glossary'] +
                      '\nFor this domain ALLOWED_SYMBOLS are exactly the eight predicate names above.'
                      '\nThe library supports these DECLARE templates: ' + ', '.join(PATTERNS) +
                      '. Select ONE template that preserves the whole instruction exactly.'
                      '\nReturn PATTERN: template_name followed by a newline and SYMBOLS: '
                      'predicate_names separated by comma-space, in template argument order.'
                      '\nIf no single template captures the instruction, or the intent is ambiguous '
                      'or ungrounded, return UNSUPPORTED: explanation. Do not omit requested clauses.'
                      '\nDo not return an arbitrary LTL formula or Python code.\n')
    if data['action'] == 'request':
        # Exercise the inherited engine request construction with a canned reply.
        translate(data['instruction'], engine)
        return {'request': payloads[0], 'patterns': list(PATTERNS)}
    if content.strip().startswith('UNSUPPORTED:'):
        return {'formula': None, 'explicit_rejection': True, 'candidates': []}
    formulas = translate(data['instruction'], engine)
    candidates = sorted(({'formula': infix(template.to_ltlf()), 'confidence': confidence,
                          'template': type(template).__name__}
                         for template, confidence in formulas.items()), key=lambda x: x['formula'])
    parsed = parse_gpt_output(SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content))]), 'chatCompletion')
    result = {'candidates': candidates, 'pattern': parsed.pattern, 'symbols': parsed.entities,
              'explicit_rejection': False, 'formula': None}
    if not candidates:
        result['error'] = 'IBM returned no candidate'
    else:
        best = max(row['confidence'] for row in candidates)
        winners = {row['formula'] for row in candidates if row['confidence'] == best}
        if len(winners) != 1:
            result['error'] = 'Multiple equally ranked IBM candidates; no gold-based selection allowed'
        else:
            result['formula'] = next(iter(winners))
    return result


if __name__ == '__main__':
    try:
        data = json.loads(sys.stdin.read(131073))
        result = process(data)
    except Exception as exc:
        result = {'formula': None, 'explicit_rejection': False,
                  'error': f'{type(exc).__name__}: {exc}'}
    print(json.dumps(result))
