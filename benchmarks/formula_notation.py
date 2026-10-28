"""Bounded, data-only conversion from benchmark notations to strict Spot LTL."""

from dataclasses import dataclass
import re


MAX_CHARACTERS = 16_384
MAX_TOKENS = 2048
MAX_DEPTH = 64
IDENTIFIER = r"[A-Za-z_][A-Za-z_0-9]*"
IDENT = re.compile(IDENTIFIER)
PREDICATE = re.compile(rf"{IDENTIFIER}\({IDENTIFIER}\)")
PREFIX_OPERATORS = {
    "AND": ("&", 2), "OR": ("|", 2), "NEGATION": ("!", 1),
    "NEXT": ("X", 1), "UNTIL": ("U", 2), "EVENTUALLY": ("F", 1),
    "ALWAYS": ("G", 1), "IMPLY": ("->", 2),
}
RESERVED = {*PREFIX_OPERATORS, "F", "G", "X", "U", "true", "false"}
QUOTED_TOKEN = re.compile(r"'([^'\\\r\n]*)'|\"([^\"\\\r\n]*)\"")
PRECEDENCE = {"->": 1, "|": 2, "&": 3, "U": 4}


class InvalidFormula(ValueError):
    """Malformed or over-limit input in the benchmark's supported grammar."""


@dataclass(frozen=True)
class Node:
    operator: str
    operands: tuple = ()
    atom: str | None = None
    depth: int = 1


def node(operator, *operands):
    depth = 1 + max((item.depth for item in operands), default=0)
    if depth > MAX_DEPTH:
        raise InvalidFormula("Formula exceeds the nesting limit.")
    return Node(operator, operands, depth=depth)


def atom(value):
    if value in {"true", "false"}:
        return node(value)
    name = value.split("(", 1)[0]
    if name in RESERVED or not (IDENT.fullmatch(value) or PREDICATE.fullmatch(value)):
        raise InvalidFormula("Invalid atomic proposition or unsupported operator.")
    return Node("atom", atom=value)


def check_text(text):
    if not isinstance(text, str) or not text.strip():
        raise InvalidFormula("Formula must be nonempty text.")
    if len(text) > MAX_CHARACTERS:
        raise InvalidFormula("Formula exceeds the character limit.")


def prefix_tree(text):
    """Parse a flat list of quoted tokens, with no Python literal evaluation."""
    check_text(text)
    source = text.strip()
    if not source.startswith("[") or not source.endswith("]"):
        raise InvalidFormula("Expected a flat listed-prefix formula.")
    tokens = []
    position = 1
    end = len(source) - 1
    while position < end:
        while position < end and source[position].isspace():
            position += 1
        if position == end:
            break
        match = QUOTED_TOKEN.match(source, position)
        if not match:
            raise InvalidFormula("Expected a quoted prefix token without escapes.")
        tokens.append(match.group(1) if match.group(1) is not None else match.group(2))
        if len(tokens) > MAX_TOKENS:
            raise InvalidFormula("Formula exceeds the token limit.")
        position = match.end()
        while position < end and source[position].isspace():
            position += 1
        if position == end:
            break
        if source[position] != ",":
            raise InvalidFormula("Expected a comma between prefix tokens.")
        position += 1
    cursor = 0

    def parse(depth=1):
        nonlocal cursor
        if depth > MAX_DEPTH:
            raise InvalidFormula("Formula exceeds the nesting limit.")
        if cursor >= len(tokens):
            raise InvalidFormula("Missing prefix operand.")
        value = tokens[cursor]
        cursor += 1
        if value in PREFIX_OPERATORS:
            operator, arity = PREFIX_OPERATORS[value]
            return node(operator, *(parse(depth + 1) for _ in range(arity)))
        return atom(value)

    tree = parse()
    if cursor != len(tokens):
        raise InvalidFormula("Extra prefix operands.")
    return tree


def infix_tokens(text):
    check_text(text)
    result = []
    position = 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        match = IDENT.match(text, position)
        if match:
            token = match.group()
            # A predicate spelling is indivisible. Reserved temporal operators
            # still parse F(p) and G(p | q) as operators followed by parentheses.
            if token not in RESERVED and match.end() < len(text) and text[match.end()] == "(":
                predicate = PREDICATE.match(text, position)
                if not predicate:
                    raise InvalidFormula("Malformed predicate proposition.")
                token = predicate.group()
                position = predicate.end()
            else:
                position = match.end()
        elif text.startswith("->", position):
            token = "->"
            position += 2
        elif text[position] in "!&|()":
            token = text[position]
            position += 1
        else:
            raise InvalidFormula("Unsupported infix token.")
        result.append(token)
        if len(result) > MAX_TOKENS:
            raise InvalidFormula("Formula exceeds the token limit.")
    return result


def infix_tree(text):
    tokens = infix_tokens(text)
    cursor = 0

    def parse(minimum=1, depth=1):
        nonlocal cursor
        if depth > MAX_DEPTH:
            raise InvalidFormula("Formula exceeds the nesting limit.")
        if cursor >= len(tokens):
            raise InvalidFormula("Missing infix operand.")
        value = tokens[cursor]
        cursor += 1
        if value in {"!", "X", "F", "G"}:
            left = node(value, parse(5, depth + 1))
        elif value == "(":
            left = parse(1, depth + 1)
            if cursor >= len(tokens) or tokens[cursor] != ")":
                raise InvalidFormula("Missing closing parenthesis.")
            cursor += 1
        else:
            left = atom(value)
        while cursor < len(tokens) and PRECEDENCE.get(tokens[cursor], 0) >= minimum:
            operator = tokens[cursor]
            cursor += 1
            precedence = PRECEDENCE[operator]
            # Implication and strong until associate to the right.
            right = parse(precedence if operator in {"->", "U"} else precedence + 1, depth + 1)
            left = node(operator, left, right)
        return left

    tree = parse()
    if cursor != len(tokens):
        raise InvalidFormula("Extra infix operands or delimiters.")
    return tree


def to_spot(text, notation):
    if notation == "listed_prefix":
        tree = prefix_tree(text)
    elif notation == "infix":
        tree = infix_tree(text)
    else:
        raise InvalidFormula("Unsupported formula notation.")
    propositions = set()

    def render(item):
        if item.operator == "atom":
            propositions.add(item.atom)
            # The accepted ASCII grammar excludes quotes and escape characters.
            # Double quotes make the entire predicate spelling one Spot atom.
            return '"' + item.atom + '"'
        if not item.operands:
            return "1" if item.operator == "true" else "0"
        if len(item.operands) == 1:
            return f"{item.operator}({render(item.operands[0])})"
        return f"({render(item.operands[0])} {item.operator} {render(item.operands[1])})"

    return render(tree), sorted(propositions)
