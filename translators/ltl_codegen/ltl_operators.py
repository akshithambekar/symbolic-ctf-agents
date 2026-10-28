prefix = True
listed = True


def ltl_and(a: str, b: str):
    """
    Computes the LTL conjunction of two formulas.
    `a and b` means `a` and `b` hold.
    """
    if prefix:
        if listed:
            return f"'AND', {a}, {b}"
        return f"& {a} {b}"
    return f"({a}) & ({b})"


def ltl_or(a: str, b: str):
    """
    Computes the LTL disjunction of two formulas.
    `a or b` means `a` or `b` holds.
    """
    if prefix:
        if listed:
            return f"'OR', {a}, {b}"
        return f"| {a} {b}"
    return f"({a}) | ({b})"


def ltl_not(a: str):
    """
    Computes the LTL negation of a formula.
    `not a` means `a` does not hold.
    """
    if prefix:
        if listed:
            return f"'NEGATION', {a}"
        return f"! {a}"
    return f"! ({a})"


def ltl_next(a: str):
    """
    Computes the LTL next operator of a formula.
    `next a` means `a` holds in the next state.
    """
    if prefix:
        if listed:
            return f"'NEXT', {a}"
        return f"X {a}"
    return f"X ({a})"


def ltl_until(a: str, b: str):
    """
    Computes the LTL until operator of two formulas.
    `a until b` means `a` holds until `b` holds.
    """
    if prefix:
        if listed:
            return f"'UNTIL', {a}, {b}"
        return f"U {a} {b}"
    return f"({a}) U ({b})"


def ltl_eventually(a: str):
    """
    Computes the LTL eventually operator of a formula.
    `eventually a` means `a` holds at some point in the future.
    """
    if prefix:
        if listed:
            return f"'EVENTUALLY', {a}"
        return f"F {a}"
    return f"F ({a})"


def ltl_always(a: str):
    """
    Computes the LTL always operator of a formula.
    `always a` means `a` holds at all points in the future.
    """
    if prefix:
        if listed:
            return f"'ALWAYS', {a}"
        return f"G {a}"
    return f"G ({a})"


def ltl_imply(a: str, b: str):
    if prefix:
        if listed:
            return f"'IMPLY', {a}, {b}"
        return f"i {a} {b}"
    return f"({a}) -> ({b})"


def ap(action: str, obj: str):
    if listed:
        return f"'{action}({obj})'"
    return f"{action}({obj})"
