# Please help write the code for translating instruction to LTL formula.

from available_actions import reach  # available actions

# Necessary functions and variables are imported for the translation.
from ltl_operators import ap  # ap(action, obj)
from ltl_operators import ltl_and, ltl_or, ltl_not, ltl_until, ltl_eventually, ltl_always, ltl_imply  # LTL operators


def example_0():
    """
    Here are examples of how to use the LTL operators
    """
    # create atomic propositions
    a = ap(reach, "a")  # a
    b = ap(reach, "b")  # b

    # example of using LTL operators
    ltl_and(a, b)  # `a` and `b`, accepts two arguments
    ltl_or(a, b)  # `a` or `b`, accepts two arguments
    ltl_not(a)  # not `a`, accepts one argument
    ltl_until(a, b)  # `a` should be true until `b` is true, accepts two arguments
    ltl_until(a, b)  # until `b` is true, `a` should be true, i.e. `a` holds until `b` holds
    ltl_eventually(a)  # eventually `a`, i.e. `a` holds at some point in the future, accepts one argument
    ltl_always(a)  # always `a`, i.e. `a` holds at all points in the future, accepts one argument
    ltl_imply(a, b)  # `a` implies `b`, i.e. if `a` holds then `b` holds, accepts two arguments


# Here are some examples of translating instruction to LTL. The instruction is provided in the function docstring.
# Each example is a function that returns the LTL formula based on the instruction.
# Ignore objects in the instruction that are not in the form object_x, where x is a number.


def example_1():
    """
    Reach object_2 and object_1
    """
    # explanation of the instruction:
    # object_1 and object_2 are both reached at some point in the future, but no specific order is mentioned explicitly or implicitly.

    # create atomic propositions for objects
    reach_obj_2 = ap(reach, "object_2")  # reach(object_2)
    reach_obj_1 = ap(reach, "object_1")  # reach(object_1)
    # describe the constraints in the instruction
    c1 = ltl_eventually(reach_obj_2)  # Reach object_2 at some point
    c2 = ltl_eventually(reach_obj_1)  # Reach object_1 at some point
    c3 = ltl_and(c1, c2)  # Reach object_2 and object_1
    return c3


def example_2():
    """
    Reach object_8, object_13, object_14, object_17 in the room.
    """
    # explanation of the instruction:
    # The robot is expected to reach object_8, object_13, object_14, and object_17 at some point in the future.
    # The order of reaching the objects is not specified explicitly or implicitly.

    # create atomic propositions for objects
    reach_obj_8 = ap(reach, "object_8")  # reach(object_8)
    reach_obj_13 = ap(reach, "object_13")  # reach(object_13)
    reach_obj_14 = ap(reach, "object_14")  # reach(object_14)
    reach_obj_17 = ap(reach, "object_17")  # reach(object_17)
    # describe the constraints in the instruction
    c1 = ltl_eventually(reach_obj_8)  # Reach object_8 at some point
    c2 = ltl_eventually(reach_obj_13)  # Reach object_13 at some point
    c3 = ltl_eventually(reach_obj_14)  # Reach object_14 at some point
    c4 = ltl_eventually(reach_obj_17)  # Reach object_17 at some point
    formula = ltl_and(c1, c2)  # Reach object_8 and object_13
    formula = ltl_and(formula, c3)  # Reach object_8, object_13 and object_14
    formula = ltl_and(formula, c4)  # Reach object_8, object_13, object_14 and object_17
    return formula


def example_3():
    """
    Reach object_2 and object_1 simultaneously
    """
    # explanation of the instruction:
    # The robot is expected to reach object_2 and object_1 simultaneously.
    # The mentioned objects should be reached at the same time.

    # create atomic propositions for objects
    reach_obj_1 = ap(reach, "object_1")  # reach(object_1)
    reach_obj_2 = ap(reach, "object_2")  # reach(object_2)
    # describe the constraints in the instruction
    c1 = ltl_and(reach_obj_1, reach_obj_2)  # object_1 and object_2 are reached simultaneously
    c2 = ltl_eventually(c1)  # object_1 and object_2 are reached at some point in the future
    return c2


def example_4():
    """
    Reach object_2 until object_1 is reached.
    """
    # explanation of the instruction:
    # The robot is expected to keep reaching object_2 until object_1 is reached. (`do something until` rule)

    # create atomic propositions for objects
    reach_obj_1 = ap(reach, "object_1")  # reach(object_1)
    reach_obj_2 = ap(reach, "object_2")  # reach(object_2)
    # describe the constraints in the instruction
    c1 = ltl_until(reach_obj_2, reach_obj_1)  # Reach object_2 until object_1 is reached
    return c1


def example_5():
    """
    Don't reach object_2 until object_1 is reached.
    """
    # explanation of the instruction:
    # The robot is expected to reach object_1 before object_2. The instruction does not require the robot to reach any object.

    # create atomic propositions for objects
    reach_obj_1 = ap(reach, "object_1")  # reach(object_1)
    reach_obj_2 = ap(reach, "object_2")  # reach(object_2)
    # describe the constraints in the instruction
    c1 = ltl_until(ltl_not(reach_obj_2), reach_obj_1)  # Don't reach object_2 until object_1 is reached
    return c1


def example_6():
    """
    For all time steps, until reach(object_1) and reach(object_2) is true, don't start reach(object_3).
    """
    # explanation of the instruction:
    # The robot should not reach object_3 until it has reached object_1 and object_2.
    # The instruction does not require the robot to reach any object.

    # create atomic propositions for objects
    reach_obj_1 = ap(reach, "object_1")  # reach(object_1)
    reach_obj_2 = ap(reach, "object_2")  # reach(object_2)
    reach_obj_3 = ap(reach, "object_3")  # reach(object_3)
    # describe the constraints in the instruction
    c1 = ltl_and(reach_obj_1, reach_obj_2)  # Reach object_1 and object_2
    c2 = ltl_not(reach_obj_3)  # Don't reach object_3
    formula = ltl_until(c2, c1)  # Until reach(object_1) and reach(object_2) is true, don't start reach(object_3)
    return formula


def example_7():
    """
    Reach object_1 then object_2.
    """
    # explanation of the instruction:
    # The robot is expected to reach object_1 and then object_2.
    # object_1 to object_2: object_2 should not be reached before object_1 (`not until rules`)

    # create atomic propositions for objects
    reach_obj_1 = ap(reach, "object_1")  # reach(object_1)
    reach_obj_2 = ap(reach, "object_2")  # reach(object_2)
    # describe the constraints in the instruction
    c1 = ltl_eventually(ltl_and(reach_obj_1, ltl_eventually(reach_obj_2)))  # Reach object_1 then object_2
    c2 = ltl_until(ltl_not(reach_obj_2), reach_obj_1)  # object_2 should not be reached before object_1
    formula = ltl_and(c1, c2)  # Reach object_1 then object_2
    return formula


def example_8():
    """
    Reach object_1 then object_2 then object_3.
    """
    # explanation of the instruction:
    # The robot is expected to reach object_1, then object_2, and finally object_3.
    # object_1 to object_2: object_2 should not be reached before object_1 (`not until rules`)
    # object_2 to object_3: object_3 should not be reached before object_2 (`not until rules`)

    # create atomic propositions for objects
    reach_obj_1 = ap(reach, "object_1")  # reach(object_1)
    reach_obj_2 = ap(reach, "object_2")  # reach(object_2)
    reach_obj_3 = ap(reach, "object_3")  # reach(object_3)
    # describe the constraints in the instruction
    c1 = ltl_eventually(ltl_and(reach_obj_2, ltl_eventually(reach_obj_3)))  # Reach object_2 then object_3
    c2 = ltl_eventually(ltl_and(reach_obj_1, c1))  # Reach object_1 then object_2 then object_3
    c3 = ltl_until(ltl_not(reach_obj_2), reach_obj_1)  # object_2 should not be reached before object_1
    c4 = ltl_until(ltl_not(reach_obj_3), reach_obj_2)  # object_3 should not be reached before object_2
    c5 = ltl_and(c3, c4)  # Combine the constraints
    formula = ltl_and(c2, c5)  # Reach object_1 then object_2 then object_3
    return formula


def example_9():
    """
    Reach object3 after object_2, and object_2 after reaching to object_1.
    """
    # explanation of the instruction:
    # This example is the same as example_8, but the requirement is described in a different way.
    # The robot is expected to reach object_1, then object_2, and finally object_3.
    # object_1 to object_2: object_2 should not be reached before object_1 (`not until rules`)
    # object_2 to object_3: object_3 should not be reached before object_2 (`not until rules`)

    # create atomic propositions for objects
    reach_obj_1 = ap(reach, "object_1")  # reach(object_1)
    reach_obj_2 = ap(reach, "object_2")  # reach(object_2)
    reach_obj_3 = ap(reach, "object_3")  # reach(object_3)
    # describe the constraints in the instruction
    c1 = ltl_eventually(ltl_and(reach_obj_2, ltl_eventually(reach_obj_3)))  # Reach object_2 then object_3
    c2 = ltl_eventually(ltl_and(reach_obj_1, c1))  # Reach object_1 then object_2 then object_3
    c3 = ltl_until(ltl_not(reach_obj_2), reach_obj_1)  # object_2 should not be reached before object_1
    c4 = ltl_until(ltl_not(reach_obj_3), reach_obj_2)  # object_3 should not be reached before object_2
    c5 = ltl_and(c3, c4)  # Combine the constraints
    formula = ltl_and(c2, c5)  # Reach object_1 then object_2 then object_3
    return formula


def example_10():
    """
    Reach object_2, subsequently visit object_1 and come back to object_2.
    """
    # explanation of the instruction:
    # The robot is expected to reach object_2, then object_1, and then object_2 again.
    # object_2 then object_1: object_1 should not be reached before object_2 (`not until rules`)
    # object_1 then object_2: object_2 is reached again, no `not until rules`

    # create atomic propositions for objects
    reach_obj_1 = ap(reach, "object_1")  # reach(object_1)
    reach_obj_2 = ap(reach, "object_2")  # reach(object_2)
    # describe the constraints in the instruction
    c1 = ltl_eventually(ltl_and(reach_obj_1, ltl_eventually(reach_obj_2)))  # Reach object_1 then object_2
    c2 = ltl_eventually(ltl_and(reach_obj_2, c1))  # Reach object_2 then object_1 then object_2
    c3 = ltl_until(ltl_not(reach_obj_1), reach_obj_2)  # object_1 should not be reached before object_2
    formula = ltl_and(c2, c3)  # Reach object_2 then object_1 and come back to object_2
    return formula


def example_11():
    """
    Reach object_1, next go to object_2, and then object_1, last to object_3.
    """
    # explanation of the instruction:
    # The robot is expected to reach object_1, then object_2, then object_1, and then object_3
    # note that object_1 is reached twice.
    # object_1 then object_2: object_2 should not be reached before object_1 (`not until rules`)
    # object_2 then object_1: object_1 is reached again, no `not until rules`
    # object_1 then object_3: object_3 should not be reached before object_1 (`not until rules`)

    # create atomic propositions for objects
    reach_obj_1 = ap(reach, "object_1")  # reach(object_1)
    reach_obj_2 = ap(reach, "object_2")  # reach(object_2)
    reach_obj_3 = ap(reach, "object_3")  # reach(object_3)
    # describe the constraints in the instruction
    c1 = ltl_eventually(ltl_and(reach_obj_1, ltl_eventually(reach_obj_3)))  # Reach object_1 then object_3
    c2 = ltl_eventually(ltl_and(reach_obj_2, c1))  # Reach object_2 then object_1 then object_3
    c3 = ltl_eventually(ltl_and(reach_obj_1, c2))  # Reach object_1 then object_2 then object_1 then object_3
    c4 = ltl_until(ltl_not(reach_obj_2), reach_obj_1)  # object_2 should not be reached before object_1
    c5 = ltl_until(ltl_not(reach_obj_3), reach_obj_1)  # object_3 should not be reached before object_1
    c6 = ltl_and(c4, c5)  # Combine the constraints
    formula = ltl_and(c3, c6)  # Combine the constraints
    return formula


def example_12():
    """
    Reach object_2, object_3, and avoid object_1.
    """
    # explanation of the instruction:
    # The robot is expected to reach object_2 and object_3 at some point in the future. Additionally, the robot should avoid reaching object_1 at any time.

    # create atomic propositions for objects
    reach_obj_2 = ap(reach, "object_2")  # reach(object_2)
    reach_obj_3 = ap(reach, "object_3")  # reach(object_3)
    reach_obj_1 = ap(reach, "object_1")  # reach(object_1)
    # describe the constraints in the instruction
    c1 = ltl_eventually(reach_obj_2)  # Reach object_2 at some point
    c2 = ltl_eventually(reach_obj_3)  # Reach object_3 at some point
    c3 = ltl_always(ltl_not(reach_obj_1))  # Avoid object_1
    formula = ltl_and(c1, c2)  # Reach object_2 and object_3 at some point
    formula = ltl_and(formula, c3)  # Reach object_2 and object_3 and avoid object_1
    return formula


def example_13():
    """
    Go to object_1 and always avoid both object_2 and object_3.
    """
    # explanation of the instruction:
    # The robot is expected to go to object_1 at some point in the future. Additionally, the robot should always avoid reaching object_2 and object_3.

    # create atomic propositions for objects
    reach_obj_1 = ap(reach, "object_1")  # reach(object_1)
    reach_obj_2 = ap(reach, "object_2")  # reach(object_2)
    reach_obj_3 = ap(reach, "object_3")  # reach(object_3)
    # describe the constraints in the instruction
    c1 = ltl_eventually(reach_obj_1)  # Reach object_1 at some point
    c2 = ltl_not(reach_obj_2)  # Avoid object_2
    c3 = ltl_not(reach_obj_3)  # Avoid object_3
    c4 = ltl_always(ltl_and(c2, c3))  # Always avoid object_2 and object_3
    formula = ltl_and(c1, c4)  # Reach object_1 and always avoid object_2 and object_3
    return formula


def example_14():
    """
    Reach object_2 if you leave object_1.
    """
    # explanation of the instruction:
    # This instruction is an example of a conditional statement. If the robot leaves object_1, it should reach object_2 at some point.

    # create atomic propositions for objects
    reach_obj_1 = ap(reach, "object_1")  # reach(object_1)
    reach_obj_2 = ap(reach, "object_2")  # reach(object_2)
    # describe the constraints in the instruction
    c1 = ltl_not(reach_obj_1)  # Leave object_1
    c2 = ltl_eventually(reach_obj_2)  # Reach object_2 at some point
    formula = ltl_imply(c1, c2)  # Reach object_2 if you leave object_1
    return formula


def example_15():
    """
    If you have reached object_1 and object_2 and not reached object_3 or object_4, then you reach object_5 eventually.
    """
    # explanation of the instruction:
    # If the robot has reached object_1 and object_2 and has not reached object_3 or object_4, it should reach object_5 at some point in the future.

    # create atomic propositions for objects
    reach_obj_1 = ap(reach, "object_1")  # reach(object_1)
    reach_obj_2 = ap(reach, "object_2")  # reach(object_2)
    reach_obj_3 = ap(reach, "object_3")  # reach(object_3)
    reach_obj_4 = ap(reach, "object_4")  # reach(object_4)
    reach_obj_5 = ap(reach, "object_5")  # reach(object_5)
    # describe the constraints in the instruction
    c1 = ltl_and(reach_obj_1, reach_obj_2)  # Reach object_1 and object_2
    c2 = ltl_not(ltl_or(reach_obj_3, reach_obj_4))  # Don't reach object_3 or object_4
    c3 = ltl_and(c1, c2)  # Reach object_1 and object_2 and don't reach object_3 or object_4
    c4 = ltl_eventually(reach_obj_5)  # Reach object_5 at some point
    formula = ltl_imply(c3, c4)
    return formula


def example_16():
    """
    I think object_3 maybe on object_2 or object_5. Check it out.
    """
    # explanation of the instruction:
    # The robot is expected to reach object_2 or object_5, and reach object_3 simultaneously at some point in the future.

    # create atomic propositions for objects
    reach_obj_3 = ap(reach, "object_3")  # reach(object_3)
    reach_obj_2 = ap(reach, "object_2")  # reach(object_2)
    reach_obj_5 = ap(reach, "object_5")  # reach(object_5)
    # describe the constraints in the instruction
    c1 = ltl_or(reach_obj_2, reach_obj_5)  # object_2 or object_5
    c2 = ltl_and(reach_obj_3, c1)  # object_3 and object_2 or object_5
    formula = ltl_eventually(c2)  # object_3 and object_2 or object_5 at some point
    return formula


def example_17():
    """
    Try going to object_3 near object_5, otherwise go to object_2
    """
    # explanation of the instruction:
    # The robot is expected to reach object_3 and object_5 at some point in the future. Or reach object_2 at some point.

    # create atomic propositions for objects
    reach_obj_3 = ap(reach, "object_3")  # reach(object_3)
    reach_obj_5 = ap(reach, "object_5")  # reach(object_5)
    reach_obj_2 = ap(reach, "object_2")  # reach(object_2)
    # describe the constraints in the instruction
    c1 = ltl_and(reach_obj_3, reach_obj_5)  # object_3 and object_5
    c2 = ltl_eventually(c1)  # object_3 and object_5 at some point
    c3 = ltl_eventually(reach_obj_2)  # object_2 at some point
    formula = ltl_or(c2, c3)  # object_3 and object_5 at some point or object_2 at some point
    return formula


def example_18():
    """
    Reach object_3 every time you leave object_1. After object_2, you should visit object_4 at some point.
    """
    # explanation of the instruction:
    # If the robot leaves object_1, it should reach object_3. If the robot reaches object_2, it should reach object_4 at some point.

    # create atomic propositions for objects
    reach_obj_1 = ap(reach, "object_1")  # reach(object_1)
    reach_obj_2 = ap(reach, "object_2")  # reach(object_2)
    reach_obj_3 = ap(reach, "object_3")  # reach(object_3)
    reach_obj_4 = ap(reach, "object_4")  # reach(object_4)
    # describe the constraints in the instruction
    c1 = ltl_not(reach_obj_1)  # Leave object_1
    c2 = ltl_imply(c1, reach_obj_3)  # If the robot leaves object_1, it should reach object_3
    c3 = ltl_imply(reach_obj_2, ltl_eventually(reach_obj_4))  # If the robot reaches object_2, it should reach object_4 at some point
    formula = ltl_and(c2, c3)  # Reach object_3 every time you leave object_1 and visit object_4 after object_2
    return formula


def example_19():
    """
    The Robot is given a sequence of objects. They are expected to visit each object in order of appearance. Sequence: object_4, object_2, object_3, object_1, object_5.
    """
    # explanation of the instruction:
    # The robot is expected to reach object_4 then object_2 then object_3 then object_1 and then object_5.
    # object_4 to object_2: object_2 should not be reached before object_4 (`not until rules`)
    # object_2 to object_3: object_3 should not be reached before object_2 (`not until rules`)
    # object_3 to object_1: object_1 should not be reached before object_3 (`not until rules`)
    # object_1 to object_5: object_5 should not be reached before object_1 (`not until rules`)

    # create atomic propositions for objects
    reach_obj_4 = ap(reach, "object_4")  # reach(object_4)
    reach_obj_2 = ap(reach, "object_2")  # reach(object_2)
    reach_obj_3 = ap(reach, "object_3")  # reach(object_3)
    reach_obj_1 = ap(reach, "object_1")  # reach(object_1)
    reach_obj_5 = ap(reach, "object_5")  # reach(object_5)
    # describe the constraints in the instruction
    c1 = ltl_eventually(ltl_and(reach_obj_1, ltl_eventually(reach_obj_5)))  # Reach object_1 then object_5
    c2 = ltl_eventually(ltl_and(reach_obj_3, c1))  # Reach object_3 then object_1 then object_5
    c3 = ltl_eventually(ltl_and(reach_obj_2, c2))  # Reach object_2 then object_3 then object_1 then object_5
    c4 = ltl_eventually(ltl_and(reach_obj_4, c3))  # Reach object_4 then object_2 then object_3 then object_1 then object_5
    c5 = ltl_until(ltl_not(reach_obj_2), reach_obj_4)  # object_2 should not be reached before object_4
    c6 = ltl_until(ltl_not(reach_obj_3), reach_obj_2)  # object_3 should not be reached before object_2
    c7 = ltl_until(ltl_not(reach_obj_1), reach_obj_3)  # object_1 should not be reached before object_3
    c8 = ltl_until(ltl_not(reach_obj_5), reach_obj_1)  # object_5 should not be reached before object_1
    c9 = ltl_and(c5, c6)  # Combine the constraints
    c10 = ltl_and(c7, c8)  # Combine the constraints
    formula = ltl_and(c4, c9)  # Combine the constraints
    formula = ltl_and(formula, c10)  # Combine the constraints
    return formula


def example_20():
    """
    Get the chicken from object_6 then and heat it in object_3, go clean yourself in object_2 and check if chicken is cooked in object 3.
    """
    # explanation of the instruction:
    # The robot is expected to reach object_6, then object_3, then object_2, and then object_3 again.
    # object_6 then object_3: object_3 should not be reached before object_6 (`not until rules`)
    # object_3 then object_2: object_2 should not be reached before object_3 (`not until rules`)
    # object_2 then object_3: object_3 is reached again, no `not until rules`

    # create atomic propositions for objects
    # reach_chicken = ap(reach, "chicken")  # ignore chicken because it is not in the form object_x
    reach_obj_6 = ap(reach, "object_6")  # reach(object_6)
    reach_obj_3 = ap(reach, "object_3")  # reach(object_3)
    reach_obj_2 = ap(reach, "object_2")  # reach(object_2)
    # describe the constraints in the instruction
    c1 = ltl_eventually(ltl_and(reach_obj_2, ltl_eventually(reach_obj_3)))  # Reach object_2 then object_3
    c2 = ltl_eventually(ltl_and(reach_obj_3, c1))  # Reach object_3 then object_2 then object_3
    c3 = ltl_eventually(ltl_and(reach_obj_6, c2))  # Reach object_6 then object_3 then object_2 then object_3
    c4 = ltl_until(ltl_not(reach_obj_3), reach_obj_6)  # object_3 should not be reached before object_6
    c5 = ltl_until(ltl_not(reach_obj_2), reach_obj_3)  # object_2 should not be reached before object_3
    c6 = ltl_and(c4, c5)  # Combine the constraints
    formula = ltl_and(c3, c6)  # Combine the constraints
    return formula


# Now, please finish the following Python code for translating instruction to LTL formula.
# You can differentiate between Sequential and revisiting tasks by key phrases such as 'revisit', 'go back to', 'come back' 'return', 'again' etc.
# Ignore objects in the instruction that are not in the form object_x, where x is a number.
# The returned output should only contain the code that starts with `def` and ends with `return` statement.


def question():
    """
    {instruction}{previous_answer}{failure_reason}
    """
