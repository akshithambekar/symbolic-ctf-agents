"""Transactional one-step action gate for the exact local simulator model.

Only a predicted safe simulator successor is committed. If every tested joint
action is unsafe, the episode fails closed without advancing the game. This is
not a robust real-world invariant or an adversarial winning-region shield.
"""

import copy
import random
import numpy as np
from .temporal import compile_ltlf


class PredictiveShield:
    def __init__(self, formula="G(in_bounds)"):
        self.automaton = compile_ltlf(formula)
        # A temporal gate can reject prefixes with no accepting continuation;
        # it cannot enforce eventuality by a deadline without a horizon model.
        self.state = 0
        self.interventions = self.deadlocks = self.predictions = 0

    def initialize(self, labels):
        self.state = self.automaton.advance(0, labels)
        if self.state not in self.automaton.live:
            raise ValueError("Initial state violates shield contract")

    def transition(self, game, candidates, ground):
        np_state, py_state = np.random.get_state(), random.getstate()
        for index, actions in enumerate(candidates):
            np.random.set_state(np_state)
            random.setstate(py_state)
            preview = copy.deepcopy(game)
            result = preview.step(actions)
            self.predictions += 1
            nxt = self.automaton.advance(self.state, ground(preview))
            if nxt in self.automaton.live:
                self.state = nxt
                self.interventions += int(index > 0)
                # Commit exactly the checked successor and its RNG effects.
                return preview, result, index
        np.random.set_state(np_state)
        random.setstate(py_state)
        self.deadlocks += 1
        return None, None, None
