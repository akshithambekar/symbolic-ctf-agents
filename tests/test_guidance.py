import pytest
from maritime_nsrl.guidance import Guidance
from maritime_nsrl.logic import validate


def test_discounted_potential_terms_telescope_at_finite_deadline():
    g = Guidance(validate("F(captured)"), gamma=0.9, milestone_reward=0, violation_penalty=0)
    potentials = [0.2, 0.7, 1.0, 0.0]
    discounted = sum(
        g.gamma**i * g.reward(a, b, 0, False)
        for i, (a, b) in enumerate(zip(potentials, potentials[1:]))
    )
    assert discounted == pytest.approx(-g.scale * potentials[0])
