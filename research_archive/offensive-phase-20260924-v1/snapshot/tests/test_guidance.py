import pytest
from maritime_nsrl.guidance import Guidance
from maritime_nsrl.logic import validate


def test_discounted_potential_terms_telescope_at_finite_deadline():
    g = Guidance(validate("F(captured)"), gamma=.9, milestone_reward=0, violation_penalty=0)
    potentials = [.2, .7, 1., 0.]
    discounted = sum(g.gamma**i * g.reward(a,b,0,False)
                     for i,(a,b) in enumerate(zip(potentials, potentials[1:])))
    assert discounted == pytest.approx(-g.scale * potentials[0])
