"""
consumer_oracle.py  —  the eat/drink oracle for Proto 06.

Perfect execution of the consume model: given the current deficit, return the fill
fraction that lands the drive at `fill_target`, using world's true dynamics. Eat and
drink are the same policy on different drives, so this is one class instantiated twice.

`fill_target` is the sweep knob you asked for — it's *policy*, not physics, so it lives
here, not in world.py. Sweep it (e.g. 1.2 .. 2.2) to find where the overfill/ buffer
trade-off pays best; 1.7 reproduces the monolith's oracle. The asymmetric comfort
surface (OVER_W << UNDER_W) is why the optimum sits above IDEAL: a little overfill is
cheap insurance against decay before the next consume, being caught under is dear.
"""

from __future__ import annotations
from model_modules.contract_v1 import Consumer, ConsumerObs
import world_v1 as world


class OracleConsumer(Consumer):
    def __init__(self, resource: str, fill_target: float = 1.7):
        assert resource in ("water", "food")
        self.resource = resource
        self.fill_target = fill_target

    def consume(self, obs: ConsumerObs) -> float:
        # brightness is in ConsumerObs for the *learned* consumer to exploit later;
        # the oracle ignores it — it doesn't enter the consume dynamics, only decay.
        if self.resource == "water":
            return world.drink_frac_for(obs.h, obs.s, self.fill_target)   # respects the (0.8+0.3 s) coupling
        return world.eat_frac_for(obs.s, self.fill_target)


# convenience: the two slots the composer wires.
def make_eat(fill_target: float = 1.7) -> OracleConsumer:
    return OracleConsumer("food", fill_target)

def make_drink(fill_target: float = 1.7) -> OracleConsumer:
    return OracleConsumer("water", fill_target)