"""Controller slot shared by MuJoCo connectors.

An IsaacLab policy implements the same calls. This package does not ship one.
"""

from typing import Protocol


class TorqueController(Protocol):
    def make_carry(self): ...

    def step(
        self,
        carry,
        qpos,
        qvel,
        foot,
        command,
        contact,
        terrain=None,
        *,
        base_quat_ref=None,
        use_base_quat_ref=None,
        foot_ref_anchor=None,
        use_foot_ref_anchor=None,
    ): ...

    def reset(self, carry, qpos, qvel, foot): ...
