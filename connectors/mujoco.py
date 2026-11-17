"""MuJoCo snapshot in, joint torques out."""

import jax


class MujocoMpxConnector:
    """Torque seam for the MPX control law.

    ``step`` is a ``jax.jit`` closure. Locomotion packs with ``pack_state`` and
    passes ``terrain``. Balance writes ``qpos`` / ``qvel`` / ``foot`` into
    ``initial_state`` and forwards the four pose references.

    Attributes other than ``step``, ``reset``, ``make_carry``, and ``wrapper``
    are read from the wrapped MPC object. Multi-env loops still vmap
    ``run`` on an already packed state.
    """

    def __init__(self, wrapper, kind):
        if kind not in ("locomotion", "balance"):
            raise ValueError(f"kind must be locomotion or balance, got {kind!r}")
        self.wrapper = wrapper
        self.kind = kind
        self.step = _jit_step(wrapper, kind)
        self.reset = jax.jit(wrapper.reset)

    def make_carry(self):
        return self.wrapper.make_data()

    def __getattr__(self, name):
        return getattr(self.wrapper, name)

    @classmethod
    def for_locomotion(cls, config, limited_memory=True):
        from framework.controllers.mpx.locomotion.inverse import make_locomotion_mpc

        return cls(make_locomotion_mpc(config, limited_memory=limited_memory), kind="locomotion")

    @classmethod
    def for_balance(cls, config, feet, limited_memory=True):
        if feet == 3:
            from framework.controllers.mpx.balance.three_foot import MPCWrapper
        elif feet == 4:
            from framework.controllers.mpx.balance.four_foot import MPCWrapper
        else:
            raise ValueError(f"feet must be 3 or 4, got {feet}")
        return cls(MPCWrapper(config, limited_memory=limited_memory), kind="balance")


def _jit_step(mpc, kind):
    if kind == "locomotion":

        @jax.jit
        def step(carry, qpos, qvel, foot, command, contact, terrain=None):
            x0 = mpc.pack_state(qpos, qvel, foot)
            return mpc.run(carry, x0, command, contact, terrain=terrain)

        return step

    @jax.jit
    def step(
        carry,
        qpos,
        qvel,
        foot,
        command,
        contact,
        base_quat_ref,
        use_base_quat_ref,
        foot_ref_anchor,
        use_foot_ref_anchor,
    ):
        x0 = (
            mpc.initial_state.at[mpc.qpos_slice].set(qpos)
            .at[mpc.qvel_slice].set(qvel)
            .at[mpc.foot_slice].set(foot)
        )
        return mpc.run(
            carry,
            x0,
            command,
            contact,
            base_quat_ref,
            use_base_quat_ref,
            foot_ref_anchor,
            use_foot_ref_anchor,
        )

    return step
