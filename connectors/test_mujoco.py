import jax
import jax.numpy as jnp
import pytest

from framework.controllers.mpx.connectors.mujoco import MujocoMpxConnector


class _Locomotion:
    def pack_state(self, qpos, qvel, foot):
        self.packed = (qpos, qvel, foot)
        return "x0"

    def run(self, carry, x0, command, contact, terrain=None):
        self.ran = (carry, x0, command, contact, terrain)
        return carry, "tau"

    def reset(self, carry, qpos, qvel, foot):
        self.reset_args = (carry, qpos, qvel, foot)
        return "reset-carry"

    def make_data(self):
        return "carry"


class _Balance:
    def __init__(self):
        self.initial_state = jnp.zeros(6)
        self.qpos_slice = slice(0, 2)
        self.qvel_slice = slice(2, 4)
        self.foot_slice = slice(4, 6)

    def run(
        self,
        carry,
        x0,
        command,
        contact,
        base_quat_ref,
        use_base_quat_ref,
        foot_ref_anchor,
        use_foot_ref_anchor,
    ):
        self.ran = (
            carry,
            x0,
            command,
            contact,
            base_quat_ref,
            use_base_quat_ref,
            foot_ref_anchor,
            use_foot_ref_anchor,
        )
        return carry, "tau"

    def reset(self, carry, qpos, qvel, foot):
        return carry

    def make_data(self):
        return "carry"


def test_locomotion_step_packs_state_and_passes_terrain():
    wrapper = _Locomotion()
    controller = MujocoMpxConnector(wrapper, kind="locomotion")
    with jax.disable_jit():
        carry, tau = controller.step("c", 1, 2, 3, 4, 5, 6)
    assert wrapper.packed == (1, 2, 3)
    assert wrapper.ran == ("c", "x0", 4, 5, 6)
    assert (carry, tau) == ("c", "tau")


def test_locomotion_reset_forwards_the_snapshot():
    wrapper = _Locomotion()
    controller = MujocoMpxConnector(wrapper, kind="locomotion")
    with jax.disable_jit():
        assert controller.reset("c", 1, 2, 3) == "reset-carry"
    assert wrapper.reset_args == ("c", 1, 2, 3)


def test_balance_step_writes_slices_and_forwards_references():
    wrapper = _Balance()
    controller = MujocoMpxConnector(wrapper, kind="balance")
    qpos = jnp.array([1.0, 2.0])
    qvel = jnp.array([3.0, 4.0])
    foot = jnp.array([5.0, 6.0])
    with jax.disable_jit():
        carry, tau = controller.step("c", qpos, qvel, foot, "cmd", "contact", "bq", "ub", "fa", "uf")
    assert carry == "c" and tau == "tau"
    state = wrapper.ran[1]
    assert jnp.allclose(state[wrapper.qpos_slice], qpos)
    assert jnp.allclose(state[wrapper.qvel_slice], qvel)
    assert jnp.allclose(state[wrapper.foot_slice], foot)
    assert wrapper.ran[2:] == ("cmd", "contact", "bq", "ub", "fa", "uf")


def test_default_controller_is_mpc():
    import inspect

    from framework.controllers import DEFAULT_CONTROLLER, make_controller

    assert DEFAULT_CONTROLLER == "mpc"
    assert inspect.signature(make_controller).parameters["name"].default == "mpc"
    with pytest.raises(ValueError):
        make_controller(object(), "locomotion", name="isaaclab")


def test_balance_feet_must_be_three_or_four():
    with pytest.raises(ValueError):
        MujocoMpxConnector.for_balance(object(), feet=2)


def test_shim_make_locomotion_mpc_is_the_moved_function():
    from framework.controllers.mpx.locomotion.inverse import make_locomotion_mpc as moved
    from framework.utils.quad_utils_locomotion.mpc_wrapper_inverse import make_locomotion_mpc as shim

    assert shim is moved
