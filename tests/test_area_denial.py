import jax
import jax.numpy as jnp
import numpy as np

from skill_learning_mpe.env import DEFAULT_ENV_IDS, make_env


def _env():
    return make_env(DEFAULT_ENV_IDS["area_denial"])


def _noop_actions(env):
    return {agent: jnp.asarray(0, dtype=jnp.int32) for agent in env.agents}


def _state_with(env, state, positions, *, progress=0.0, step=None):
    p_pos = jnp.asarray(positions, dtype=jnp.float32)
    return state.replace(
        p_pos=p_pos,
        p_vel=jnp.zeros_like(p_pos),
        prev_p_pos=p_pos,
        control_progress=jnp.asarray(progress, dtype=jnp.float32),
        prev_control_progress=jnp.asarray(progress, dtype=jnp.float32),
        step=state.step if step is None else jnp.asarray(step, dtype=jnp.int32),
    )


def _attacker_inside_positions():
    return (
        (-10.0, 0.0),
        (-10.0, 3.0),
        (-10.0, -3.0),
        (3.0, 0.0),
        (-1.5, 2.6),
        (-1.5, -2.6),
    )


def _defender_inside_positions():
    return (
        (3.0, 0.0),
        (-1.5, 2.6),
        (-1.5, -2.6),
        (10.0, 0.0),
        (10.0, 3.0),
        (10.0, -3.0),
    )


def test_area_denial_reset_step_jit_and_vmap_static_shapes():
    env = _env()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    assert obs["agent_0"].shape == (28,)

    obs, state, reward, done, info = env.step(key, state, _noop_actions(env))
    assert obs["agent_0"].shape == (28,)
    assert reward["agent_0"].shape == ()
    assert reward["adversary_0"].shape == ()
    assert "__all__" in done
    assert info["subtask_obs"]["agent_0"].shape == (10,)
    assert np.isfinite(np.asarray(reward["agent_0"]))

    keys = jax.random.split(jax.random.PRNGKey(1), 2)
    batched_obs, batched_state = jax.vmap(env.reset)(keys)
    assert batched_obs["agent_0"].shape == (2, 28)
    actions = {agent: jnp.zeros((2,), dtype=jnp.int32) for agent in env.agents}
    step_keys = jax.random.split(jax.random.PRNGKey(2), 2)
    next_obs, next_state, rewards, dones, info = jax.vmap(env.step)(
        step_keys, batched_state, actions
    )
    assert next_obs["agent_0"].shape == (2, 28)
    assert next_state.p_pos.shape == (2, len(env.agents), 2)
    assert rewards["agent_0"].shape == (2,)
    assert dones["__all__"].shape == (2,)
    assert info["subtask_obs"]["agent_0"].shape == (2, 10)


def test_area_denial_attacker_inside_increases_progress_and_rewards_attackers():
    env = _env()
    key = jax.random.PRNGKey(3)
    _, state = env.reset(key)
    state = _state_with(env, state, _attacker_inside_positions())

    _, next_state, rewards, done, info = env.step_env(key, state, _noop_actions(env))

    assert next_state.control_progress > state.control_progress
    assert rewards["adversary_0"] > 0.0
    assert rewards["agent_0"] < 0.0
    assert not bool(done["__all__"])
    assert info["subtask_obs"]["adversary_0"][1] == 1.0


def test_area_denial_defender_only_area_presence_decreases_progress():
    env = _env()
    key = jax.random.PRNGKey(4)
    _, state = env.reset(key)
    state = _state_with(env, state, _defender_inside_positions(), progress=0.5)

    _, next_state, rewards, _, info = env.step_env(key, state, _noop_actions(env))

    assert next_state.control_progress < state.control_progress
    assert rewards["agent_0"] > 0.0
    assert rewards["adversary_0"] < 0.0
    assert info["subtask_obs"]["agent_0"][5] == 1.0
    assert info["subtask_obs"]["adversary_0"][6] == 1.0


def test_area_denial_capture_and_timeout_terminal_rewards():
    env = _env()
    key = jax.random.PRNGKey(5)
    _, state = env.reset(key)
    capture_state = _state_with(env, state, _attacker_inside_positions(), progress=0.99)

    _, next_state, rewards, done, info = env.step_env(
        key, capture_state, _noop_actions(env)
    )
    assert next_state.control_progress == 1.0
    assert bool(done["__all__"])
    assert info["attacker_captured"]
    assert rewards["adversary_0"] > 0.0
    assert rewards["agent_0"] < 0.0

    timeout_state = _state_with(
        env,
        state,
        _defender_inside_positions(),
        progress=0.0,
        step=env.max_steps - 1,
    )
    _, _, rewards, done, info = env.step_env(key, timeout_state, _noop_actions(env))
    assert bool(done["__all__"])
    assert info["defender_timed_out"]
    assert rewards["agent_0"] > 0.0
    assert rewards["adversary_0"] < 0.0


def test_area_denial_enter_exit_events():
    env = _env()
    key = jax.random.PRNGKey(6)
    _, state = env.reset(key)
    current = jnp.asarray(
        (
            (5.0, 0.0),
            (-1.5, 2.6),
            (-1.5, -2.6),
            (3.0, 0.0),
            (10.0, 3.0),
            (10.0, -3.0),
        ),
        dtype=jnp.float32,
    )
    previous = current.at[0].set(jnp.asarray((3.0, 0.0))).at[3].set(jnp.asarray((5.0, 0.0)))
    state = state.replace(p_pos=current, prev_p_pos=previous)

    events = env.subtask_obs_fn(state, previous, jnp.asarray(0.04))

    assert events["agent_0"][3] == 1.0
    assert events["adversary_0"][2] == 1.0
    assert events["agent_0"][4] == 1.0
    assert events["adversary_0"][5] == 1.0
