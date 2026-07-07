import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skill_learning_mpe.env import DEFAULT_ENV_IDS, make_env


def _env():
    return make_env(DEFAULT_ENV_IDS["area_denial"])


def _scripted_env(mode="direct", *, num_good=3, num_adv=3, start="static"):
    return make_env(
        f"simple_areadenial_{num_good}v{num_adv}_0obs_discrete_0s_{start}_ctrl-{mode}_v0"
    )


def _noop_actions(env):
    return {agent: jnp.asarray(0, dtype=jnp.int32) for agent in env.agents}


def _defender_noop_actions(env):
    return {agent: jnp.asarray(0, dtype=jnp.int32) for agent in env.good_agents}


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
    assert not env.scripted_attackers_enabled
    assert env.controlled_agents == env.agents
    assert env.scripted_agents == []
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


def test_scripted_direct_controller_accepts_defender_only_actions():
    env = _scripted_env("direct")
    assert env.scripted_attackers_enabled
    assert env.attacker_controller_mode == "direct"
    assert env.controlled_agents == env.good_agents
    assert env.scripted_agents == env.adversaries
    key = jax.random.PRNGKey(7)
    _, state = env.reset(key)
    state = _state_with(
        env,
        state,
        (
            (0.0, 6.0),
            (0.0, -6.0),
            (-6.0, 0.0),
            (10.0, 0.0),
            (10.0, 3.0),
            (10.0, -3.0),
        ),
    )
    before = jnp.linalg.norm(state.p_pos[env.num_good_agents :] - state.area_pos, axis=1)

    _, mid_state, _, _, _ = env.step_env(key, state, _defender_noop_actions(env))
    _, next_state, _, _, info = env.step_env(
        key, mid_state, _defender_noop_actions(env)
    )

    after = jnp.linalg.norm(next_state.p_pos[env.num_good_agents :] - state.area_pos, axis=1)
    assert jnp.all(after < before)
    assert bool(info["scripted_attackers_enabled"])


def test_scripted_controller_ignores_provided_adversary_actions():
    env = _scripted_env("direct")
    key = jax.random.PRNGKey(8)
    _, state = env.reset(key)
    state = _state_with(
        env,
        state,
        (
            (0.0, 6.0),
            (0.0, -6.0),
            (-6.0, 0.0),
            (10.0, 0.0),
            (10.0, 3.0),
            (10.0, -3.0),
        ),
    )
    defender_only = _defender_noop_actions(env)
    full_actions = {
        **defender_only,
        **{agent: jnp.asarray(2, dtype=jnp.int32) for agent in env.adversaries},
    }

    _, defender_only_state, _, _, _ = env.step_env(key, state, defender_only)
    _, full_action_state, _, _, _ = env.step_env(key, state, full_actions)

    np.testing.assert_allclose(defender_only_state.p_pos, full_action_state.p_pos)
    np.testing.assert_allclose(defender_only_state.p_vel, full_action_state.p_vel)


def test_scripted_circle_and_split_modes_are_finite():
    key = jax.random.PRNGKey(9)
    for mode in ("circle", "split"):
        env = _scripted_env(mode)
        obs, state = env.reset(key)
        obs, state, rewards, dones, info = env.step(
            key, state, _defender_noop_actions(env)
        )
        assert obs["agent_0"].shape == env.observation_space("agent_0").shape
        assert np.isfinite(np.asarray(state.p_pos)).all()
        assert np.isfinite(np.asarray(rewards["agent_0"]))
        assert "__all__" in dones
        assert bool(info["scripted_attackers_enabled"])


def test_scripted_controller_scales_to_non_3v3_under_vmap():
    env = _scripted_env("split", num_good=2, num_adv=4)
    key = jax.random.PRNGKey(10)
    keys = jax.random.split(key, 2)
    obs, state = jax.vmap(env.reset)(keys)
    assert obs["agent_0"].shape == (2,) + env.observation_space("agent_0").shape
    actions = {agent: jnp.zeros((2,), dtype=jnp.int32) for agent in env.good_agents}
    step_keys = jax.random.split(jax.random.PRNGKey(11), 2)

    next_obs, next_state, rewards, dones, info = jax.vmap(env.step)(
        step_keys, state, actions
    )

    assert next_obs["agent_0"].shape == (2,) + env.observation_space("agent_0").shape
    assert next_state.p_pos.shape == (2, len(env.agents), 2)
    assert rewards["agent_0"].shape == (2,)
    assert rewards["adversary_3"].shape == (2,)
    assert dones["__all__"].shape == (2,)
    assert info["subtask_obs"]["adversary_3"].shape == (2, 10)


def test_scripted_controller_rejects_unknown_mode():
    with pytest.raises(ValueError, match="attacker controller mode"):
        make_env("simple_areadenial_3v3_0obs_discrete_0s_static_ctrl-weird_v0")


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
