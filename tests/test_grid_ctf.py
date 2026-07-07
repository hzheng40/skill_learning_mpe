import jax
import jax.numpy as jnp

from skill_learning_mpe.env import DEFAULT_ENV_IDS, make_env


def _grid_env():
    return make_env(DEFAULT_ENV_IDS["grid_ctf_buttons"])


def _noop_actions(env):
    return {agent: jnp.asarray(0, dtype=jnp.int32) for agent in env.agents}


def _actions(env, **overrides):
    actions = _noop_actions(env)
    for agent, action in overrides.items():
        actions[agent] = jnp.asarray(action, dtype=jnp.int32)
    return actions


def _state_with_positions(env, state, positions):
    p_pos = jnp.asarray(positions, dtype=jnp.float32)
    return state.replace(
        p_pos=p_pos,
        p_vel=jnp.zeros_like(p_pos),
        prev_p_pos=p_pos,
        flag_moving=jnp.zeros((2,), dtype=bool),
        flag_carrier=jnp.zeros((2,), dtype=jnp.int32),
        flag_p_pos=jnp.stack((state.agent_zone, state.adversary_zone)).astype(jnp.float32),
        flag_p_vel=jnp.zeros((2, 2), dtype=jnp.float32),
    )


def test_grid_ctf_reset_step_jit_and_vmap_static_shapes():
    env = _grid_env()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    assert obs["agent_0"].shape == (39,)

    actions = _noop_actions(env)
    obs, state, reward, done, info = env.step(key, state, actions)
    assert obs["agent_0"].shape == (39,)
    assert reward["agent_0"].shape == ()
    assert "__all__" in done
    assert info["subtask_obs"]["agent_0"].shape == (10,)

    keys = jax.random.split(key, 2)
    batched_obs, batched_state = jax.vmap(env.reset)(keys)
    assert batched_obs["agent_0"].shape == (2, 39)
    batched_actions = {
        agent: jnp.zeros((2,), dtype=jnp.int32) for agent in env.agents
    }
    step_keys = jax.random.split(jax.random.PRNGKey(1), 2)
    next_obs, next_state, rewards, dones, step_info = jax.vmap(env.step)(
        step_keys, batched_state, batched_actions
    )
    assert next_obs["agent_0"].shape == (2, 39)
    assert next_state.p_pos.shape == (2, len(env.agents), 2)
    assert rewards["agent_0"].shape == (2,)
    assert dones["__all__"].shape == (2,)
    assert step_info["subtask_obs"]["agent_0"].shape == (2, 10)


def test_grid_ctf_obstacle_blocks_movement_and_sets_collision_event():
    env = _grid_env()
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    state = _state_with_positions(
        env,
        state,
        (
            (6, 4),
            (1, 3),
            (1, 5),
            (14, 6),
            (14, 7),
            (14, 8),
        ),
    )

    _, next_state, _, _, info = env.step_env(
        key, state, _actions(env, agent_0=2)
    )

    assert jnp.array_equal(next_state.p_pos[0], state.p_pos[0])
    assert info["subtask_obs"]["agent_0"][7] == 1.0


def test_grid_ctf_same_cell_conflicts_and_direct_swaps_are_blocked():
    env = _grid_env()
    key = jax.random.PRNGKey(1)
    _, state = env.reset(key)
    conflict_state = _state_with_positions(
        env,
        state,
        (
            (4, 4),
            (6, 4),
            (1, 5),
            (14, 6),
            (14, 7),
            (14, 8),
        ),
    )
    _, next_state, _, _, info = env.step_env(
        key, conflict_state, _actions(env, agent_0=2, agent_1=1)
    )
    assert jnp.array_equal(next_state.p_pos[:2], conflict_state.p_pos[:2])
    assert info["subtask_obs"]["agent_0"][8] == 1.0
    assert info["subtask_obs"]["agent_1"][8] == 1.0

    swap_state = _state_with_positions(
        env,
        state,
        (
            (4, 4),
            (5, 4),
            (1, 5),
            (14, 6),
            (14, 7),
            (14, 8),
        ),
    )
    _, next_state, _, _, info = env.step_env(
        key, swap_state, _actions(env, agent_0=2, agent_1=1)
    )
    assert jnp.array_equal(next_state.p_pos[:2], swap_state.p_pos[:2])
    assert info["subtask_obs"]["agent_0"][8] == 1.0
    assert info["subtask_obs"]["agent_1"][8] == 1.0


def test_grid_ctf_noop_on_button_unlocks_teammate_zone_entry_only():
    env = _grid_env()
    key = jax.random.PRNGKey(2)
    _, state = env.reset(key)
    positions = (
        (2, 7),
        (10, 4),
        (1, 5),
        (14, 6),
        (14, 7),
        (14, 8),
    )
    state = _state_with_positions(env, state, positions)

    _, next_state, _, _, info = env.step_env(
        key, state, _actions(env, agent_0=0, agent_1=2)
    )
    assert jnp.array_equal(next_state.p_pos[1], jnp.asarray((11.0, 4.0)))
    assert info["subtask_obs"]["agent_0"][0] == 1.0
    assert info["subtask_obs"]["agent_1"][1] == 1.0

    off_button_state = _state_with_positions(
        env,
        state,
        (
            (2, 6),
            (10, 4),
            (1, 5),
            (14, 6),
            (14, 7),
            (14, 8),
        ),
    )
    _, blocked_state, _, _, info = env.step_env(
        key, off_button_state, _actions(env, agent_0=0, agent_1=2)
    )
    assert jnp.array_equal(blocked_state.p_pos[1], off_button_state.p_pos[1])
    assert info["subtask_obs"]["agent_0"][0] == 0.0
    assert info["subtask_obs"]["agent_1"][1] == 0.0


def test_grid_ctf_flag_pickup_and_capture_update_state_reward_and_events():
    env = _grid_env()
    key = jax.random.PRNGKey(3)
    _, state = env.reset(key)
    state = _state_with_positions(
        env,
        state,
        (
            (2, 7),
            (10, 4),
            (1, 5),
            (14, 6),
            (14, 7),
            (14, 8),
        ),
    )

    _, carrying_state, rewards, _, info = env.step_env(
        key, state, _actions(env, agent_0=0, agent_1=2)
    )
    assert carrying_state.flag_moving[1]
    assert carrying_state.flag_carrier[0] == 1
    assert info["subtask_obs"]["agent_1"][5] == 1.0
    assert rewards["agent_1"] == 0.0

    p_pos = carrying_state.p_pos.at[1].set(carrying_state.agent_zone)
    capture_state = carrying_state.replace(
        p_pos=p_pos,
        prev_p_pos=p_pos,
        p_vel=jnp.zeros_like(p_pos),
    )
    _, final_state, rewards, _, info = env.step_env(
        key, capture_state, _noop_actions(env)
    )

    assert final_state.flag_moving[1] == 0
    assert rewards["agent_0"] == 1.0
    assert rewards["agent_1"] == 1.0
    assert rewards["agent_2"] == 1.0
    assert info["agent_dropped"] == 1
    assert info["subtask_obs"]["agent_1"][6] == 1.0
