import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from functools import partial
from typing import Dict, List, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from ..multi_agent_env import MultiAgentEnv
from ..spaces import Box, Discrete
from .default_params import CTF_MAX_STEPS, DISCRETE_ACT


@struct.dataclass
class State:
    """Grid CTF state with CTF-compatible continuous-style fields."""

    p_pos: chex.Array
    p_vel: chex.Array
    flag_moving: chex.Array
    flag_carrier: chex.Array
    flag_p_pos: chex.Array
    flag_p_vel: chex.Array
    done: chex.Array
    step: int
    agent_zone: chex.Array
    adversary_zone: chex.Array
    obs_pos: chex.Array
    agent_names: List[str] = struct.field(pytree_node=False)
    adversary_names: List[str] = struct.field(pytree_node=False)
    prev_p_pos: chex.Array
    button_pos: chex.Array
    grid_size: chex.Array


class SimpleGridCTFButtons(MultiAgentEnv):
    """Discrete grid CTF with button-gated zone entry.

    This environment is semantic-equivalent to SimpleCTFButtons, not a physics
    discretization. It keeps the CTF state field names so downstream object,
    factor, event, and wrapper code can adapt with minimal branching.
    """

    def __init__(
        self,
        *,
        num_good_agents: int = 3,
        num_adversaries: int = 3,
        num_obstacles: int = 1,
        width: int = 15,
        height: int = 9,
        action_type: str = DISCRETE_ACT,
        random_start: bool = True,
        zero_sum: bool = False,
        zone_radius: int = 1,
        agent_size: float = 0.5,
        obstacle_size: float = 0.5,
        button_radius: float = 0.5,
        vel_eps: float = 0.1,
        max_steps: int = CTF_MAX_STEPS,
        **_: object,
    ):
        if action_type != DISCRETE_ACT:
            raise ValueError("SimpleGridCTFButtons only supports discrete actions")
        if num_good_agents != 3 or num_adversaries != 3:
            raise ValueError("SimpleGridCTFButtons currently supports the 3v3 layout")
        if num_obstacles != 1:
            raise ValueError("SimpleGridCTFButtons currently supports one obstacle")
        super().__init__(num_agents=num_good_agents + num_adversaries)
        self.num_good_agents = num_good_agents
        self.num_adversaries = num_adversaries
        self.num_actors = self.num_agents
        self.num_obstacles = num_obstacles
        self.width = int(width)
        self.height = int(height)
        self.zone_radius_cells = int(zone_radius)
        self.zone_size = float(zone_radius) + 0.75
        self.agent_size = float(agent_size)
        self.obstacle_size = float(obstacle_size)
        self.button_radius = float(button_radius)
        self.vel_eps = float(vel_eps)
        self.max_steps = int(max_steps)
        self.zero_sum = bool(zero_sum)
        self.random_start = bool(random_start)
        self.action_type = action_type
        self.include_velocities = True

        self.good_agents = [f"agent_{i}" for i in range(num_good_agents)]
        self.adversaries = [f"adversary_{i}" for i in range(num_adversaries)]
        self.agents = self.good_agents + self.adversaries
        self.agent_range = jnp.arange(self.num_agents)
        self.a_to_i = {agent: idx for idx, agent in enumerate(self.agents)}
        self.action_spaces = {agent: Discrete(5) for agent in self.agents}
        obs_dim = self._obs_dim()
        self.observation_spaces = {
            agent: Box(-jnp.inf, jnp.inf, (obs_dim,)) for agent in self.agents
        }

    @property
    def name(self) -> str:
        return "SimpleGridCTFButtons"

    @property
    def agent_classes(self) -> dict:
        return {
            "agent": self.good_agents,
            "adversary": self.adversaries,
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        agent_zone = self._agent_zone()
        adversary_zone = self._adversary_zone()
        obs_pos = self._obstacle_positions()
        button_pos = self._button_positions()
        base_pos = self._base_agent_positions()
        key_offsets, _ = jax.random.split(key)
        offsets = jax.random.randint(
            key_offsets, (self.num_agents,), minval=0, maxval=3, dtype=jnp.int32
        ) - 1
        random_offsets = jnp.stack(
            (
                jnp.zeros((self.num_agents,), dtype=jnp.int32),
                offsets,
            ),
            axis=1,
        )
        p_pos = jnp.where(self.random_start, base_pos + random_offsets, base_pos)
        p_pos = self._clip_grid(p_pos)
        p_pos = self._avoid_static_cells(p_pos, base_pos)
        p_pos_f = p_pos.astype(jnp.float32)
        state = State(
            p_pos=p_pos_f,
            p_vel=jnp.zeros_like(p_pos_f),
            flag_moving=jnp.zeros((2,), dtype=bool),
            flag_carrier=jnp.zeros((2,), dtype=jnp.int32),
            flag_p_pos=jnp.stack((agent_zone, adversary_zone)).astype(jnp.float32),
            flag_p_vel=jnp.zeros((2, 2), dtype=jnp.float32),
            done=jnp.zeros((self.num_agents,), dtype=bool),
            step=jnp.asarray(0, dtype=jnp.int32),
            agent_zone=agent_zone.astype(jnp.float32),
            adversary_zone=adversary_zone.astype(jnp.float32),
            obs_pos=obs_pos.astype(jnp.float32),
            agent_names=self.good_agents,
            adversary_names=self.adversaries,
            prev_p_pos=p_pos_f,
            button_pos=button_pos.astype(jnp.float32),
            grid_size=jnp.asarray((self.width, self.height), dtype=jnp.int32),
        )
        return self.obs_fn_o(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]):
        del key
        old_pos = state.p_pos.astype(jnp.int32)
        action_vec = jnp.asarray([actions[agent] for agent in self.agents], dtype=jnp.int32)
        pressed = self._pressed_mask(state, action_vec)

        proposed = self._clip_grid(old_pos + self._action_deltas(action_vec))
        obstacle_hit = self._obstacle_hits(proposed, state.obs_pos.astype(jnp.int32))
        proposed = jnp.where(obstacle_hit[:, None], old_pos, proposed)
        proposed = self._block_gated_entries(old_pos, proposed, pressed)
        new_pos_i, collision_masks = self._resolve_collisions(old_pos, proposed, obstacle_hit)
        new_pos = new_pos_i.astype(jnp.float32)
        new_vel = new_pos - state.p_pos

        moving_before = state.flag_moving
        carrier_before = state.flag_carrier
        good_carrier_onehot = (
            moving_before[1]
            & (jnp.arange(self.num_good_agents, dtype=jnp.int32) == carrier_before[0])
        )
        adv_carrier_onehot = (
            moving_before[0]
            & (jnp.arange(self.num_adversaries, dtype=jnp.int32) == carrier_before[1])
        )

        good_capture = moving_before[1] & jnp.any(
            good_carrier_onehot & self._in_agent_zone(new_pos_i[: self.num_good_agents])
        )
        adv_capture = moving_before[0] & jnp.any(
            adv_carrier_onehot & self._in_adversary_zone(new_pos_i[self.num_good_agents :])
        )

        moving_after_capture = jnp.asarray(
            (
                moving_before[0] & ~adv_capture,
                moving_before[1] & ~good_capture,
            )
        )
        good_pickup_candidates = self._in_adversary_zone(new_pos_i[: self.num_good_agents])
        adv_pickup_candidates = self._in_agent_zone(new_pos_i[self.num_good_agents :])
        good_pickup = (~moving_after_capture[1]) & (~good_capture) & jnp.any(good_pickup_candidates)
        adv_pickup = (~moving_after_capture[0]) & (~adv_capture) & jnp.any(adv_pickup_candidates)
        good_pickup_idx = jnp.nonzero(good_pickup_candidates, size=1, fill_value=0)[0][0]
        adv_pickup_idx = jnp.nonzero(adv_pickup_candidates, size=1, fill_value=0)[0][0]

        flag_carrier = jnp.asarray(
            (
                jnp.where(good_pickup, good_pickup_idx, carrier_before[0]),
                jnp.where(adv_pickup, adv_pickup_idx, carrier_before[1]),
            ),
            dtype=jnp.int32,
        )
        flag_moving = jnp.asarray(
            (
                moving_after_capture[0] | adv_pickup,
                moving_after_capture[1] | good_pickup,
            )
        )
        flag_p_pos = self._flag_positions_after_events(
            state,
            new_pos,
            moving_before,
            flag_moving,
            flag_carrier,
            good_capture,
            adv_capture,
        )
        flag_p_vel = flag_p_pos - state.flag_p_pos

        step = state.step + 1
        done_all = step >= self.max_steps
        done = jnp.full((self.num_agents,), done_all)
        next_state = state.replace(
            p_pos=new_pos,
            p_vel=new_vel,
            flag_moving=flag_moving,
            flag_carrier=flag_carrier,
            flag_p_pos=flag_p_pos,
            flag_p_vel=flag_p_vel,
            done=done,
            step=step,
            prev_p_pos=new_pos,
        )
        rewards = self._rewards(good_capture, adv_capture)
        dones = {agent: done[idx] for idx, agent in enumerate(self.agents)}
        dones["__all__"] = done_all
        subtask_obs = self.subtask_obs_fn(
            pressed,
            old_pos,
            new_pos_i,
            flag_moving,
            flag_carrier,
            moving_before,
            carrier_before,
            good_capture,
            adv_capture,
            collision_masks,
        )
        info = {
            "adversary_dropped": adv_capture,
            "agent_dropped": good_capture,
            "button_pos": next_state.button_pos,
            "subtask_obs": subtask_obs,
        }
        return self.obs_fn_o(next_state), next_state, rewards, dones, info

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        return self.obs_fn_o(state)

    @partial(jax.jit, static_argnums=(0,))
    def obs_fn_o(self, state: State) -> Dict[str, chex.Array]:
        heading_vec = -state.p_pos
        theta = jnp.arctan2(heading_vec[:, 1], heading_vec[:, 0])
        sin = jnp.sin(theta)
        cos = jnp.cos(theta)
        rot = jnp.stack(
            (
                jnp.stack((cos, sin), axis=1),
                jnp.stack((-sin, cos), axis=1),
            ),
            axis=1,
        )
        rel_pos = jnp.einsum("nij,nkj->nki", rot, state.p_pos[None, :, :] - state.p_pos[:, None, :])
        rel_vel = jnp.einsum("nij,nkj->nki", rot, state.p_vel[None, :, :] - state.p_vel[:, None, :])
        rel_obs = jnp.einsum("nij,nkj->nki", rot, state.obs_pos[None, :, :] - state.p_pos[:, None, :])
        rel_flag = jnp.einsum("nij,nkj->nki", rot, state.flag_p_pos[None, :, :] - state.p_pos[:, None, :])
        rel_agent_zone = jnp.einsum("nij,nj->ni", rot, state.agent_zone[None, :] - state.p_pos)
        rel_adv_zone = jnp.einsum("nij,nj->ni", rot, state.adversary_zone[None, :] - state.p_pos)
        rel_buttons = jnp.einsum("nij,nkj->nki", rot, state.button_pos[None, :, :] - state.p_pos[:, None, :])
        pressed_any = jnp.any(self._pressed_mask(state, jnp.zeros((self.num_agents,), dtype=jnp.int32)), axis=1).astype(jnp.float32)

        def obs_for_agent(i: int, name: str) -> chex.Array:
            is_good = i < self.num_good_agents
            local_idx = i if is_good else i - self.num_good_agents
            own_start = 0 if is_good else self.num_good_agents
            own_count = self.num_good_agents if is_good else self.num_adversaries
            opp_start = self.num_good_agents if is_good else 0
            opp_count = self.num_adversaries if is_good else self.num_good_agents
            own_positions = jnp.roll(
                rel_pos[i, own_start : own_start + own_count],
                shift=own_count - local_idx - 1,
                axis=0,
            )[: own_count - 1]
            own_velocities = jnp.roll(
                rel_vel[i, own_start : own_start + own_count],
                shift=own_count - local_idx - 1,
                axis=0,
            )[: own_count - 1]
            opp_positions = rel_pos[i, opp_start : opp_start + opp_count]
            opp_velocities = rel_vel[i, opp_start : opp_start + opp_count]
            carrying = jnp.asarray(
                jax.lax.select(
                    is_good,
                    state.flag_moving[1] & (state.flag_carrier[0] == local_idx),
                    state.flag_moving[0] & (state.flag_carrier[1] == local_idx),
                ),
                dtype=jnp.float32,
            ).reshape((1,))
            first_flag = rel_flag[i, 1 if is_good else 0][None, :]
            second_flag = rel_flag[i, 0 if is_good else 1][None, :]
            own_zone = rel_agent_zone[i][None, :] if is_good else rel_adv_zone[i][None, :]
            opp_zone = rel_adv_zone[i][None, :] if is_good else rel_agent_zone[i][None, :]
            del name
            return jnp.concatenate(
                (
                    state.p_vel[i].reshape(-1),
                    own_positions.reshape(-1),
                    own_velocities.reshape(-1),
                    opp_positions.reshape(-1),
                    opp_velocities.reshape(-1),
                    rel_obs[i].reshape(-1),
                    first_flag.reshape(-1),
                    second_flag.reshape(-1),
                    own_zone.reshape(-1),
                    opp_zone.reshape(-1),
                    carrying.reshape(-1),
                    rel_buttons[i].reshape(-1),
                    pressed_any.reshape(-1),
                ),
                axis=0,
            )

        return {agent: obs_for_agent(idx, agent) for idx, agent in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=(0,))
    def subtask_obs_fn(
        self,
        pressed: chex.Array,
        old_pos: chex.Array,
        new_pos: chex.Array,
        flag_moving: chex.Array,
        flag_carrier: chex.Array,
        moving_before: chex.Array,
        carrier_before: chex.Array,
        good_capture: chex.Array,
        adv_capture: chex.Array,
        collision_masks: chex.Array,
    ) -> Dict[str, chex.Array]:
        old_agent = self._in_agent_zone(old_pos)
        old_adv = self._in_adversary_zone(old_pos)
        new_agent = self._in_agent_zone(new_pos)
        new_adv = self._in_adversary_zone(new_pos)
        crossed_in_agent = ~old_agent & new_agent
        crossed_out_agent = old_agent & ~new_agent
        crossed_in_adv = ~old_adv & new_adv
        crossed_out_adv = old_adv & ~new_adv
        moving_opp = jnp.asarray(
            (
                moving_before[1],
                moving_before[0],
            )
        )
        carrier_opp = jnp.asarray(
            (
                carrier_before[0],
                carrier_before[1],
            ),
            dtype=jnp.int32,
        )

        def per_agent(i: int, name: str) -> chex.Array:
            is_good = i < self.num_good_agents
            local_idx = i if is_good else i - self.num_good_agents
            pressed_own = pressed[0, i] if is_good else pressed[1, i]
            cross_in_opp = crossed_in_adv[i] if is_good else crossed_in_agent[i]
            cross_out_opp = crossed_out_adv[i] if is_good else crossed_out_agent[i]
            cross_in_own = crossed_in_agent[i] if is_good else crossed_in_adv[i]
            cross_out_own = crossed_out_agent[i] if is_good else crossed_out_adv[i]
            carrying = (flag_moving[1] & (flag_carrier[0] == local_idx)) if is_good else (
                flag_moving[0] & (flag_carrier[1] == local_idx)
            )
            was_carrying = (moving_opp[0] & (carrier_opp[0] == local_idx)) if is_good else (
                moving_opp[1] & (carrier_opp[1] == local_idx)
            )
            dropping = (was_carrying & good_capture) if is_good else (was_carrying & adv_capture)
            del name
            return jnp.asarray(
                (
                    pressed_own,
                    cross_in_opp,
                    cross_out_opp,
                    cross_in_own,
                    cross_out_own,
                    carrying,
                    dropping,
                    collision_masks[i, 0],
                    collision_masks[i, 1],
                    collision_masks[i, 2],
                ),
                dtype=jnp.float32,
            )

        return {agent: per_agent(idx, agent) for idx, agent in enumerate(self.agents)}

    def _obs_dim(self) -> int:
        return (
            2
            + 2 * (self.num_good_agents - 1)
            + 2 * (self.num_good_agents - 1)
            + 2 * self.num_adversaries
            + 2 * self.num_adversaries
            + 2 * self.num_obstacles
            + 2
            + 2
            + 2
            + 2
            + 1
            + 4
            + 2
        )

    def _agent_zone(self) -> chex.Array:
        return jnp.asarray((2, self.height // 2), dtype=jnp.int32)

    def _adversary_zone(self) -> chex.Array:
        return jnp.asarray((self.width - 3, self.height // 2), dtype=jnp.int32)

    def _button_positions(self) -> chex.Array:
        y = self.height // 2
        return jnp.asarray(
            (
                (2, min(self.height - 1, y + self.zone_radius_cells + 2)),
                (self.width - 3, max(0, y - self.zone_radius_cells - 2)),
            ),
            dtype=jnp.int32,
        )

    def _obstacle_positions(self) -> chex.Array:
        return jnp.asarray(((self.width // 2, self.height // 2),), dtype=jnp.int32)

    def _base_agent_positions(self) -> chex.Array:
        y = self.height // 2
        return jnp.asarray(
            (
                (1, y - 1),
                (1, y),
                (1, y + 1),
                (self.width - 2, y - 1),
                (self.width - 2, y),
                (self.width - 2, y + 1),
            ),
            dtype=jnp.int32,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _action_deltas(self, actions: chex.Array) -> chex.Array:
        table = jnp.asarray(
            (
                (0, 0),
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
            ),
            dtype=jnp.int32,
        )
        return table[jnp.clip(actions, 0, 4)]

    @partial(jax.jit, static_argnums=(0,))
    def _clip_grid(self, cells: chex.Array) -> chex.Array:
        return jnp.stack(
            (
                jnp.clip(cells[..., 0], 0, self.width - 1),
                jnp.clip(cells[..., 1], 0, self.height - 1),
            ),
            axis=-1,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _avoid_static_cells(self, p_pos: chex.Array, fallback: chex.Array) -> chex.Array:
        static = jnp.concatenate(
            (self._obstacle_positions(), self._button_positions(), self._agent_zone()[None, :], self._adversary_zone()[None, :]),
            axis=0,
        )
        blocked = jnp.any(jnp.all(p_pos[:, None, :] == static[None, :, :], axis=-1), axis=1)
        return jnp.where(blocked[:, None], fallback, p_pos)

    @partial(jax.jit, static_argnums=(0,))
    def _pressed_mask(self, state: State, actions: chex.Array) -> chex.Array:
        on_button = jnp.all(state.p_pos[None, :, :] == state.button_pos[:, None, :], axis=-1)
        noop = actions[None, :] == 0
        return on_button & noop

    @partial(jax.jit, static_argnums=(0,))
    def _obstacle_hits(self, proposed: chex.Array, obs_pos: chex.Array) -> chex.Array:
        return jnp.any(jnp.all(proposed[:, None, :] == obs_pos[None, :, :], axis=-1), axis=1)

    @partial(jax.jit, static_argnums=(0,))
    def _block_gated_entries(self, old_pos: chex.Array, proposed: chex.Array, pressed: chex.Array) -> chex.Array:
        old_agent = self._in_agent_zone(old_pos)
        old_adv = self._in_adversary_zone(old_pos)
        new_agent = self._in_agent_zone(proposed)
        new_adv = self._in_adversary_zone(proposed)
        entering_agent_zone = ~old_agent & new_agent
        entering_adv_zone = ~old_adv & new_adv
        idx = jnp.arange(self.num_agents)
        is_good = idx < self.num_good_agents

        def pressed_by_other(button_id: int, agent_id: int) -> chex.Array:
            return jnp.any(pressed[button_id] & (idx != agent_id))

        button_0_other = jax.vmap(lambda i: pressed_by_other(0, i))(idx)
        button_1_other = jax.vmap(lambda i: pressed_by_other(1, i))(idx)
        block_good = is_good & entering_adv_zone & ~button_0_other
        block_adv = (~is_good) & entering_agent_zone & ~button_1_other
        blocked = block_good | block_adv
        return jnp.where(blocked[:, None], old_pos, proposed)

    @partial(jax.jit, static_argnums=(0,))
    def _resolve_collisions(self, old_pos: chex.Array, proposed: chex.Array, obstacle_hit: chex.Array) -> Tuple[chex.Array, chex.Array]:
        moving = jnp.any(proposed != old_pos, axis=1)
        same_target = jnp.all(proposed[:, None, :] == proposed[None, :, :], axis=-1)
        same_count = jnp.sum(same_target, axis=1)
        contested = moving & (same_count > 1)
        swap = jnp.any(
            (
                jnp.all(proposed[:, None, :] == old_pos[None, :, :], axis=-1)
                & jnp.all(proposed[None, :, :] == old_pos[:, None, :], axis=-1)
                & moving[:, None]
                & moving[None, :]
                & (~jnp.eye(self.num_agents, dtype=bool))
            ),
            axis=1,
        )
        blocked = contested | swap
        final_pos = jnp.where(blocked[:, None], old_pos, proposed)
        collide_obs = obstacle_hit
        same_final = jnp.all(final_pos[:, None, :] == final_pos[None, :, :], axis=-1) & ~jnp.eye(self.num_agents, dtype=bool)
        idx = jnp.arange(self.num_agents)
        is_good = idx < self.num_good_agents
        same_team = is_good[:, None] == is_good[None, :]
        collide_own = jnp.any(same_final & same_team, axis=1) | contested | swap
        collide_opp = jnp.any(same_final & ~same_team, axis=1)
        return final_pos, jnp.stack((collide_obs, collide_own, collide_opp), axis=1)

    @partial(jax.jit, static_argnums=(0,))
    def _in_agent_zone(self, cells: chex.Array) -> chex.Array:
        return jnp.max(jnp.abs(cells - self._agent_zone()), axis=-1) <= self.zone_radius_cells

    @partial(jax.jit, static_argnums=(0,))
    def _in_adversary_zone(self, cells: chex.Array) -> chex.Array:
        return jnp.max(jnp.abs(cells - self._adversary_zone()), axis=-1) <= self.zone_radius_cells

    @partial(jax.jit, static_argnums=(0,))
    def _flag_positions_after_events(
        self,
        state: State,
        new_pos: chex.Array,
        moving_before: chex.Array,
        flag_moving: chex.Array,
        flag_carrier: chex.Array,
        good_capture: chex.Array,
        adv_capture: chex.Array,
    ) -> chex.Array:
        agent_flag_pos = jax.lax.select(
            adv_capture,
            new_pos[flag_carrier[1] + self.num_good_agents],
            jax.lax.select(
                flag_moving[0],
                new_pos[flag_carrier[1] + self.num_good_agents],
                jax.lax.select(moving_before[0], state.flag_p_pos[0], state.agent_zone),
            ),
        )
        adversary_flag_pos = jax.lax.select(
            good_capture,
            new_pos[flag_carrier[0]],
            jax.lax.select(
                flag_moving[1],
                new_pos[flag_carrier[0]],
                jax.lax.select(moving_before[1], state.flag_p_pos[1], state.adversary_zone),
            ),
        )
        return jnp.stack((agent_flag_pos, adversary_flag_pos)).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _rewards(self, good_capture: chex.Array, adv_capture: chex.Array) -> Dict[str, chex.Array]:
        rewards = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        rewards = rewards.at[: self.num_good_agents].add(good_capture.astype(jnp.float32))
        rewards = rewards.at[self.num_good_agents :].add(adv_capture.astype(jnp.float32))
        if self.zero_sum:
            rewards = rewards.at[: self.num_good_agents].add(-adv_capture.astype(jnp.float32))
            rewards = rewards.at[self.num_good_agents :].add(-good_capture.astype(jnp.float32))
        return {agent: rewards[idx] for idx, agent in enumerate(self.agents)}
