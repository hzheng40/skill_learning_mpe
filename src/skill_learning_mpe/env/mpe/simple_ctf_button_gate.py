import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import chex
from flax import struct
from functools import partial
from typing import Dict, Tuple, List, Literal

from .simple import SimpleMPE
from .simple import State as SimpleState
from .default_params import *

from .simple_ctf import SimpleCTF as _BaseCTF

from ..spaces import Box


@struct.dataclass
class State:
    """
    CTF with *button-gated entry* to zones (exits are always free).

    Extends the base state with:
      - prev_p_pos: previous step's positions (for boundary crossing detection)
      - button_pos: (2,2) positions for [agent_side_button, adversary_side_button]
    """

    # === base fields (kept identical names/shapes) ===
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
    # === extended ===
    prev_p_pos: chex.Array
    button_pos: chex.Array  # (2,2)


class SimpleCTFButtons(_BaseCTF):
    """
    Variant of SimpleCTF where an agent may **ENTER** a zone **only if** the
    *opposite* button is being pressed by another agent; **LEAVING** any zone is free.

    Mapping:
      - Button near Team A's zone (index 0) unlocks **Team B's zone**.
      - Button near Team B's zone (index 1) unlocks **Team A's zone**.

    A button is considered *pressed* if any qualifying agent is within
    `button_radius` of the button and is *stationary* (||vel|| <= vel_eps).
    By default, any agent (either team) can press; set `button_team` to 'same' or
    'opposite' to restrict who can press relative to the crossing agent.

    Parameters
    ----------
    button_radius : float
        Distance threshold for pressing. Default 1.2 (larger visual/activation area).
    vel_eps : float
        Velocity magnitude considered stationary. Default 0.1.
    side_offset : float
        Additional distance beyond the zone boundary to place the side buttons
        along the **tangential** direction (perpendicular to the line connecting zones).
        Default 1.0.
    button_team : {'any','same','opposite'}
        Who is allowed to press relative to the crossing agent. Default 'any'.
    """

    def __init__(
        self,
        *,
        button_radius: float = 1.2,
        vel_eps: float = 2.0,
        side_offset: float = 1.0,
        button_team: Literal["any", "same", "opposite"] = "any",
        block: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.button_radius = button_radius
        self.vel_eps = vel_eps
        self.side_offset = side_offset
        self.button_team = button_team
        self.block = block

        # update observation space sizes (+6 scalars per agent)
        for name in self.agents:
            old_dim = self.observation_spaces[name].shape[0]
            new_dim = old_dim + 6
            self.observation_spaces[name] = Box(-jnp.inf, jnp.inf, (new_dim,))

    @partial(jax.jit, static_argnums=[0])
    def _place_buttons(
        self, agent_zone: chex.Array, adversary_zone: chex.Array
    ) -> chex.Array:
        """Place one button near the **side** of each zone.

        Buttons are positioned tangent to the zone boundary, i.e., perpendicular
        to the line connecting the two zones, so they appear "on the sides".
        - Button 0 sits at agent_zone + v * (zone_size + side_offset)
        - Button 1 sits at adversary_zone - v * (zone_size + side_offset)
          where v = perp(unit(adversary_zone - agent_zone)).
        """
        vec = adversary_zone - agent_zone
        dist = jnp.linalg.norm(vec) + 1e-8
        u = vec / dist  # from A to B
        v = jnp.array([-u[1], u[0]])  # tangent (left of u)
        a_btn = agent_zone + v * (self.zone_size + self.side_offset)
        b_btn = adversary_zone - v * (self.zone_size + self.side_offset)
        return jnp.vstack((a_btn, b_btn))  # [0]=near A, [1]=near B

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        obs, base_state = super().reset(key)
        button_pos = self._place_buttons(
            base_state.agent_zone, base_state.adversary_zone
        )
        ext_state = State(
            p_pos=base_state.p_pos,
            p_vel=base_state.p_vel,
            flag_moving=base_state.flag_moving,
            flag_carrier=base_state.flag_carrier,
            flag_p_pos=base_state.flag_p_pos,
            flag_p_vel=base_state.flag_p_vel,
            done=base_state.done,
            step=base_state.step,
            agent_zone=base_state.agent_zone,
            adversary_zone=base_state.adversary_zone,
            obs_pos=base_state.obs_pos,
            agent_names=base_state.agent_names,
            adversary_names=base_state.adversary_names,
            prev_p_pos=base_state.p_pos,
            button_pos=button_pos,
        )
        return self.obs_fn_o(ext_state), ext_state

    @partial(jax.jit, static_argnums=[0])
    def _pressed_mask(self, state: State) -> chex.Array:
        """
        Returns pressed[k, j] indicating whether agent j is pressing button for zone k.
        A press is within button_radius and stationary (||v|| <= vel_eps).
        """
        btn = state.button_pos  # (2,2)
        pos = state.p_pos  # (N,2)
        vel = state.p_vel  # (N,2)
        dists = jnp.linalg.norm(btn[:, None, :] - pos[None, :, :], axis=-1)  # (2,N)
        near = dists <= self.button_radius
        still = jnp.linalg.norm(vel[None, :, :], axis=-1) <= self.vel_eps
        return jnp.logical_and(near, still)  # shape (2, N)

    @partial(jax.jit, static_argnums=[0])
    def _entry_blocks(
        self, state: State, new_pos: chex.Array, per_agent_pressed: chex.Array
    ):
        """
        Compute which agents must be blocked from **entering** zones (exits are free).

        Button mapping (opposite unlock):
          - enter Team A zone requires **button 1** (near Team B) pressed by another.
          - enter Team B zone requires **button 0** (near Team A) pressed by another.
        """
        N = self.num_actors
        num_good = self.num_good_agents

        pos_prev = state.prev_p_pos
        pos_curr = new_pos

        def entering(center):
            # rad_in = self.zone_size - self.agent_size
            rad_in = self.zone_size
            prev_in = jnp.linalg.norm(pos_prev - center, axis=1) < (rad_in - 1e-6)
            curr_in = jnp.linalg.norm(pos_curr - center, axis=1) < (rad_in - 1e-6)
            return jnp.logical_and(jnp.logical_not(prev_in), curr_in)

        enter_A = entering(state.agent_zone)  # (N,)
        enter_B = entering(state.adversary_zone)  # (N,)

        is_agent = jnp.arange(N) < num_good
        press = per_agent_pressed  # (2, N)

        def team_ok(i, j):
            if self.button_team == "any":
                return True
            same = jnp.equal(is_agent[i], is_agent[j])
            return jax.lax.select(
                self.button_team == "same", same, jnp.logical_not(same)
            )

        def per_i(i):
            others = jnp.arange(N) != i
            team_valid = jax.vmap(lambda j: team_ok(i, j))(jnp.arange(N))
            valid = jnp.logical_and(others, team_valid)
            # pressed_by_other[k] for k in {0,1}
            return jnp.array(
                [
                    jnp.any(jnp.logical_and(press[0], valid)),
                    jnp.any(jnp.logical_and(press[1], valid)),
                ]
            )

        pressed_by_other = jax.vmap(per_i)(jnp.arange(N))  # (N,2)

        # Opposite unlock mapping
        need_for_A = pressed_by_other[:, 1]  # need button near B to enter A
        need_for_B = pressed_by_other[:, 0]  # need button near A to enter B

        # Only gate the zone against the opposing team when the button is not held.
        block_A = jnp.logical_and(
            jnp.logical_and(enter_A, jnp.logical_not(need_for_A)),
            jnp.logical_not(is_agent),
        )
        block_B = jnp.logical_and(
            jnp.logical_and(enter_B, jnp.logical_not(need_for_B)),
            is_agent,
        )
        return jnp.logical_or(block_A, block_B)  # (N,)

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        # ===== Base CTF logic up to physics step =====
        opp_flag_in = jnp.linalg.norm(state.flag_p_pos[1] - state.agent_zone) <= (
            self.zone_size - self.agent_size
        )
        ego_flag_in = jnp.linalg.norm(state.flag_p_pos[0] - state.adversary_zone) <= (
            self.zone_size - self.agent_size
        )

        reset_opp_flag = jax.lax.select(
            opp_flag_in, state.adversary_zone, state.flag_p_pos[1]
        )
        reset_ego_flag = jax.lax.select(
            ego_flag_in, state.agent_zone, state.flag_p_pos[0]
        )
        state = state.replace(flag_p_pos=jnp.vstack((reset_ego_flag, reset_opp_flag)))

        agent_carrying = jax.lax.select(
            state.flag_moving[1],
            jnp.full((self.num_good_agents,), False)
            .at[state.flag_carrier[0]]
            .set(True),
            jnp.full((self.num_good_agents,), False),
        )
        adv_carrying = jax.lax.select(
            state.flag_moving[0],
            jnp.full((self.num_adversaries,), False)
            .at[state.flag_carrier[1]]
            .set(True),
            jnp.full((self.num_adversaries,), False),
        )

        agent_can_drop = jnp.linalg.norm(
            state.p_pos[: self.num_good_agents] - state.agent_zone, axis=1
        ) <= (self.zone_size - self.agent_size)
        adv_can_drop = jnp.linalg.norm(
            state.p_pos[self.num_good_agents :] - state.adversary_zone, axis=1
        ) <= (self.zone_size - self.agent_size)
        agent_dropping_mask = jnp.logical_and(agent_carrying, agent_can_drop)
        adv_dropping_mask = jnp.logical_and(adv_carrying, adv_can_drop)

        flag_moving_after_drop = jnp.hstack(
            (
                jax.lax.select(
                    state.flag_moving[0],
                    ~jnp.logical_and(jnp.any(adv_dropping_mask), state.flag_moving[0]),
                    False,
                ),
                jax.lax.select(
                    state.flag_moving[1],
                    ~jnp.logical_and(
                        jnp.any(agent_dropping_mask), state.flag_moving[1]
                    ),
                    False,
                ),
            )
        )
        state = state.replace(flag_moving=flag_moving_after_drop)

        agent_can_pickup = jnp.linalg.norm(
            state.p_pos[: self.num_good_agents] - state.adversary_zone, axis=1
        ) <= (self.zone_size - self.agent_size)
        adv_can_pickup = jnp.linalg.norm(
            state.p_pos[self.num_good_agents :] - state.agent_zone, axis=1
        ) <= (self.zone_size - self.agent_size)

        state = state.replace(
            flag_carrier=jnp.hstack(
                (
                    jax.lax.select(
                        state.flag_moving[1],
                        state.flag_carrier[0],
                        jnp.nonzero(agent_can_pickup, size=1, fill_value=0)[0][0],
                    ),
                    jax.lax.select(
                        state.flag_moving[0],
                        state.flag_carrier[1],
                        jnp.nonzero(adv_can_pickup, size=1, fill_value=0)[0][0],
                    ),
                )
            )
        )

        state = state.replace(
            flag_moving=jnp.hstack(
                (
                    jnp.logical_or(jnp.any(adv_can_pickup), state.flag_moving[0]),
                    jnp.logical_or(jnp.any(agent_can_pickup), state.flag_moving[1]),
                )
            )
        )

        simple_state = SimpleState(
            p_pos=jnp.vstack(
                (state.p_pos, state.obs_pos, state.agent_zone, state.adversary_zone)
            ),
            p_vel=jnp.vstack((state.p_vel, jnp.zeros((self.num_obstacles + 2, 2)))),
            done=state.done,
            step=state.step,
            goal=None,
            c=jnp.zeros((self.num_actors, self.dim_c)),
        )
        simple_obs, simple_state, simple_reward, simple_dones, simple_info = (
            SimpleMPE.step_env(self, key, simple_state, actions)
        )

        new_actor_pos = simple_state.p_pos[: self.num_actors, :]
        new_actor_vel = simple_state.p_vel[: self.num_actors, :]

        # ====== BUTTON GATE ENFORCEMENT (ENTRY ONLY with opposite unlock) ======
        pressed = self._pressed_mask(state)  # (2, N)
        if self.block:
            must_block = self._entry_blocks(state, new_actor_pos, pressed)

            gated_pos = jnp.where(must_block[:, None], state.prev_p_pos, new_actor_pos)
            gated_vel = jnp.where(
                must_block[:, None], jnp.zeros_like(new_actor_vel), new_actor_vel
            )

            state = state.replace(
                p_pos=gated_pos,
                p_vel=gated_vel,
                done=simple_state.done,
                step=simple_state.step,
            )
        else:
            state = state.replace(
                p_pos=new_actor_pos,
                p_vel=new_actor_vel,
                done=simple_state.done,
                step=simple_state.step,
            )

        # Flags ride with carriers
        new_flag_p_pos = jnp.vstack(
            (
                jax.lax.select(
                    state.flag_moving[0],
                    state.p_pos[state.flag_carrier[1] + self.num_good_agents, :],
                    state.flag_p_pos[0, :],
                ),
                jax.lax.select(
                    state.flag_moving[1],
                    state.p_pos[state.flag_carrier[0], :],
                    state.flag_p_pos[1, :],
                ),
            )
        )
        new_flag_p_vel = jnp.vstack(
            (
                jax.lax.select(
                    state.flag_moving[0],
                    state.p_vel[state.flag_carrier[1] + self.num_good_agents, :],
                    state.flag_p_vel[0, :],
                ),
                jax.lax.select(
                    state.flag_moving[1],
                    state.p_vel[state.flag_carrier[0], :],
                    state.flag_p_vel[1, :],
                ),
            )
        )
        state = state.replace(flag_p_pos=new_flag_p_pos, flag_p_vel=new_flag_p_vel)

        rewards = jax.lax.cond(
            self.zero_sum, self.get_zero_sum_rewards, self.get_rewards, state
        )

        # Update prev positions AFTER enforcement
        state = state.replace(prev_p_pos=state.p_pos)

        obs = self.obs_fn_o(state)
        subtask_obs = self.subtask_obs_fn(
            pressed,
            agent_can_pickup,
            adv_can_pickup,
            agent_can_drop,
            adv_can_drop,
            agent_carrying,
            adv_carrying,
            agent_dropping_mask,
            adv_dropping_mask,
            state,
        )
        info = {
            "adversary_dropped": ego_flag_in,
            "agent_dropped": opp_flag_in,
            "button_pos": state.button_pos,
            "subtask_obs": subtask_obs,
        }
        return obs, state, rewards, simple_dones, info

    @partial(jax.jit, static_argnums=[0])
    def subtask_obs_fn(
        self,
        pressed: chex.Array,
        curr_agent_in_adv_zone: chex.Array,
        curr_adv_in_agent_zone: chex.Array,
        curr_agent_in_own_zone: chex.Array,
        curr_adv_in_own_zone: chex.Array,
        agent_carrying: chex.Array,
        adv_carrying: chex.Array,
        agent_dropping_mask: chex.Array,
        adv_dropping_mask: chex.Array,
        state: State,
    ):
        """Subtask observations for all agents.
        pressed: (2, N) boolean array indicating which agents are pressing which buttons.
        agent_can_pickup: (N_good,) boolean array indicating which agents can pick up the adversary flag.
        adv_can_pickup: (N_adv,) boolean array indicating which adversaries can pick up the agent flag.
        agent_carrying: (N_good,) boolean array indicating which agents are carrying the adversary flag.
        adv_carrying: (N_adv,) boolean array indicating which adversaries are carrying the agent flag.
        agent_dropping_mask: (N_good,) boolean array indicating which agents are dropping the adversary flag.
        adv_dropping_mask: (N_adv,) boolean array indicating which adversaries are dropping the agent flag.
        state: State object containing the environment state.

        returns the subtask observations for all agents as a dictionary.
        Each agent's observation is a concatenation of the following 6 booleans:
            [pressed_own, in_opp_zone, carrying_opp_flag, dropping_opp_flag, moving_towards_own_zone, moving_towards_opp_zone]
        """

        # crossing in/out of zones
        # prev not in zone  + curr in zone / prev in zone + curr not in zone

        prev_in_agent_zone = jnp.linalg.norm(
            state.prev_p_pos - state.agent_zone, axis=1
        ) <= (self.zone_size - self.agent_size)
        prev_out_agent_zone = jnp.linalg.norm(
            state.prev_p_pos - state.agent_zone, axis=1
        ) > (self.zone_size + self.agent_size)
        prev_in_adv_zone = jnp.linalg.norm(
            state.prev_p_pos - state.adversary_zone, axis=1
        ) <= (self.zone_size - self.agent_size)
        prev_out_adv_zone = jnp.linalg.norm(
            state.prev_p_pos - state.adversary_zone, axis=1
        ) > (self.zone_size + self.agent_size)
        curr_out_agent_zone = jnp.linalg.norm(
            state.p_pos - state.agent_zone, axis=1
        ) > (self.zone_size + self.agent_size)
        curr_out_adv_zone = jnp.linalg.norm(
            state.p_pos - state.adversary_zone, axis=1
        ) > (self.zone_size + self.agent_size)

        ag_cross_into_adv_zone = jnp.logical_and(
            prev_out_adv_zone[: self.num_good_agents],
            curr_agent_in_adv_zone,
        )
        ag_cross_outof_adv_zone = jnp.logical_and(
            prev_in_adv_zone[: self.num_good_agents],
            curr_out_adv_zone[: self.num_good_agents],
        )
        ag_cross_into_own_zone = jnp.logical_and(
            prev_out_agent_zone[: self.num_good_agents],
            curr_agent_in_own_zone,
        )
        ag_cross_outof_own_zone = jnp.logical_and(
            prev_in_agent_zone[: self.num_good_agents],
            curr_out_agent_zone[: self.num_good_agents],
        )
        adv_cross_into_agent_zone = jnp.logical_and(
            prev_out_agent_zone[self.num_good_agents :],
            curr_adv_in_agent_zone,
        )
        adv_cross_outof_agent_zone = jnp.logical_and(
            prev_in_agent_zone[self.num_good_agents :],
            curr_out_agent_zone[self.num_good_agents :],
        )
        adv_cross_into_own_zone = jnp.logical_and(
            prev_out_adv_zone[self.num_good_agents :],
            curr_adv_in_own_zone,
        )
        adv_cross_outof_own_zone = jnp.logical_and(
            prev_in_adv_zone[self.num_good_agents :],
            curr_out_adv_zone[self.num_good_agents :],
        )

        out = {}
        for i, name in enumerate(self.agents):
            is_agent = i < self.num_good_agents
            idx = i if is_agent else i - self.num_good_agents

            pressed_own = pressed[0, i] if is_agent else pressed[1, i]

            cross_in_opp_zone = (
                ag_cross_into_adv_zone[idx]
                if is_agent
                else adv_cross_into_agent_zone[idx]
            )
            cross_outof_opp_zone = (
                ag_cross_outof_adv_zone[idx]
                if is_agent
                else adv_cross_outof_agent_zone[idx]
            )
            cross_in_own_zone = (
                ag_cross_into_own_zone[idx]
                if is_agent
                else adv_cross_into_own_zone[idx]
            )
            cross_outof_own_zone = (
                ag_cross_outof_own_zone[idx]
                if is_agent
                else adv_cross_outof_own_zone[idx]
            )

            carrying_opp_flag = agent_carrying[idx] if is_agent else adv_carrying[idx]
            dropping_opp_flag = (
                agent_dropping_mask[idx] if is_agent else adv_dropping_mask[idx]
            )

            # colliding into anything
            collide_obs = jnp.any(
                jnp.linalg.norm(state.p_pos[i] - state.obs_pos)
                <= (self.agent_size + self.obstacle_size)
            )

            # collide teammate
            if is_agent:
                own_dist = jnp.linalg.norm(
                    state.p_pos[idx]
                    - jnp.delete(state.p_pos[: self.num_good_agents], idx, axis=0),
                    axis=1,
                )
                other_dist = jnp.linalg.norm(
                    state.p_pos[idx] - state.p_pos[self.num_good_agents :], axis=1
                )
                collide_own = jnp.any(own_dist <= 2 * self.agent_size)
                collide_opp = jnp.any(other_dist <= 2 * self.agent_size)
            else:
                own_dist = jnp.linalg.norm(
                    state.p_pos[i]
                    - jnp.delete(state.p_pos[self.num_good_agents :], idx, axis=0),
                    axis=1,
                )
                other_dist = jnp.linalg.norm(
                    state.p_pos[i] - state.p_pos[: self.num_good_agents], axis=1
                )
                collide_own = jnp.any(own_dist <= 2 * self.agent_size)
                collide_opp = jnp.any(other_dist <= 2 * self.agent_size)

            obs_vec = jnp.array(
                [
                    pressed_own,
                    cross_in_opp_zone,
                    cross_outof_opp_zone,
                    cross_in_own_zone,
                    cross_outof_own_zone,
                    carrying_opp_flag,
                    dropping_opp_flag,
                    collide_obs,
                    collide_own,
                    collide_opp,
                ],
                dtype=jnp.float32,
            )
            out[name] = obs_vec
        return out

    @partial(jax.jit, static_argnums=[0])
    def obs_fn_o(self, state: State):
        """Parent obs + button positions + pressed flags (absolute/global).

        Appends 6 scalars per agent, in this order:
        [btnA_x, btnA_y, btnB_x, btnB_y, pressed_A, pressed_B]

        pressed_* are 1.0 if any agent (of either team) is currently pressing
        that button (within radius and below vel_eps), else 0.0.
        """
        base = self.obs_fn(state)

        # 4 scalars: button centers (absolute)
        # btn_flat = state.button_pos.reshape(-1)  # (4,)

        btns = state.button_pos

        # 2 scalars: pressed flags (any agent pressing per button)
        pressed_mask = self._pressed_mask(state)  # (2, N)
        pressed_any = jnp.any(pressed_mask, axis=1).astype(jnp.float32)  # (2,)

        out = {}

        for i, name in enumerate(self.agents):
            heading_vec = -state.p_pos[i]
            theta = jnp.arctan2(heading_vec[1], heading_vec[0])
            sin = jnp.sin(theta)
            cos = jnp.cos(theta)
            rotmat = jnp.array([[cos, sin], [-sin, cos]])
            rel = ((btns - state.p_pos[i]) @ rotmat.T).flatten()
            out[name] = jnp.concatenate([base[name], rel, pressed_any], axis=-1)

        return out
