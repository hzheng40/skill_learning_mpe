import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import chex
from flax import struct

from typing import Tuple, List, Dict
from functools import partial

from .simple import SimpleMPE
from .simple import State as SimpleState
from .default_params import *

from ..spaces import Box, Discrete
from ..multi_agent_env import MultiAgentEnv


@struct.dataclass
class State:
    """
    Basic CTF State
    """

    # dynamic things
    p_pos: chex.Array  # [num_entities, [x, y]]
    p_vel: chex.Array  # [n, [x, y]]
    flag_moving: chex.Array  # [2, ]
    flag_carrier: chex.Array  # [2, ]
    flag_p_pos: chex.Array  # [2, [x, y]]
    flag_p_vel: chex.Array  # [2, [x, y]]
    done: chex.Array  # bool [num_agents, ]
    step: int  # current step
    # static things
    agent_zone: chex.Array  # [x, y]
    adversary_zone: chex.Array  # [x, y]
    obs_pos: chex.Array  # [num_obstacles, [x, y]]
    # untraced things
    agent_names: List[str] = struct.field(pytree_node=False)
    adversary_names: List[str] = struct.field(pytree_node=False)


class SimpleCTF(SimpleMPE):
    """
    Capture the Flag environment.
    Two teams of agents compete to capture the other team's flag and return it to their side.

    Actions: (NOOP, L, R, D, U, Pickup/Drop Flag)
    """

    def __init__(
        self,
        num_good_agents: int = 3,
        num_adversaries: int = 3,
        num_obstacles: int = 2,
        action_type=DISCRETE_ACT,
        zone_size: float = 5.0,
        dist_between_zones: float = 15.0,
        agent_size: float = 1.0,
        obstacle_size: float = 1.5,
        zero_sum: bool = False,
        abs_obs: bool = False,
        random_start: bool = False,
        transform: bool = False,
        init_agent_everywhere: bool = False,
        include_velocities: bool = True,
    ):
        # zone and agent radius
        self.zone_size = zone_size
        self.agent_size = agent_size
        self.obstacle_size = obstacle_size
        self.num_good_agents = num_good_agents
        self.num_adversaries = num_adversaries
        self.dist_between_zones = dist_between_zones
        self.num_obstacles = num_obstacles
        self.action_type = action_type
        self.max_steps = CTF_MAX_STEPS
        self.zero_sum = zero_sum
        self.abs_obs = abs_obs
        self.random_start = random_start
        self.init_agent_everywhere = init_agent_everywhere
        self.include_velocities = include_velocities

        # dot keys
        self.good_agents = [f"agent_{i}" for i in range(num_good_agents)]
        self.adversaries = [f"adversary_{i}" for i in range(num_adversaries)]
        self.flags = ["agent_flag", "adversary_flag"]

        # agents = self.good_agents + self.adversaries + self.flags
        agents = self.good_agents + self.adversaries

        obs = [f"obstacle_{i}" for i in range(num_obstacles)]
        zones = ["agent_zone", "adversary_zone"]
        landmarks = obs + zones

        if action_type == "Discrete":
            self.action_spaces = {i: Discrete(5) for i in agents}
        elif action_type == "Continuous":
            self.action_spaces = {i: Box(-1, 1, (5,)) for i in agents}

        # agents from both teams
        self.num_actors = num_good_agents + num_adversaries
        # (for physics) num agents is actors + 2 flags
        # num_agents = self.num_actors + 2

        # 2 zones for each team and obstacles
        num_landmarks = num_obstacles + 2

        self.agent_range = jnp.arange(self.num_actors)

        # collision enabled for actors, flags and obstacles, but not for zones
        collides = jnp.concatenate(
            [
                jnp.full((self.num_actors,), True),
                jnp.full((num_obstacles,), True),
                jnp.full((2,), False),
            ]
        )
        # radius for collisions
        rad = jnp.concatenate(
            [
                jnp.full((self.num_actors,), agent_size),
                jnp.full((num_obstacles,), obstacle_size),
                jnp.full((2,), zone_size),
            ]
        )

        self.no_op_actions = (
            jnp.zeros((1,)) if self.action_type == "Discrete" else jnp.zeros((6,))
        )

        # agent observations:
        # [self_pos, self_vel, all_other_rel_pos, all_other_rel_vel,
        #  all_flag_rel_pos, obs_rel_pos, all_zone_pos]
        # observation_dim = self.num_actors * 4 + 4 + num_obstacles * 2 + 4 + 2
        # Velocities: self_vel (2), teammate_vels (2*num_teammates), opp_vels (2*num_opponents)
        vel_dim = 2 + 2 * (self.num_good_agents - 1) + 2 * self.num_adversaries if include_velocities else 0
        
        rel_observation_dim = (self.num_actors + 1) * 4 + 4 + num_obstacles * 2 + 5
        if not include_velocities:
            # Remove velocities: self_vel (2) + all_other_vels (2*num_actors) = 2 + 2*num_actors
            rel_observation_dim -= (2 + 2 * self.num_actors)
        
        abs_observation_dim = self.num_actors * 4 + num_obstacles * 2 + 4 + 4 + 2
        if not include_velocities:
            # Remove all velocities: 2 * num_actors
            abs_observation_dim -= (2 * self.num_actors)
        
        # For local observations (get_ctf_obs_heading):
        # Structure: self_vel(2) + teammate_pos(4) + teammate_vel(4) + opp_pos(6) + opp_vel(6) + obs(2) + flags(4) + zones(4) + carrying(1) = 33
        # Without velocities: teammate_pos(4) + opp_pos(6) + obs(2) + flags(4) + zones(4) + carrying(1) = 21
        if include_velocities:
            # self_vel(2) + (num_good_agents-1)*2 (tm_pos) + (num_good_agents-1)*2 (tm_vel) + num_adversaries*2 (opp_pos) + num_adversaries*2 (opp_vel) + num_obstacles*2 + 4 (flags) + 4 (zones) + 1 (carrying)
            local_observation_dim = 2 + 4 * (self.num_good_agents - 1) + 4 * self.num_adversaries + num_obstacles * 2 + 9
        else:
            # (num_good_agents-1)*2 (tm_pos) + num_adversaries*2 (opp_pos) + num_obstacles*2 + 4 (flags) + 4 (zones) + 1 (carrying)
            local_observation_dim = 2 * (self.num_good_agents - 1) + 2 * self.num_adversaries + num_obstacles * 2 + 9

        if transform:
            self.obs_fn = self.get_ctf_obs_heading
            observation_dim = local_observation_dim
        else:
            if abs_obs:
                self.obs_fn = self.get_ctf_obs_central_abs
                observation_dim = abs_observation_dim
            else:
                self.obs_fn = self.get_ctf_obs
                observation_dim = rel_observation_dim
        self.observation_spaces = {
            i: Box(-jnp.inf, jnp.inf, (observation_dim,)) for i in agents
        }

        super().__init__(
            num_agents=self.num_actors,
            agents=agents,
            landmarks=landmarks,
            num_landmarks=num_landmarks,
            action_type=action_type,
            action_spaces=self.action_spaces,
            observation_spaces=self.observation_spaces,
            collide=collides,
            rad=rad,
            max_steps=self.max_steps,
        )

    # @partial(jax.jit, static_argnums=[0])
    # def _flag_action_discrete(self, actions: dict):
    #     agent_flag_action = jnp.array([actions[i] == 5 for i in self.good_agents])
    #     adv_flag_action = jnp.array([actions[i] == 5 for i in self.adversaries])
    #     return agent_flag_action.flatten(), adv_flag_action.flatten()

    # @partial(jax.jit, static_argnums=[0])
    # def _flag_action_continuous(self, actions: dict):
    #     agent_flag_action = jnp.array([actions[i][-1] >= 0.5 for i in self.good_agents])
    #     adv_flag_action = jnp.array([actions[i][-1] >= 0.5 for i in self.adversaries])
    #     return agent_flag_action.flatten(), adv_flag_action.flatten()

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        key, agent_init_key, adversary_init_key, adv_zone_init_key, obs_init_key = (
            jax.random.split(key, 5)
        )

        # randomize team zone locations if random_start
        init_key, key = jax.random.split(key)
        if self.random_start:
            agent_zone_init_pos = jax.random.uniform(
                init_key,
                (2,),
                minval=-self.dist_between_zones,
                maxval=self.dist_between_zones,
            )
            agent_flag_init_pos = jnp.copy(agent_zone_init_pos)

            # adversary zone always same distance away, randomize radial position, flag at the center
            adv_z_ang = jax.random.uniform(
                adv_zone_init_key, (1,), minval=0, maxval=2 * jnp.pi
            )
            adversary_zone_init_pos = (
                jnp.array(
                    [
                        self.dist_between_zones * jnp.cos(adv_z_ang),
                        self.dist_between_zones * jnp.sin(adv_z_ang),
                    ]
                ).flatten()
                + agent_zone_init_pos
            )
            adversary_flag_init_pos = jnp.copy(adversary_zone_init_pos)

            # randomize obstacle locations between two zones
            between_zone_vec = adversary_zone_init_pos - agent_zone_init_pos
            vecs = jnp.vstack([between_zone_vec] * self.num_obstacles)
            # equidistance from both zones
            spacing = jnp.linspace(0.0, 1.0, self.num_obstacles + 2)[1:-1][:, None]
            obs_init_pos = agent_zone_init_pos + vecs * spacing

            shifts = jax.random.uniform(
                obs_init_key,
                (self.num_obstacles, 1),
                minval=-self.zone_size,
                maxval=self.zone_size,
            )
            obs_shifted_pos = obs_init_pos + jnp.hstack(
                (
                    shifts * jnp.cos(adv_z_ang + jnp.pi / 2),
                    shifts * jnp.sin(adv_z_ang + jnp.pi / 2),
                )
            )

            if self.init_agent_everywhere:
                # initialize anywhere in the map away from obstacles, inside the circle that encloses both zones
                agent_everywhere_radius = jnp.linalg.norm(
                    adversary_zone_init_pos - agent_zone_init_pos
                )
                # center around obstacle center, radius between obs_size to agent_everywhere_radius, random angle
                ag_r_key, ag_a_key = jax.random.split(agent_init_key)
                agent_init_radius = jax.random.uniform(
                    ag_r_key,
                    (self.num_good_agents, 1),
                    minval=self.obstacle_size,
                    maxval=agent_everywhere_radius,
                )
                agent_init_angle = jax.random.uniform(
                    ag_a_key,
                    (self.num_good_agents, 1),
                    minval=0.0,
                    maxval=2 * jnp.pi,
                )
                agent_init_pos = (
                    agent_init_radius
                    * jnp.hstack(
                        (
                            jnp.cos(agent_init_angle),
                            jnp.sin(agent_init_angle),
                        )
                    )
                    + obs_shifted_pos
                )
                adv_r_key, adv_a_key = jax.random.split(adversary_init_key)
                adversary_init_radius = jax.random.uniform(
                    adv_r_key,
                    (self.num_adversaries, 1),
                    minval=self.obstacle_size,
                    maxval=agent_everywhere_radius,
                )
                adversary_init_angle = jax.random.uniform(
                    adv_a_key,
                    (self.num_adversaries, 1),
                    minval=0.0,
                    maxval=2 * jnp.pi,
                )
                adversary_init_pos = (
                    adversary_init_radius
                    * jnp.hstack(
                        (
                            jnp.cos(adversary_init_angle),
                            jnp.sin(adversary_init_angle),
                        )
                    )
                    + obs_shifted_pos
                )
            else:
                # randomize agent loc inside zone
                max_between_agent_init_radius = self.zone_size - 2 * self.agent_size
                agent_init_pos = (
                    jax.random.uniform(
                        agent_init_key, (self.num_good_agents, 2), minval=-1, maxval=1
                    )
                    * max_between_agent_init_radius
                ) + agent_zone_init_pos
                adversary_init_pos = (
                    jax.random.uniform(
                        adversary_init_key,
                        (self.num_adversaries, 2),
                        minval=-1,
                        maxval=1,
                    )
                    * max_between_agent_init_radius
                ) + adversary_zone_init_pos
                # adversary_init_pos = agent_init_pos
        else:
            # fixed zone locations
            agent_zone_init_pos = jnp.array([-self.dist_between_zones / 2, 0.0])
            adversary_zone_init_pos = jnp.array([self.dist_between_zones / 2, 0.0])
            agent_flag_init_pos = jnp.copy(agent_zone_init_pos)
            adversary_flag_init_pos = jnp.copy(adversary_zone_init_pos)

            # fixed agent loc inside zone with a ring around zone center
            agent_angles = jnp.linspace(
                0, 2 * jnp.pi, self.num_good_agents, endpoint=False
            )
            agent_init_radius = self.zone_size / 2
            agent_init_pos_x_diff = agent_init_radius * jnp.cos(agent_angles)
            agent_init_pos_y_diff = agent_init_radius * jnp.sin(agent_angles)
            agent_init_pos = jnp.vstack(
                (
                    agent_init_pos_x_diff + agent_zone_init_pos[0],
                    agent_init_pos_y_diff + agent_zone_init_pos[1],
                )
            ).T

            # fixed adversary loc inside zone with a ring around zone center
            adversary_angles = jnp.linspace(
                -jnp.pi, jnp.pi, self.num_adversaries, endpoint=False
            )
            adversary_init_pos_x_diff = agent_init_radius * jnp.cos(adversary_angles)
            adversary_init_pos_y_diff = agent_init_radius * jnp.sin(adversary_angles)
            adversary_init_pos = jnp.vstack(
                (
                    adversary_init_pos_x_diff + adversary_zone_init_pos[0],
                    adversary_init_pos_y_diff + adversary_zone_init_pos[1],
                )
            ).T

            # fixed obstacle locations between two zones
            between_zone_vec = adversary_zone_init_pos - agent_zone_init_pos
            vecs = jnp.vstack([between_zone_vec] * self.num_obstacles)
            # equidistance from both zones
            spacing = jnp.linspace(0.0, 1.0, self.num_obstacles + 2)[1:-1][:, None]
            obs_shifted_pos = agent_zone_init_pos + vecs * spacing

        # flag carrier is -1 if not picked up
        # agent integer id (opposite team) if picked up

        # create new state
        state = State(
            p_pos=jnp.vstack((agent_init_pos, adversary_init_pos)),
            p_vel=jnp.zeros((self.num_actors, 2)),
            flag_moving=jnp.full((2,), False),
            flag_carrier=jnp.full((2,), 0, dtype=int),
            flag_p_pos=jnp.vstack((agent_flag_init_pos, adversary_flag_init_pos)),
            flag_p_vel=jnp.zeros((2, 2)),
            done=jnp.full((self.num_actors), False),
            step=0,
            agent_zone=agent_zone_init_pos,
            adversary_zone=adversary_zone_init_pos,
            obs_pos=obs_shifted_pos,
            agent_names=self.good_agents,
            adversary_names=self.adversaries,
        )
        return self.obs_fn(state), state

    @partial(jax.jit, static_argnums=[0])
    def get_ctf_obs(self, state: State) -> Dict[str, chex.Array]:
        # observations:
        # [self_pos, self_vel, all_other_rel_pos, all_other_rel_vel,
        #  flag_rel_pos, obs_rel_pos, agent_zone_rel_pos, adv_zone_rel_pos, flag_moving]

        @partial(jax.vmap, in_axes=(0, None))
        def _common_stats(aidx, state: State):
            rel_pos = state.p_pos - state.p_pos[aidx]
            other_pos = jnp.roll(rel_pos, shift=self.num_agents - aidx - 1, axis=0)[
                : self.num_agents - 1
            ]
            rel_vel = state.p_vel - state.p_vel[aidx]
            other_vel = jnp.roll(rel_vel, shift=self.num_agents - aidx - 1, axis=0)[
                : self.num_agents - 1
            ]
            rel_obs_pos = state.obs_pos - state.p_pos[aidx]
            rel_flag_pos = state.flag_p_pos - state.p_pos[aidx]
            rel_agent_zone_pos = state.agent_zone - state.p_pos[aidx]
            rel_adv_zone_pos = state.adversary_zone - state.p_pos[aidx]
            return (
                rel_pos,
                other_pos,
                other_vel,
                rel_obs_pos,
                rel_flag_pos,
                rel_agent_zone_pos,
                rel_adv_zone_pos,
            )

        (
            rel_pos,
            other_pos,
            other_vel,
            rel_obs_pos,
            rel_flag_pos,
            rel_agent_zone_pos,
            rel_adv_zone_pos,
        ) = _common_stats(self.agent_range, state)

        def _obs(aidx, astr):
            ag_carrying_ind = (
                jax.lax.select(
                    state.flag_moving[1],  # 1: adversary flag moving
                    state.flag_carrier[0] == aidx,  # 0: flag carrier is this agent
                    False,
                )
                .astype(int)
                .reshape((1,))
            )
            adv_carrying_ind = (
                jax.lax.select(
                    state.flag_moving[0],  # 0: agent flag moving
                    state.flag_carrier[1]
                    == (aidx - self.num_good_agents),  # 1: flag carrier is this adv
                    False,
                )
                .astype(int)
                .reshape((1,))
            )
            # Build observation components conditionally
            if self.include_velocities:
                agent_obs_components = [
                    state.p_pos[aidx][None, :],
                    state.p_vel[aidx][None, :],
                    state.p_pos[: self.num_good_agents],
                    state.p_vel[: self.num_good_agents],
                    state.p_pos[self.num_good_agents :],
                    state.p_vel[self.num_good_agents :],
                    rel_obs_pos[aidx],
                    rel_agent_zone_pos[aidx][None, :],
                    rel_adv_zone_pos[aidx][None, :],
                ]
                adv_obs_components = [
                    state.p_pos[aidx][None, :],
                    state.p_vel[aidx][None, :],
                    state.p_pos[self.num_good_agents :],
                    state.p_vel[self.num_good_agents :],
                    state.p_pos[: self.num_good_agents],
                    state.p_vel[: self.num_good_agents],
                    rel_obs_pos[aidx],
                    rel_adv_zone_pos[aidx][None, :],
                    rel_agent_zone_pos[aidx][None, :],
                ]
            else:
                agent_obs_components = [
                    state.p_pos[aidx][None, :],
                    state.p_pos[: self.num_good_agents],
                    state.p_pos[self.num_good_agents :],
                    rel_obs_pos[aidx],
                    rel_agent_zone_pos[aidx][None, :],
                    rel_adv_zone_pos[aidx][None, :],
                ]
                adv_obs_components = [
                    state.p_pos[aidx][None, :],
                    state.p_pos[self.num_good_agents :],
                    state.p_pos[: self.num_good_agents],
                    rel_obs_pos[aidx],
                    rel_adv_zone_pos[aidx][None, :],
                    rel_agent_zone_pos[aidx][None, :],
                ]
            
            ret = jax.lax.select(
                ("agent" in astr),
                jnp.concatenate(
                    [
                        jnp.concatenate(agent_obs_components).flatten(),
                        ag_carrying_ind,
                        state.flag_p_pos[1].flatten(),
                        state.flag_p_pos[0].flatten(),
                    ]
                ),
                jnp.concatenate(
                    [
                        jnp.concatenate(adv_obs_components).flatten(),
                        adv_carrying_ind,
                        state.flag_p_pos[0].flatten(),
                        state.flag_p_pos[1].flatten(),
                    ]
                ),
            )

            return ret

        obs = {a: _obs(i, a) for i, a in enumerate(self.agents)}
        return obs

    @partial(jax.jit, static_argnums=[0])
    def get_ctf_obs_heading(self, state: State) -> Dict[str, chex.Array]:
        # observations:
        # [self_pos, self_vel, all_other_rel_pos, all_other_rel_vel,
        #  flag_rel_pos, obs_rel_pos, agent_zone_rel_pos, adv_zone_rel_pos, flag_moving]

        # NOTE:
        # Assume agents always faces (0, 0)

        @partial(jax.vmap, in_axes=(0, None))
        def _common_stats(aidx, state: State):
            heading_vec = -state.p_pos[aidx]
            theta = jnp.arctan2(heading_vec[1], heading_vec[0])
            sin = jnp.sin(theta)
            cos = jnp.cos(theta)
            rotmat = jnp.array([[cos, sin], [-sin, cos]])
            rel_pos_local = (state.p_pos - state.p_pos[aidx]) @ rotmat.T
            rel_obs_local = (state.obs_pos - state.p_pos[aidx]) @ rotmat.T
            rel_flag_local = (state.flag_p_pos - state.p_pos[aidx]) @ rotmat.T
            rel_agent_zone_local = (state.agent_zone - state.p_pos[aidx]) @ rotmat.T
            rel_adv_zone_local = (state.adversary_zone - state.p_pos[aidx]) @ rotmat.T
            rel_vel_local = (state.p_vel - state.p_vel[aidx]) @ rotmat.T

            # roll relative positions and velocities
            # other_pos = jnp.roll(rel_pos_local, shift=self.num_agents - aidx - 1, axis=0)[
            #     : self.num_agents - 1
            # ]
            # other_vel = jnp.roll(rel_vel_local, shift=self.num_agents - aidx - 1, axis=0)[
            #     : self.num_agents - 1
            # ]

            return (
                rel_pos_local,
                rel_vel_local,
                rel_obs_local,
                rel_flag_local,
                rel_agent_zone_local,
                rel_adv_zone_local,
            )

        (
            pos_local,
            vel_local,
            obs_local,
            flag_local,
            agent_zone_local,
            adv_zone_local,
        ) = _common_stats(self.agent_range, state)

        def _obs(aidx, astr):
            ag_carrying_ind = (
                jax.lax.select(
                    state.flag_moving[1],  # 1: adversary flag moving
                    state.flag_carrier[0] == aidx,  # 0: flag carrier is this agent
                    False,
                )
                .astype(int)
                .reshape((1,))
            )
            adv_carrying_ind = (
                jax.lax.select(
                    state.flag_moving[0],  # 0: agent flag moving
                    state.flag_carrier[1]
                    == (aidx - self.num_good_agents),  # 1: flag carrier is this adv
                    False,
                )
                .astype(int)
                .reshape((1,))
            )
            # Build observation components conditionally
            if self.include_velocities:
                agent_obs_components = [
                    state.p_vel[aidx][None, :],
                    jnp.roll(
                        pos_local[aidx, : self.num_good_agents, :],
                        shift=self.num_good_agents - aidx - 1,
                        axis=0,
                    )[: self.num_good_agents - 1],
                    jnp.roll(
                        vel_local[aidx, : self.num_good_agents, :],
                        shift=self.num_good_agents - aidx - 1,
                        axis=0,
                    )[: self.num_good_agents - 1],
                    pos_local[aidx, self.num_good_agents :, :],
                    vel_local[aidx, self.num_good_agents :, :],
                    obs_local[aidx],
                    flag_local[aidx, 1, :][None, :],
                    flag_local[aidx, 0, :][None, :],
                    agent_zone_local[aidx][None, :],
                    adv_zone_local[aidx][None, :],
                ]
                adv_obs_components = [
                    state.p_vel[aidx][None, :],
                    jnp.roll(
                        pos_local[aidx, self.num_good_agents :, :],
                        shift=self.num_adversaries - aidx - 1,
                        axis=0,
                    )[: self.num_adversaries - 1],
                    jnp.roll(
                        vel_local[aidx, self.num_good_agents :, :],
                        shift=self.num_adversaries - aidx - 1,
                        axis=0,
                    )[: self.num_adversaries - 1],
                    pos_local[aidx, : self.num_good_agents, :],
                    vel_local[aidx, : self.num_good_agents, :],
                    obs_local[aidx],
                    flag_local[aidx, 0, :][None, :],
                    flag_local[aidx, 1, :][None, :],
                    adv_zone_local[aidx][None, :],
                    agent_zone_local[aidx][None, :],
                ]
            else:
                agent_obs_components = [
                    jnp.roll(
                        pos_local[aidx, : self.num_good_agents, :],
                        shift=self.num_good_agents - aidx - 1,
                        axis=0,
                    )[: self.num_good_agents - 1],
                    pos_local[aidx, self.num_good_agents :, :],
                    obs_local[aidx],
                    flag_local[aidx, 1, :][None, :],
                    flag_local[aidx, 0, :][None, :],
                    agent_zone_local[aidx][None, :],
                    adv_zone_local[aidx][None, :],
                ]
                adv_obs_components = [
                    jnp.roll(
                        pos_local[aidx, self.num_good_agents :, :],
                        shift=self.num_adversaries - aidx - 1,
                        axis=0,
                    )[: self.num_adversaries - 1],
                    pos_local[aidx, : self.num_good_agents, :],
                    obs_local[aidx],
                    flag_local[aidx, 0, :][None, :],
                    flag_local[aidx, 1, :][None, :],
                    adv_zone_local[aidx][None, :],
                    agent_zone_local[aidx][None, :],
                ]
            
            ret = jax.lax.select(
                ("agent" in astr),
                jnp.concatenate(
                    [
                        jnp.concatenate(agent_obs_components).flatten(),
                        ag_carrying_ind,
                    ]
                ),
                jnp.concatenate(
                    [
                        jnp.concatenate(adv_obs_components).flatten(),
                        adv_carrying_ind,
                    ]
                ),
            )

            return ret

        obs = {a: _obs(i, a) for i, a in enumerate(self.agents)}
        return obs

    @partial(jax.jit, static_argnums=[0])
    def get_ctf_obs_central_abs(self, state: State) -> Dict[str, chex.Array]:
        # central absolute observations:
        # [all_pos, all_vel, flag_pos, obs_pos, agent_zone_pos, adv_zone_pos, flag_moving]

        def _obs(aidx, astr):
            components = [
                state.p_pos,
                state.flag_p_pos,
                state.obs_pos,
                state.agent_zone[None, :],
                state.adversary_zone[None, :],
                state.flag_moving[None, :],
            ]
            if self.include_velocities:
                components.insert(1, state.p_vel)
            ret = jnp.concatenate(components).flatten()
            return ret

        obs = {a: _obs(i, a) for i, a in enumerate(self.agents)}
        return obs

    @partial(jax.jit, static_argnums=[0])
    def get_rewards(self, state: State) -> Dict[str, float]:
        r = jnp.zeros((self.num_actors,))

        # reward for opp flag in own zone (team)
        r = r.at[: self.num_good_agents].add(
            jnp.linalg.norm(
                state.flag_p_pos[1] - state.agent_zone,
            )
            <= (self.zone_size + self.agent_size)
        )
        r = r.at[self.num_good_agents :].add(
            jnp.linalg.norm(
                state.flag_p_pos[0] - state.adversary_zone,
            )
            <= (self.zone_size + self.agent_size)
        )

        rew = {a: r[i] for i, a in enumerate(self.agents)}
        return rew

    @partial(jax.jit, static_argnums=[0])
    def get_zero_sum_rewards(self, state: State) -> Dict[str, float]:
        r = jnp.zeros((self.num_actors,))

        # reward for opp flag in own zone (team)
        r = r.at[: self.num_good_agents].add(
            jnp.linalg.norm(
                state.flag_p_pos[1] - state.agent_zone,
            )
            <= (self.zone_size - self.agent_size)
        )
        r = r.at[self.num_good_agents :].subtract(
            jnp.linalg.norm(
                state.flag_p_pos[1] - state.agent_zone,
            )
            <= (self.zone_size - self.agent_size)
        )
        r = r.at[self.num_good_agents :].add(
            jnp.linalg.norm(
                state.flag_p_pos[0] - state.adversary_zone,
            )
            <= (self.zone_size - self.agent_size)
        )
        r = r.at[: self.num_good_agents].subtract(
            jnp.linalg.norm(
                state.flag_p_pos[0] - state.adversary_zone,
            )
            <= (self.zone_size - self.agent_size)
        )

        rew = {a: r[i] for i, a in enumerate(self.agents)}
        return rew

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        # agents can pick up flags when they're entirely inside the flag zone
        # agents can drop flags when they're entirely inside their own zone

        # if flag inside opposite zone, reset to staring zones
        opp_flag_in = jnp.linalg.norm(
            state.flag_p_pos[1] - state.agent_zone,
        ) <= (self.zone_size - self.agent_size)
        ego_flag_in = jnp.linalg.norm(
            state.flag_p_pos[0] - state.adversary_zone,
        ) <= (self.zone_size - self.agent_size)
        reset_opp_flag = jax.lax.select(
            opp_flag_in,
            state.adversary_zone,
            state.flag_p_pos[1],
        )
        reset_ego_flag = jax.lax.select(
            ego_flag_in,
            state.agent_zone,
            state.flag_p_pos[0],
        )
        state = state.replace(flag_p_pos=jnp.vstack((reset_ego_flag, reset_opp_flag)))

        # check if carrying
        # length #agents #advs
        agent_carrying = jax.lax.select(
            state.flag_moving[1],  # 1: adversary flag moving
            jnp.full((self.num_good_agents,), False)
            .at[state.flag_carrier[0]]  # 0: flag carrier in agent team
            .set(True),
            jnp.full((self.num_good_agents,), False),
        )
        adv_carrying = jax.lax.select(
            state.flag_moving[0],  # 0: agent flag moving
            jnp.full((self.num_adversaries,), False)
            .at[state.flag_carrier[1]]  # 1: flag carrier in adversary team
            .set(True),
            jnp.full((self.num_adversaries,), False),
        )

        # drop only if close to own zone
        agent_can_drop = jnp.linalg.norm(
            state.p_pos[: self.num_good_agents] - state.agent_zone, axis=1
        ) <= (self.zone_size - self.agent_size)
        adv_can_drop = jnp.linalg.norm(
            state.p_pos[self.num_good_agents :] - state.adversary_zone, axis=1
        ) <= (self.zone_size - self.agent_size)

        # create mask if any drops
        agent_dropping_mask = jnp.logical_and(agent_carrying, agent_can_drop)
        adv_dropping_mask = jnp.logical_and(adv_carrying, adv_can_drop)

        # unset flag carrier if dropped
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
        # flag carrier shouldn't matter here

        # check if agents can pick up flags
        # masked with agent_dropping_mask
        agent_can_pickup = jnp.linalg.norm(
            state.p_pos[: self.num_good_agents] - state.adversary_zone, axis=1
        ) <= (self.zone_size - self.agent_size)
        adv_can_pickup = jnp.linalg.norm(
            state.p_pos[self.num_good_agents :] - state.agent_zone, axis=1
        ) <= (self.zone_size - self.agent_size)

        # set flag carrier if flag picked up, unset if dropped

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

        # actions.update(flag_actions_to_add)
        # step agents with concatenated actions (since flag treated as an particle)
        simple_obs, simple_state, simple_reward, simple_dones, simple_info = (
            super().step_env(key, simple_state, actions)
        )

        # split pos and vel back into separate entities
        # ordering: [agents, advs, flags, obs, zones]
        state = state.replace(
            p_pos=simple_state.p_pos[: self.num_actors, :],
            p_vel=simple_state.p_vel[: self.num_actors, :],
            done=simple_state.done,
            step=simple_state.step,
        )

        # set flag pos and vel to agent pos and vel if picked up
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

        state = state.replace(
            flag_p_pos=new_flag_p_pos,
            flag_p_vel=new_flag_p_vel,
        )

        # calculate rewards
        # rewards = jax.lax.select(
        #     self.zero_sum, self.get_zero_sum_rewards(state), self.get_rewards(state)
        # )

        rewards = jax.lax.cond(
            self.zero_sum, self.get_zero_sum_rewards, self.get_rewards, state
        )

        # get obs
        obs = self.obs_fn(state)

        # info
        info = {"adversary_dropped": ego_flag_in, "agent_dropped": opp_flag_in}

        return obs, state, rewards, simple_dones, info
