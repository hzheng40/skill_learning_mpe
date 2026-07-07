import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from functools import partial
from typing import Dict, List, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from ..spaces import Box, Discrete
from .default_params import CTF_MAX_STEPS, DISCRETE_ACT
from .simple import SimpleMPE
from .simple import State as SimpleState


@struct.dataclass
class State:
    """Area-denial state with fixed-shape MPE fields."""

    p_pos: chex.Array
    p_vel: chex.Array
    done: chex.Array
    step: int
    prev_p_pos: chex.Array
    area_pos: chex.Array
    control_progress: chex.Array
    prev_control_progress: chex.Array
    agent_spawn_center: chex.Array
    adversary_spawn_center: chex.Array
    obs_pos: chex.Array
    agent_names: List[str] = struct.field(pytree_node=False)
    adversary_names: List[str] = struct.field(pytree_node=False)
    option_assignment: chex.Array
    cum_subtask_obs: chex.Array


class SimpleAreaDenial(SimpleMPE):
    """Continuous MPE area-denial task.

    The agent team defends a protected area while adversaries try to occupy it
    and accumulate reversible capture progress. Rewards are dense and team
    shared; semantic subtask observations are for evaluation/wrappers only.
    """

    def __init__(
        self,
        *,
        num_good_agents: int = 3,
        num_adversaries: int = 3,
        num_obstacles: int = 0,
        action_type=DISCRETE_ACT,
        area_radius: float = 4.0,
        spawn_distance: float = 20.0,
        spawn_cluster_radius: float = 3.0,
        agent_size: float = 1.0,
        obstacle_size: float = 1.5,
        capture_rate: float = 0.04,
        recovery_rate: float = 0.04,
        terminal_bonus: float = 5.0,
        zero_sum: bool = True,
        random_start: bool = True,
        init_agent_everywhere: bool = False,
        num_skills: int = 10,
        random_skills: bool = False,
        assign_subtasks: bool = False,
        **_: object,
    ):
        self.num_good_agents = int(num_good_agents)
        self.num_adversaries = int(num_adversaries)
        self.num_actors = self.num_good_agents + self.num_adversaries
        self._user_num_obstacles = int(num_obstacles)
        self.num_obstacles = max(self._user_num_obstacles, 1)
        self.action_type = action_type
        self.area_radius = float(area_radius)
        self.zone_size = float(area_radius)
        self.spawn_distance = float(spawn_distance)
        self.spawn_cluster_radius = float(spawn_cluster_radius)
        self.agent_size = float(agent_size)
        self.obstacle_size = float(obstacle_size) if self._user_num_obstacles > 0 else 0.001
        self.capture_rate = float(capture_rate)
        self.recovery_rate = float(recovery_rate)
        self.terminal_bonus = float(terminal_bonus)
        self.zero_sum = bool(zero_sum)
        self.random_start = bool(random_start)
        self.init_agent_everywhere = bool(init_agent_everywhere)
        self.max_steps = CTF_MAX_STEPS

        self.num_skills = int(num_skills)
        self.random_skills = bool(random_skills)
        self.assign_subtasks = bool(assign_subtasks)
        self.num_subtasks = 10

        self.good_agents = [f"agent_{i}" for i in range(self.num_good_agents)]
        self.adversaries = [f"adversary_{i}" for i in range(self.num_adversaries)]
        agents = self.good_agents + self.adversaries
        landmarks = ["protected_area"] + [f"obstacle_{i}" for i in range(self.num_obstacles)]

        if action_type == DISCRETE_ACT:
            self.action_spaces = {agent: Discrete(5) for agent in agents}
        else:
            self.action_spaces = {agent: Box(-1, 1, (5,)) for agent in agents}

        obs_dim = (
            2
            + 2 * (self.num_good_agents - 1)
            + 2 * (self.num_good_agents - 1)
            + 2 * self.num_adversaries
            + 2 * self.num_adversaries
            + 2
            + 4
            + 2 * self._user_num_obstacles
        )
        self.observation_spaces = {
            agent: Box(-jnp.inf, jnp.inf, (obs_dim,)) for agent in agents
        }

        collides = jnp.concatenate(
            (
                jnp.full((self.num_actors,), True),
                jnp.full((1,), False),
                jnp.full((self.num_obstacles,), True),
            )
        )
        rad = jnp.concatenate(
            (
                jnp.full((self.num_actors,), self.agent_size),
                jnp.full((1,), self.area_radius),
                jnp.full((self.num_obstacles,), self.obstacle_size),
            )
        )
        mass = jnp.ones((self.num_actors + 1 + self.num_obstacles,))
        moveable = jnp.concatenate(
            (
                jnp.full((self.num_actors,), True),
                jnp.full((1 + self.num_obstacles,), False),
            )
        )
        max_speed = jnp.concatenate(
            (
                jnp.full((self.num_actors,), -1.0),
                jnp.full((1 + self.num_obstacles,), 0.0),
            )
        )

        super().__init__(
            num_agents=self.num_actors,
            agents=agents,
            landmarks=landmarks,
            num_landmarks=1 + self.num_obstacles,
            action_type=action_type,
            action_spaces=self.action_spaces,
            observation_spaces=self.observation_spaces,
            collide=collides,
            rad=rad,
            mass=mass,
            moveable=moveable,
            max_speed=max_speed,
            max_steps=self.max_steps,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        key, layout_key, agent_key, adv_key, obs_key, option_key = jax.random.split(key, 6)
        area_pos = jnp.zeros((2,), dtype=jnp.float32)

        if self.random_start:
            angle = jax.random.uniform(layout_key, (), minval=0.0, maxval=2 * jnp.pi)
            agent_spawn_center = area_pos
            adversary_spawn_center = self.spawn_distance * jnp.asarray(
                (jnp.cos(angle), jnp.sin(angle)), dtype=jnp.float32
            )
            if self.init_agent_everywhere:
                center_pos = area_pos
                agent_init_pos = self._random_disc_positions(
                    agent_key,
                    self.num_good_agents,
                    center_pos,
                    self.spawn_distance,
                )
                adversary_init_pos = self._random_disc_positions(
                    adv_key,
                    self.num_adversaries,
                    center_pos,
                    self.spawn_distance,
                )
            else:
                agent_init_pos = self._random_disc_positions(
                    agent_key,
                    self.num_good_agents,
                    area_pos,
                    self.area_radius * 0.5,
                )
                adversary_init_pos = self._random_box_positions(
                    adv_key,
                    self.num_adversaries,
                    adversary_spawn_center,
                    self.spawn_cluster_radius,
                )
            obs_pos = self._random_obstacle_positions(obs_key)
        else:
            agent_spawn_center = area_pos
            adversary_spawn_center = jnp.asarray((self.spawn_distance, 0.0), dtype=jnp.float32)
            agent_angles = jnp.linspace(0, 2 * jnp.pi, self.num_good_agents, endpoint=False)
            adv_angles = jnp.linspace(0, 2 * jnp.pi, self.num_adversaries, endpoint=False)
            agent_init_pos = area_pos + (self.area_radius * 0.45) * jnp.stack(
                (jnp.cos(agent_angles), jnp.sin(agent_angles)), axis=1
            )
            adversary_init_pos = adversary_spawn_center + self.spawn_cluster_radius * 0.5 * jnp.stack(
                (jnp.cos(adv_angles), jnp.sin(adv_angles)), axis=1
            )
            obs_pos = self._static_obstacle_positions()

        if self.random_skills:
            option_assignment = jax.random.dirichlet(
                option_key, jnp.ones((self.num_skills,)), shape=(self.num_actors,)
            )
        else:
            option_assignment = jax.random.randint(
                option_key, (self.num_actors,), minval=0, maxval=self.num_skills
            )

        p_pos = jnp.vstack((agent_init_pos, adversary_init_pos))
        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_actors, 2), dtype=jnp.float32),
            done=jnp.full((self.num_actors,), False),
            step=jnp.asarray(0, dtype=jnp.int32),
            prev_p_pos=p_pos,
            area_pos=area_pos,
            control_progress=jnp.asarray(0.0, dtype=jnp.float32),
            prev_control_progress=jnp.asarray(0.0, dtype=jnp.float32),
            agent_spawn_center=agent_spawn_center,
            adversary_spawn_center=adversary_spawn_center,
            obs_pos=obs_pos,
            agent_names=self.good_agents,
            adversary_names=self.adversaries,
            option_assignment=option_assignment,
            cum_subtask_obs=jnp.zeros((self.num_actors, self.num_subtasks), dtype=jnp.float32),
        )
        return self.obs_fn(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        prev_p_pos = state.p_pos
        prev_control_progress = state.control_progress

        simple_state = SimpleState(
            p_pos=jnp.vstack((state.p_pos, state.area_pos[None, :], state.obs_pos)),
            p_vel=jnp.vstack((state.p_vel, jnp.zeros((1 + self.num_obstacles, 2)))),
            done=state.done,
            step=state.step,
            goal=None,
            c=jnp.zeros((self.num_actors, self.dim_c)),
        )
        _, simple_state, _, simple_dones, _ = SimpleMPE.step_env(
            self, key, simple_state, actions
        )

        next_p_pos = simple_state.p_pos[: self.num_actors]
        next_p_vel = simple_state.p_vel[: self.num_actors]
        progress_delta = self._control_progress_delta(next_p_pos)
        control_progress = jnp.clip(prev_control_progress + progress_delta, 0.0, 1.0)
        captured = control_progress >= 1.0
        timed_out = simple_state.step >= self.max_steps
        done_all = captured | timed_out
        done = jnp.full((self.num_actors,), done_all)

        state = state.replace(
            p_pos=next_p_pos,
            p_vel=next_p_vel,
            done=done,
            step=simple_state.step,
            control_progress=control_progress,
            prev_control_progress=prev_control_progress,
        )
        subtask_obs = self.subtask_obs_fn(state, prev_p_pos, progress_delta)
        subtask_matrix = jnp.stack([subtask_obs[name] for name in self.agents], axis=0)
        cum_subtask_obs = state.cum_subtask_obs + subtask_matrix
        state = state.replace(cum_subtask_obs=cum_subtask_obs)

        rewards = self.get_rewards(
            prev_p_pos,
            state,
            progress_delta,
            captured=captured,
            timed_out=timed_out & ~captured,
        )
        dones = {
            agent: done[idx] | simple_dones[agent] for idx, agent in enumerate(self.agents)
        }
        dones["__all__"] = done_all | simple_dones["__all__"]
        normalized_cum = cum_subtask_obs / self.max_steps
        info = {
            "area_pos": state.area_pos,
            "control_progress": state.control_progress,
            "control_progress_delta": progress_delta,
            "defender_inside_count": jnp.sum(self._inside_area(state.p_pos[: self.num_good_agents])),
            "attacker_inside_count": jnp.sum(self._inside_area(state.p_pos[self.num_good_agents :])),
            "attacker_captured": captured,
            "defender_timed_out": timed_out & ~captured,
            "subtask_obs": subtask_obs,
            "cum_subtask_obs": {
                name: normalized_cum[idx] for idx, name in enumerate(self.agents)
            },
            "option_assignment": state.option_assignment,
        }

        if not self.random_skills and self.assign_subtasks:
            assigned_indices = state.option_assignment[:, None]
            valid_assign = state.option_assignment >= 0
            assigned_values = jnp.take_along_axis(
                subtask_matrix, assigned_indices, axis=1
            ).squeeze(axis=1)
            completion_mask = (assigned_values > 0.5) & valid_assign
            info["subtask_completed"] = completion_mask
            any_completed = jnp.any(completion_mask)
            state = state.replace(done=jnp.where(any_completed, jnp.ones_like(state.done), state.done))
            dones = {
                name: jnp.logical_or(done_flag, any_completed)
                for name, done_flag in dones.items()
            }

        state = state.replace(prev_p_pos=prev_p_pos)
        return self.obs_fn(state), state, rewards, dones, info

    @partial(jax.jit, static_argnums=(0,))
    def subtask_obs_fn(
        self,
        state: State,
        prev_p_pos: chex.Array,
        progress_delta: chex.Array,
    ) -> Dict[str, chex.Array]:
        dists = jnp.linalg.norm(state.p_pos - state.area_pos, axis=1)
        prev_dists = jnp.linalg.norm(prev_p_pos - state.area_pos, axis=1)
        inside = dists <= self.area_radius
        prev_inside = prev_dists <= self.area_radius
        entered = (~prev_inside) & inside
        exited = prev_inside & (~inside)
        near_boundary = jnp.abs(dists - self.area_radius) <= self.agent_size
        pair = jnp.linalg.norm(
            state.p_pos[:, None, :] - state.p_pos[None, :, :], axis=2
        )
        threshold = 2.0 * self.agent_size
        idx = jnp.arange(self.num_actors)
        is_defender = idx < self.num_good_agents
        same_team = is_defender[:, None] == is_defender[None, :]
        other_agent = ~jnp.eye(self.num_actors, dtype=bool)
        collide_teammate = jnp.any((pair <= threshold) & same_team & other_agent, axis=1)
        collide_opponent = jnp.any((pair <= threshold) & (~same_team), axis=1)
        opponent_inside = jnp.where(
            is_defender,
            jnp.any(inside[self.num_good_agents :]),
            jnp.any(inside[: self.num_good_agents]),
        )
        team_positive = jnp.where(is_defender, progress_delta < -1e-6, progress_delta > 1e-6)
        team_negative = jnp.where(is_defender, progress_delta > 1e-6, progress_delta < -1e-6)
        self_near_area = dists <= self.area_radius + 2.0 * self.agent_size
        blocking = self_near_area & opponent_inside & (collide_opponent | (jnp.min(jnp.where(~same_team, pair, jnp.inf), axis=1) <= 5.0))

        vectors = jnp.stack(
            (
                near_boundary,
                inside,
                entered,
                exited,
                opponent_inside,
                team_positive,
                team_negative,
                blocking,
                collide_teammate,
                collide_opponent,
            ),
            axis=1,
        ).astype(jnp.float32)
        return {agent: vectors[idx] for idx, agent in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=(0,))
    def obs_fn(self, state: State) -> Dict[str, chex.Array]:
        dists = jnp.linalg.norm(state.p_pos - state.area_pos, axis=1)
        inside = dists <= self.area_radius
        defender_inside_frac = jnp.mean(inside[: self.num_good_agents].astype(jnp.float32))
        attacker_inside_frac = jnp.mean(inside[self.num_good_agents :].astype(jnp.float32))
        out = {}
        for i, name in enumerate(self.agents):
            is_defender = i < self.num_good_agents
            local_idx = i if is_defender else i - self.num_good_agents
            own_start = 0 if is_defender else self.num_good_agents
            own_count = self.num_good_agents if is_defender else self.num_adversaries
            opp_start = self.num_good_agents if is_defender else 0
            opp_count = self.num_adversaries if is_defender else self.num_good_agents

            heading_vec = state.area_pos - state.p_pos[i]
            theta = jnp.arctan2(heading_vec[1], heading_vec[0])
            sin = jnp.sin(theta)
            cos = jnp.cos(theta)
            rot = jnp.asarray(((cos, sin), (-sin, cos)))
            self_vel = state.p_vel[i] @ rot.T
            own_pos = jnp.roll(
                state.p_pos[own_start : own_start + own_count] - state.p_pos[i],
                shift=own_count - local_idx - 1,
                axis=0,
            )[: own_count - 1] @ rot.T
            own_vel = jnp.roll(
                state.p_vel[own_start : own_start + own_count] - state.p_vel[i],
                shift=own_count - local_idx - 1,
                axis=0,
            )[: own_count - 1] @ rot.T
            opp_pos = (state.p_pos[opp_start : opp_start + opp_count] - state.p_pos[i]) @ rot.T
            opp_vel = (state.p_vel[opp_start : opp_start + opp_count] - state.p_vel[i]) @ rot.T
            area_rel = (state.area_pos - state.p_pos[i]) @ rot.T
            team_progress = jnp.where(
                is_defender,
                1.0 - state.control_progress,
                state.control_progress,
            )
            if self._user_num_obstacles > 0:
                obs_rel = ((state.obs_pos[: self._user_num_obstacles] - state.p_pos[i]) @ rot.T).flatten()
            else:
                obs_rel = jnp.zeros((0,), dtype=jnp.float32)
            out[name] = jnp.concatenate(
                (
                    self_vel.reshape(-1),
                    own_pos.reshape(-1),
                    own_vel.reshape(-1),
                    opp_pos.reshape(-1),
                    opp_vel.reshape(-1),
                    area_rel.reshape(-1),
                    jnp.asarray(
                        (
                            state.control_progress,
                            team_progress,
                            defender_inside_frac,
                            attacker_inside_frac,
                        ),
                        dtype=jnp.float32,
                    ),
                    obs_rel,
                ),
                axis=0,
            )
        return out

    @partial(jax.jit, static_argnums=(0,))
    def get_rewards(
        self,
        prev_p_pos: chex.Array,
        state: State,
        progress_delta: chex.Array,
        *,
        captured: chex.Array,
        timed_out: chex.Array,
    ) -> Dict[str, chex.Array]:
        prev_attacker_dist = jnp.mean(
            jnp.linalg.norm(prev_p_pos[self.num_good_agents :] - state.area_pos, axis=1)
        )
        attacker_dist = jnp.mean(
            jnp.linalg.norm(state.p_pos[self.num_good_agents :] - state.area_pos, axis=1)
        )
        attacker_distance_improvement = prev_attacker_dist - attacker_dist
        inside = self._inside_area(state.p_pos)
        defender_inside_frac = jnp.mean(inside[: self.num_good_agents].astype(jnp.float32))
        attacker_inside_frac = jnp.mean(inside[self.num_good_agents :].astype(jnp.float32))
        entered_attackers = jnp.mean(
            ((jnp.linalg.norm(prev_p_pos[self.num_good_agents :] - state.area_pos, axis=1) > self.area_radius)
             & inside[self.num_good_agents :]).astype(jnp.float32)
        )
        attacker_score = (
            5.0 * progress_delta
            + 0.05 * attacker_distance_improvement
            + 0.10 * attacker_inside_frac
            - 0.05 * defender_inside_frac
            + 0.10 * entered_attackers
            + self.terminal_bonus * captured.astype(jnp.float32)
            - self.terminal_bonus * timed_out.astype(jnp.float32)
        )
        defender_score = -attacker_score if self.zero_sum else (
            -5.0 * progress_delta
            - 0.05 * attacker_distance_improvement
            - 0.10 * attacker_inside_frac
            + 0.05 * defender_inside_frac
            + self.terminal_bonus * timed_out.astype(jnp.float32)
            - self.terminal_bonus * captured.astype(jnp.float32)
        )
        rewards = jnp.concatenate(
            (
                jnp.full((self.num_good_agents,), defender_score),
                jnp.full((self.num_adversaries,), attacker_score),
            )
        )
        return {agent: rewards[idx] for idx, agent in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=(0,))
    def _control_progress_delta(self, p_pos: chex.Array) -> chex.Array:
        inside = self._inside_area(p_pos)
        defender_pressure = jnp.mean(inside[: self.num_good_agents].astype(jnp.float32))
        attacker_pressure = jnp.mean(inside[self.num_good_agents :].astype(jnp.float32))
        net_pressure = attacker_pressure - defender_pressure
        return jnp.clip(
            self.capture_rate * net_pressure,
            -self.recovery_rate,
            self.capture_rate,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _inside_area(self, p_pos: chex.Array) -> chex.Array:
        return jnp.linalg.norm(p_pos - self._area_origin(), axis=-1) <= self.area_radius

    @partial(jax.jit, static_argnums=(0,))
    def _area_origin(self) -> chex.Array:
        return jnp.zeros((2,), dtype=jnp.float32)

    def _random_disc_positions(
        self,
        key: chex.PRNGKey,
        count: int,
        center: chex.Array,
        radius: float,
    ) -> chex.Array:
        radius_key, angle_key = jax.random.split(key)
        radii = jax.random.uniform(radius_key, (count, 1), minval=0.0, maxval=radius)
        angles = jax.random.uniform(angle_key, (count, 1), minval=0.0, maxval=2 * jnp.pi)
        return center + radii * jnp.hstack((jnp.cos(angles), jnp.sin(angles)))

    def _random_box_positions(
        self,
        key: chex.PRNGKey,
        count: int,
        center: chex.Array,
        radius: float,
    ) -> chex.Array:
        offsets = jax.random.uniform(key, (count, 2), minval=-radius, maxval=radius)
        return center + offsets

    def _random_obstacle_positions(self, key: chex.PRNGKey) -> chex.Array:
        if self._user_num_obstacles > 0:
            return jax.random.uniform(
                key,
                (self._user_num_obstacles, 2),
                minval=-self.spawn_distance / 3,
                maxval=self.spawn_distance / 3,
            )
        return jnp.asarray(((1e6, 1e6),), dtype=jnp.float32)

    def _static_obstacle_positions(self) -> chex.Array:
        if self._user_num_obstacles > 0:
            angles = jnp.linspace(0, 2 * jnp.pi, self._user_num_obstacles, endpoint=False)
            return (self.spawn_distance / 4.0) * jnp.stack(
                (jnp.cos(angles), jnp.sin(angles)), axis=1
            )
        return jnp.asarray(((1e6, 1e6),), dtype=jnp.float32)
