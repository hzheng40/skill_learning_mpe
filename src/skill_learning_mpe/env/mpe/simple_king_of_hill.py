import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import chex
from flax import struct
from functools import partial
from typing import Dict, Tuple, List

from .simple import SimpleMPE
from .simple import State as SimpleState
from .default_params import *

from ..spaces import Box, Discrete


@struct.dataclass
class State:
    """
    King of the Hill State.

    Two teams compete to capture and hold two "hills" (control points).
    Hills change ownership when touched by opposing team members.
    Agents can attack opponents with melee action, causing HP loss.
    Agents respawn far away after losing all HP.
    """

    # === Agent state ===
    p_pos: chex.Array  # (num_actors, 2) agent positions
    p_vel: chex.Array  # (num_actors, 2) agent velocities
    done: chex.Array  # (num_actors,) done flags
    step: int  # current step

    # === Hill state ===
    hill_pos: chex.Array  # (2, 2) positions of two hills
    hill_owner: chex.Array  # (2,) team index owning each hill: -1=none, 0=agents, 1=adversaries

    # === Combat state ===
    agent_hp: chex.Array  # (num_actors,) HP for each agent (0-max_hp)
    
    # === Spawn locations (for respawn) ===
    agent_spawn_center: chex.Array  # (2,) center of agent team spawn area
    adversary_spawn_center: chex.Array  # (2,) center of adversary team spawn area

    # === Static environment ===
    obs_pos: chex.Array  # (num_obstacles, 2) obstacle positions

    # === Agent metadata ===
    agent_names: List[str] = struct.field(pytree_node=False)
    adversary_names: List[str] = struct.field(pytree_node=False)

    # === Extended fields for skills ===
    prev_p_pos: chex.Array  # (num_actors, 2) previous positions
    prev_hill_owner: chex.Array  # (2,) previous hill ownership
    option_assignment: chex.Array  # skill assignments
    cum_subtask_obs: chex.Array  # (num_actors, num_subtasks) cumulative subtask observations


class SimpleKingOfHill(SimpleMPE):
    """
    King of the Hill Environment.

    Two teams compete to capture and hold control points (hills).
    A hill is captured when an agent from a team touches it.
    Hills can change ownership when touched by the opposing team.
    
    Combat Mechanics:
    - Agents can attack opponents within melee range using a discrete attack action
    - Each successful attack reduces opponent HP by 1
    - Agents respawn at their team's spawn point when HP reaches 0

    Key Mechanics:
    - Two hills spawn randomly in the world
    - Teams spawn clustered together on opposite sides
    - Points are awarded for holding hills
    - Combat allows disrupting opponent control

    Subtasks (6 total):
    0. near_own_hill - Agent is near a hill owned by their team
    1. near_enemy_hill - Agent is near a hill owned by enemy
    2. near_neutral_hill - Agent is near an unclaimed hill
    3. attacked_opponent - Agent performed melee attack on opponent
    4. collide_teammate - Agent collided with teammate
    5. collide_opponent - Agent collided with opponent

    Parameters
    ----------
    num_good_agents : int
        Number of agents on Team A (default 3)
    num_adversaries : int  
        Number of agents on Team B (default 3)
    num_obstacles : int
        Number of obstacles (default 0)
    hill_radius : float
        Radius of hill capture zones (default 2.0)
    melee_range : float
        Range for melee attacks (default 2.5)
    max_hp : int
        Maximum HP per agent (default 3)
    spawn_distance : float
        Distance between team spawn points (default 20.0)
    spawn_cluster_radius : float
        Radius of team spawn cluster (default 3.0)
    num_skills : int
        Number of discrete skills for option assignment (default 12)
    random_skills : bool
        Whether to use Dirichlet skill distributions (default False)
    """

    def __init__(
        self,
        *,
        num_good_agents: int = 3,
        num_adversaries: int = 3,
        num_obstacles: int = 0,
        action_type=DISCRETE_ACT,
        hill_radius: float = 2.0,
        melee_range: float = 2.5,
        max_hp: int = 3,
        spawn_distance: float = 20.0,
        spawn_cluster_radius: float = 3.0,
        agent_size: float = 1.0,
        obstacle_size: float = 1.5,
        zero_sum: bool = True,
        random_start: bool = True,
        init_agent_everywhere: bool = False,
        num_skills: int = 12,
        random_skills: bool = False,
        assign_subtasks: bool = False,
        **kwargs,
    ):
        self.hill_radius = hill_radius
        self.melee_range = melee_range
        self.max_hp = max_hp
        self.spawn_distance = spawn_distance
        self.spawn_cluster_radius = spawn_cluster_radius
        self.agent_size = agent_size
        self.obstacle_size = obstacle_size if num_obstacles > 0 else 0.001
        self.num_good_agents = num_good_agents
        self.num_adversaries = num_adversaries
        self._user_num_obstacles = num_obstacles
        self.num_obstacles = max(num_obstacles, 1)  # At least 1 for array operations
        self.action_type = action_type
        self.max_steps = CTF_MAX_STEPS
        self.zero_sum = zero_sum
        self.random_start = random_start
        self.init_agent_everywhere = init_agent_everywhere

        # Skill parameters
        self.num_skills = num_skills
        self.random_skills = random_skills
        self.assign_subtasks = assign_subtasks
        self.num_subtasks = 6  # Number of subtask observations

        # Agent names
        self.good_agents = [f"agent_{i}" for i in range(num_good_agents)]
        self.adversaries = [f"adversary_{i}" for i in range(num_adversaries)]
        agents = self.good_agents + self.adversaries

        # Landmarks: hills + obstacles
        obs = [f"obstacle_{i}" for i in range(self.num_obstacles)]
        hills = ["hill_0", "hill_1"]
        landmarks = hills + obs

        # Action space: 5 movement actions + 1 melee attack action
        if action_type == "Discrete":
            self.action_spaces = {i: Discrete(6) for i in agents}  # 0-4: movement, 5: melee
        elif action_type == "Continuous":
            self.action_spaces = {i: Box(-1, 1, (6,)) for i in agents}

        self.num_actors = num_good_agents + num_adversaries
        # Landmarks: 2 hills + obstacles
        num_landmarks = 2 + self.num_obstacles

        self.agent_range = jnp.arange(self.num_actors)

        # Collision setup: agents and obstacles collide; hills don't
        collides = jnp.concatenate(
            [
                jnp.full((self.num_actors,), True),   # agents collide
                jnp.full((2,), False),                 # hills don't collide
                jnp.full((self.num_obstacles,), True), # obstacles collide
            ]
        )

        # Radii for collisions
        rad = jnp.concatenate(
            [
                jnp.full((self.num_actors,), agent_size),
                jnp.full((2,), hill_radius),           # hill radius
                jnp.full((self.num_obstacles,), self.obstacle_size),
            ]
        )

        # Mass
        mass = jnp.concatenate(
            [
                jnp.full((self.num_actors,), 1.0),
                jnp.full((2,), 1.0),                   # hills
                jnp.full((self.num_obstacles,), 1.0),
            ]
        )

        # Moveable: only agents can move
        moveable = jnp.concatenate(
            [
                jnp.full((self.num_actors,), True),
                jnp.full((2,), False),                 # hills don't move
                jnp.full((self.num_obstacles,), False),
            ]
        )

        # Max speed
        max_speed = jnp.concatenate(
            [
                jnp.full((self.num_actors,), -1.0),
                jnp.full((2,), 0.0),
                jnp.full((self.num_obstacles,), 0.0),
            ]
        )

        # Observation dimensions
        base_obs_dim = (
            2  # self_vel
            + (self.num_actors - 1) * 4  # other agents pos+vel
            + 2 * 2  # hills rel pos
            + 2  # hill ownership (one-hot per hill: -1/0/1 -> 3 states, simplified to owner idx)
            + self._user_num_obstacles * 2  # obstacles rel pos
            + 1  # own HP normalized
            + self.num_actors  # all agents HP normalized
        )
        option_obs_dim = 6  # additional option observations

        self.observation_spaces = {
            i: Box(-jnp.inf, jnp.inf, (base_obs_dim + option_obs_dim,)) for i in agents
        }

        if self.assign_subtasks:
            assert self.num_skills == 12, "Number of skills must be 12 when assigning subtasks"
            assert not self.random_skills, "Random skills must be False when assigning subtasks"

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
            mass=mass,
            moveable=moveable,
            max_speed=max_speed,
            max_steps=self.max_steps,
        )

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        key, agent_key, adv_key, hill_key, obs_key, option_key = jax.random.split(key, 6)

        if self.random_start:
            # Random spawn centers for teams
            spawn_key, key = jax.random.split(key)
            angle = jax.random.uniform(spawn_key, (), minval=0, maxval=2 * jnp.pi)
            
            agent_spawn_center = jnp.array([
                self.spawn_distance / 2 * jnp.cos(angle),
                self.spawn_distance / 2 * jnp.sin(angle)
            ])
            adversary_spawn_center = jnp.array([
                -self.spawn_distance / 2 * jnp.cos(angle),
                -self.spawn_distance / 2 * jnp.sin(angle)
            ])

            if self.init_agent_everywhere:
                # Initialize anywhere in the map, inside the circle that encloses both spawn centers
                agent_everywhere_radius = self.spawn_distance
                center_pos = (agent_spawn_center + adversary_spawn_center) / 2
                # center around map center, radius from agent_size to agent_everywhere_radius, random angle
                ag_r_key, ag_a_key = jax.random.split(agent_key)
                agent_init_radius = jax.random.uniform(
                    ag_r_key,
                    (self.num_good_agents, 1),
                    minval=self.agent_size,
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
                    + center_pos
                )
                adv_r_key, adv_a_key = jax.random.split(adv_key)
                adversary_init_radius = jax.random.uniform(
                    adv_r_key,
                    (self.num_adversaries, 1),
                    minval=self.agent_size,
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
                    + center_pos
                )
            else:
                # Spawn agents clustered around their spawn centers
                agent_offsets = jax.random.uniform(
                    agent_key, (self.num_good_agents, 2),
                    minval=-self.spawn_cluster_radius, maxval=self.spawn_cluster_radius
                )
                agent_init_pos = agent_spawn_center + agent_offsets

                adv_offsets = jax.random.uniform(
                    adv_key, (self.num_adversaries, 2),
                    minval=-self.spawn_cluster_radius, maxval=self.spawn_cluster_radius
                )
                adversary_init_pos = adversary_spawn_center + adv_offsets

            # Random hill positions (between the two spawn areas)
            hill_key1, hill_key2 = jax.random.split(hill_key)
            hill_pos = jax.random.uniform(
                hill_key1, (2, 2),
                minval=-self.spawn_distance / 3, maxval=self.spawn_distance / 3
            )

            # Obstacles
            if self._user_num_obstacles > 0:
                obs_pos = jax.random.uniform(
                    obs_key, (self._user_num_obstacles, 2),
                    minval=-self.spawn_distance / 2, maxval=self.spawn_distance / 2
                )
            else:
                obs_pos = jnp.array([[1e6, 1e6]])
        else:
            # Fixed positions
            agent_spawn_center = jnp.array([-self.spawn_distance / 2, 0.0])
            adversary_spawn_center = jnp.array([self.spawn_distance / 2, 0.0])

            # Fixed agent positions in a cluster
            agent_angles = jnp.linspace(0, 2 * jnp.pi, self.num_good_agents, endpoint=False)
            agent_init_pos = agent_spawn_center + self.spawn_cluster_radius * jnp.stack(
                [jnp.cos(agent_angles), jnp.sin(agent_angles)], axis=1
            ) * 0.5

            adv_angles = jnp.linspace(0, 2 * jnp.pi, self.num_adversaries, endpoint=False)
            adversary_init_pos = adversary_spawn_center + self.spawn_cluster_radius * jnp.stack(
                [jnp.cos(adv_angles), jnp.sin(adv_angles)], axis=1
            ) * 0.5

            # Fixed hill positions
            hill_pos = jnp.array([
                [-self.spawn_distance / 6, self.spawn_distance / 6],
                [self.spawn_distance / 6, -self.spawn_distance / 6]
            ])

            if self._user_num_obstacles > 0:
                obs_angles = jnp.linspace(0, 2 * jnp.pi, self._user_num_obstacles, endpoint=False)
                obs_pos = self.spawn_distance / 4 * jnp.stack(
                    [jnp.cos(obs_angles), jnp.sin(obs_angles)], axis=1
                )
            else:
                obs_pos = jnp.array([[1e6, 1e6]])

        # Option assignment
        if self.random_skills:
            alpha = jnp.ones(self.num_skills)
            option_assignment = jax.random.dirichlet(
                option_key, alpha, shape=(self.num_actors,)
            )
        else:
            option_assignment = jax.random.randint(
                option_key, (self.num_actors,), minval=0, maxval=self.num_skills
            )

        # Initialize cumulative subtask obs
        cum_subtask_obs = jnp.zeros((self.num_actors, self.num_subtasks), dtype=jnp.float32)

        p_pos = jnp.vstack((agent_init_pos, adversary_init_pos))

        # Hills start unowned (-1)
        hill_owner = jnp.full((2,), -1, dtype=jnp.int32)

        # All agents start with full HP
        agent_hp = jnp.full((self.num_actors,), self.max_hp, dtype=jnp.int32)

        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_actors, 2)),
            done=jnp.full((self.num_actors,), False),
            step=0,
            hill_pos=hill_pos,
            hill_owner=hill_owner,
            agent_hp=agent_hp,
            agent_spawn_center=agent_spawn_center,
            adversary_spawn_center=adversary_spawn_center,
            obs_pos=obs_pos,
            agent_names=self.good_agents,
            adversary_names=self.adversaries,
            prev_p_pos=p_pos,
            prev_hill_owner=hill_owner,
            option_assignment=option_assignment,
            cum_subtask_obs=cum_subtask_obs,
        )

        return self.obs_fn(state), state

    @partial(jax.jit, static_argnums=[0])
    def _get_melee_actions(self, actions: dict) -> chex.Array:
        """Extract melee attack actions from action dict.
        
        Returns (num_actors,) boolean array indicating who is attacking.
        """
        melee_actions = jnp.array([
            actions[name] == 5 if self.action_type == "Discrete" else actions[name][5] > 0.5
            for name in self.agents
        ])
        return melee_actions

    @partial(jax.jit, static_argnums=[0])
    def _process_combat(
        self, 
        key: chex.PRNGKey,
        state: State, 
        melee_actions: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Process melee combat.
        
        Returns:
            new_hp: Updated HP for all agents
            attacked_mask: (num_actors,) which agents successfully attacked
            got_attacked_mask: (num_actors,) which agents got hit
        """
        # Compute pairwise distances
        pos = state.p_pos
        dists = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        
        # Check who is in melee range of opponents
        in_range = dists <= self.melee_range
        
        # Create team masks
        is_agent_team = jnp.arange(self.num_actors) < self.num_good_agents
        
        # For each attacker, find valid targets (opponents in range)
        # Agents can only attack adversaries and vice versa
        valid_targets = jnp.zeros((self.num_actors, self.num_actors), dtype=bool)
        for i in range(self.num_actors):
            for j in range(self.num_actors):
                # i attacks j if: i is attacking, j is in range, and they're on different teams
                is_opponent = is_agent_team[i] != is_agent_team[j]
                valid_targets = valid_targets.at[i, j].set(
                    melee_actions[i] & in_range[i, j] & is_opponent
                )
        
        # Count attacks on each agent
        attacks_received = jnp.sum(valid_targets, axis=0)
        
        # Track who attacked successfully (attacked at least one opponent)
        attacked_mask = jnp.any(valid_targets, axis=1)
        
        # Track who got attacked
        got_attacked_mask = attacks_received > 0
        
        # Reduce HP based on attacks received
        new_hp = state.agent_hp - attacks_received
        new_hp = jnp.maximum(new_hp, 0)
        
        return new_hp, attacked_mask, got_attacked_mask

    @partial(jax.jit, static_argnums=[0])
    def _respawn_agents(
        self, 
        key: chex.PRNGKey,
        state: State, 
        new_hp: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Respawn agents with 0 HP at their team's spawn point.
        
        Returns:
            new_pos: Updated positions
            new_hp: Reset HP for respawned agents
            respawned_mask: (num_actors,) which agents respawned
        """
        needs_respawn = new_hp <= 0
        
        # Generate random offsets for respawn positions
        respawn_offsets = jax.random.uniform(
            key, (self.num_actors, 2),
            minval=-self.spawn_cluster_radius, maxval=self.spawn_cluster_radius
        )
        
        # Determine respawn positions based on team
        is_agent_team = jnp.arange(self.num_actors) < self.num_good_agents
        respawn_centers = jnp.where(
            is_agent_team[:, None],
            state.agent_spawn_center[None, :],
            state.adversary_spawn_center[None, :]
        )
        respawn_pos = respawn_centers + respawn_offsets
        
        # Apply respawn
        new_pos = jnp.where(needs_respawn[:, None], respawn_pos, state.p_pos)
        new_hp = jnp.where(needs_respawn, self.max_hp, new_hp)
        
        return new_pos, new_hp, needs_respawn

    @partial(jax.jit, static_argnums=[0])
    def _update_hill_ownership(self, state: State) -> chex.Array:
        """Update hill ownership based on agent proximity.
        
        A hill is captured when an agent touches it.
        Returns updated hill_owner array.
        """
        new_owner = state.hill_owner.copy()
        
        for hill_idx in range(2):
            hill_pos = state.hill_pos[hill_idx]
            
            # Check which agents are touching this hill
            dists = jnp.linalg.norm(state.p_pos - hill_pos, axis=1)
            touching = dists <= self.hill_radius + self.agent_size
            
            # Check if any agent from each team is touching
            agent_touching = jnp.any(touching[:self.num_good_agents])
            adv_touching = jnp.any(touching[self.num_good_agents:])
            
            # Priority: if both teams touch simultaneously, no change (contested)
            # Otherwise, the touching team captures
            new_owner = new_owner.at[hill_idx].set(
                jax.lax.select(
                    agent_touching & ~adv_touching,
                    0,  # Agents capture
                    jax.lax.select(
                        adv_touching & ~agent_touching,
                        1,  # Adversaries capture
                        new_owner[hill_idx]  # No change (contested or no one touching)
                    )
                )
            )
        
        return new_owner

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        # Store previous state for subtask detection
        prev_p_pos = state.p_pos
        prev_hill_owner = state.hill_owner

        # Extract melee actions
        melee_actions = self._get_melee_actions(actions)

        # Create simple state for physics (without melee action)
        simple_state = SimpleState(
            p_pos=jnp.vstack(
                (state.p_pos, state.hill_pos, state.obs_pos)
            ),
            p_vel=jnp.vstack(
                (state.p_vel, jnp.zeros((2 + self.num_obstacles, 2)))
            ),
            done=state.done,
            step=state.step,
            goal=None,
            c=jnp.zeros((self.num_actors, self.dim_c)),
        )

        # Mask melee action (5) to no-op (0) for physics step
        physics_actions = {
            name: jax.lax.select(actions[name] == 5, 0, actions[name])
            if self.action_type == "Discrete"
            else actions[name][:5]  # Only use first 5 for continuous
            for name in self.agents
        }

        # Step physics
        simple_obs, simple_state, simple_reward, simple_dones, simple_info = (
            SimpleMPE.step_env(self, key, simple_state, physics_actions)
        )

        # Extract updated positions
        new_p_pos = simple_state.p_pos[:self.num_actors, :]
        new_p_vel = simple_state.p_vel[:self.num_actors, :]

        # Update state with physics results
        state = state.replace(
            p_pos=new_p_pos,
            p_vel=new_p_vel,
            done=simple_state.done,
            step=simple_state.step,
        )

        # Process combat
        key, combat_key = jax.random.split(key)
        new_hp, attacked_mask, got_attacked_mask = self._process_combat(
            combat_key, state, melee_actions
        )

        # Respawn agents with 0 HP
        key, respawn_key = jax.random.split(key)
        new_pos, new_hp, respawned_mask = self._respawn_agents(
            respawn_key, state, new_hp
        )
        
        state = state.replace(
            p_pos=new_pos,
            agent_hp=new_hp,
        )

        # Update hill ownership
        new_hill_owner = self._update_hill_ownership(state)
        state = state.replace(hill_owner=new_hill_owner)

        # Calculate rewards
        rewards = self.get_rewards(state, prev_hill_owner)

        # Compute subtask observations
        subtask_obs = self.subtask_obs_fn(
            state,
            prev_hill_owner,
            attacked_mask,
            got_attacked_mask,
            respawned_mask,
        )

        # Update cumulative subtask observations
        subtask_matrix = jnp.stack(
            [subtask_obs[name] for name in self.agents], axis=0
        )
        new_cum_subtask_obs = state.cum_subtask_obs + subtask_matrix

        state = state.replace(cum_subtask_obs=new_cum_subtask_obs)

        # Normalize cumulative subtask obs
        normalized_cum_subtask_obs = new_cum_subtask_obs / self.max_steps

        cum_subtask_obs_dict = {
            name: normalized_cum_subtask_obs[i] for i, name in enumerate(self.agents)
        }

        info = {
            "hill_owner": state.hill_owner,
            "agent_hp": state.agent_hp,
            "attacked": attacked_mask,
            "got_attacked": got_attacked_mask,
            "respawned": respawned_mask,
            "subtask_obs": subtask_obs,
            "cum_subtask_obs": cum_subtask_obs_dict,
            "option_assignment": state.option_assignment,
        }

        if not self.random_skills and self.assign_subtasks:
            assigned_indices = state.option_assignment[:, None]
            valid_assign = state.option_assignment >= 0
            assigned_values = jnp.take_along_axis(
                subtask_matrix, assigned_indices, axis=1
            ).squeeze(axis=1)
            completion_mask = jnp.logical_and(assigned_values > 0.5, valid_assign)
            info["subtask_completed"] = completion_mask

            any_completed = jnp.any(completion_mask)
            state = state.replace(
                done=jnp.where(any_completed, jnp.ones_like(state.done), state.done)
            )
            simple_dones = {
                name: jnp.logical_or(done_flag, any_completed)
                for name, done_flag in simple_dones.items()
            }

        # Update previous state
        state = state.replace(
            prev_p_pos=prev_p_pos,
            prev_hill_owner=prev_hill_owner,
        )

        obs = self.obs_fn(state)

        return obs, state, rewards, simple_dones, info

    @partial(jax.jit, static_argnums=[0])
    def subtask_obs_fn(
        self,
        state: State,
        prev_hill_owner: chex.Array,
        attacked_mask: chex.Array,
        got_attacked_mask: chex.Array,
        respawned_mask: chex.Array,
    ) -> Dict[str, chex.Array]:
        """
        Subtask observations for all agents.

        Subtasks (6 total):
        0. near_own_hill - Agent is near a hill owned by their team
        1. near_enemy_hill - Agent is near a hill owned by enemy
        2. near_neutral_hill - Agent is near an unclaimed hill
        3. attacked_opponent - Agent performed melee attack on opponent
        4. collide_teammate - Agent collided with teammate
        5. collide_opponent - Agent collided with opponent
        """
        # Precompute distances from all agents to all hills: (num_actors, num_hills)
        all_dists_to_hills = jnp.linalg.norm(
            state.p_pos[:, None, :] - state.hill_pos[None, :, :], axis=2
        )
        # Which agents are near which hills: (num_actors, num_hills)
        near_each_hill = all_dists_to_hills <= self.hill_radius + self.agent_size

        # Precompute all pairwise distances for collisions: (num_actors, num_actors)
        pairwise_dists = jnp.linalg.norm(
            state.p_pos[:, None, :] - state.p_pos[None, :, :], axis=2
        )
        collision_threshold = 2 * self.agent_size

        out = {}
        for i, name in enumerate(self.agents):
            is_agent = i < self.num_good_agents
            idx = i if is_agent else i - self.num_good_agents

            # This agent's nearness to each hill
            agent_near_hill = near_each_hill[i]  # (num_hills,)

            # Hill ownership from this agent's perspective (Python ints, not traced)
            if is_agent:
                own_team = 0
                enemy_team = 1
                # Teammate distances (exclude self)
                teammate_dists = jnp.concatenate([
                    pairwise_dists[i, :idx],
                    pairwise_dists[i, idx+1:self.num_good_agents]
                ])
                opponent_dists = pairwise_dists[i, self.num_good_agents:]
            else:
                own_team = 1
                enemy_team = 0
                # Teammate distances (exclude self)
                adv_start = self.num_good_agents
                teammate_dists = jnp.concatenate([
                    pairwise_dists[i, adv_start:i],
                    pairwise_dists[i, i+1:]
                ])
                opponent_dists = pairwise_dists[i, :self.num_good_agents]

            near_own_hill = jnp.any(agent_near_hill & (state.hill_owner == own_team))
            near_enemy_hill = jnp.any(agent_near_hill & (state.hill_owner == enemy_team))
            near_neutral_hill = jnp.any(agent_near_hill & (state.hill_owner == -1))

            # Collisions
            collide_teammate = jnp.where(
                teammate_dists.size > 0,
                jnp.any(teammate_dists <= collision_threshold),
                False
            )
            collide_opponent = jnp.any(opponent_dists <= collision_threshold)

            obs_vec = jnp.array(
                [
                    near_own_hill,        # 0: near_own_hill
                    near_enemy_hill,      # 1: near_enemy_hill
                    near_neutral_hill,    # 2: near_neutral_hill
                    attacked_mask[i],     # 3: attacked_opponent
                    collide_teammate,     # 4: collide_teammate
                    collide_opponent,     # 5: collide_opponent
                ],
                dtype=jnp.float32,
            )
            out[name] = obs_vec

        return out

    @partial(jax.jit, static_argnums=[0])
    def obs_fn(self, state: State) -> Dict[str, chex.Array]:
        """Generate observations for all agents."""
        out = {}

        for i, name in enumerate(self.agents):
            is_agent = i < self.num_good_agents
            idx = i if is_agent else i - self.num_good_agents

            # Compute rotation matrix (agents face toward origin)
            heading_vec = -state.p_pos[i]
            theta = jnp.arctan2(heading_vec[1], heading_vec[0])
            sin = jnp.sin(theta)
            cos = jnp.cos(theta)
            rotmat = jnp.array([[cos, sin], [-sin, cos]])

            # Self velocity in local frame
            self_vel = (state.p_vel[i] @ rotmat.T)

            # Other agents (teammates first, then opponents)
            if is_agent:
                teammate_idx = jnp.concatenate([
                    jnp.arange(idx),
                    jnp.arange(idx+1, self.num_good_agents)
                ])
                opponent_idx = jnp.arange(self.num_good_agents, self.num_actors)
            else:
                teammate_idx = jnp.concatenate([
                    jnp.arange(self.num_good_agents, i),
                    jnp.arange(i+1, self.num_actors)
                ])
                opponent_idx = jnp.arange(self.num_good_agents)

            # All other agents (teammates + opponents)
            other_idx = jnp.concatenate([teammate_idx, opponent_idx])
            other_pos = state.p_pos[other_idx]
            other_vel = state.p_vel[other_idx]

            other_rel_pos = ((other_pos - state.p_pos[i]) @ rotmat.T).flatten()
            other_rel_vel = ((other_vel - state.p_vel[i]) @ rotmat.T).flatten()

            # Hills
            hills_rel_pos = ((state.hill_pos - state.p_pos[i]) @ rotmat.T).flatten()
            
            # Hill ownership (encoded as: -1 -> [0,0], 0 -> [1,0], 1 -> [0,1] for each hill)
            # Simplified: just use the owner index normalized
            hill_owner_obs = (state.hill_owner + 1) / 2.0  # Map -1,0,1 to 0,0.5,1

            # Obstacles
            if self._user_num_obstacles > 0:
                obs_rel_pos = ((state.obs_pos[:self._user_num_obstacles] - state.p_pos[i]) @ rotmat.T).flatten()
            else:
                obs_rel_pos = jnp.array([])

            # HP
            own_hp = state.agent_hp[i] / self.max_hp
            all_hp = state.agent_hp / self.max_hp

            # Option observations
            num_own_hills = jnp.sum(state.hill_owner == (0 if is_agent else 1))
            num_enemy_hills = jnp.sum(state.hill_owner == (1 if is_agent else 0))
            dist_to_nearest_hill = jnp.min(jnp.linalg.norm(state.p_pos[i] - state.hill_pos, axis=1))
            
            option_obs = jnp.array([
                num_own_hills / 2.0,
                num_enemy_hills / 2.0,
                dist_to_nearest_hill / self.spawn_distance,
                own_hp,
                jnp.mean(state.agent_hp[:self.num_good_agents] if is_agent else state.agent_hp[self.num_good_agents:]) / self.max_hp,
                jnp.mean(state.agent_hp[self.num_good_agents:] if is_agent else state.agent_hp[:self.num_good_agents]) / self.max_hp,
            ])

            obs = jnp.concatenate([
                self_vel,
                other_rel_pos,
                other_rel_vel,
                hills_rel_pos,
                hill_owner_obs,
                obs_rel_pos,
                jnp.array([own_hp]),
                all_hp,
                option_obs,
            ])

            out[name] = obs

        return out

    @partial(jax.jit, static_argnums=[0])
    def get_rewards(self, state: State, prev_hill_owner: chex.Array) -> Dict[str, float]:
        """
        Compute rewards for all agents.

        Reward structure:
        - +1 for each hill owned by team
        - +0.5 bonus for capturing a hill
        - +0.1 for being near a hill
        """
        r = jnp.zeros((self.num_actors,))

        # Points for holding hills
        agent_hills = jnp.sum(state.hill_owner == 0)
        adv_hills = jnp.sum(state.hill_owner == 1)

        r = r.at[:self.num_good_agents].add(agent_hills * 0.1)
        r = r.at[self.num_good_agents:].add(adv_hills * 0.1)

        # Bonus for capturing
        agent_captured = jnp.sum((state.hill_owner == 0) & (prev_hill_owner != 0))
        adv_captured = jnp.sum((state.hill_owner == 1) & (prev_hill_owner != 1))

        r = r.at[:self.num_good_agents].add(agent_captured * 0.5)
        r = r.at[self.num_good_agents:].add(adv_captured * 0.5)

        if self.zero_sum:
            r = r.at[:self.num_good_agents].subtract(adv_hills * 0.1)
            r = r.at[:self.num_good_agents].subtract(adv_captured * 0.5)
            r = r.at[self.num_good_agents:].subtract(agent_hills * 0.1)
            r = r.at[self.num_good_agents:].subtract(agent_captured * 0.5)

        return {a: r[i] for i, a in enumerate(self.agents)}

