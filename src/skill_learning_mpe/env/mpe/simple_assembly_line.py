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

from ..spaces import Box


@struct.dataclass
class State:
    """
    Assembly Line State.

    Two teams compete to assemble products by collecting parts from two rooms
    and delivering them to a central assembler. Parts must be collected in order
    (A first, then B). Once both parts are delivered, a team member must press
    and hold a button for T steps to complete assembly.
    
    Only one team can be assembling at a time (first to deliver parts locks assembler).
    """

    # === Agent state ===
    p_pos: chex.Array  # (num_actors, 2) agent positions
    p_vel: chex.Array  # (num_actors, 2) agent velocities
    done: chex.Array  # (num_actors,) done flags
    step: int  # current step

    # === Room positions ===
    part_a_room_pos: chex.Array  # (2,) center of part A room
    part_b_room_pos: chex.Array  # (2,) center of part B room
    
    # === Assembler state ===
    assembler_pos: chex.Array  # (2,) center of assembler zone
    assembly_button_pos: chex.Array  # (2,) assembly button position
    
    # === Agent inventory ===
    carrying_part_a: chex.Array  # (num_actors,) boolean - who's carrying part A
    carrying_part_b: chex.Array  # (num_actors,) boolean - who's carrying part B
    
    # === Assembly progress ===
    # Parts delivered to assembler: (2, 2) [team_idx, part_type]
    # team 0 = agents, team 1 = adversaries
    # part_type 0 = A, part_type 1 = B
    parts_delivered: chex.Array  # (2, 2) int count of parts delivered per team
    
    # Which team has locked the assembler (-1 = none, 0 = agents, 1 = adversaries)
    assembler_locked_by: chex.Array  # scalar int
    
    # Assembly button hold progress per team: (2,) steps held
    assembly_hold_progress: chex.Array  # (2,) int
    
    # Score per team: (2,) number of completed assemblies
    team_scores: chex.Array  # (2,) int
    
    # === Spawn centers ===
    agent_spawn_center: chex.Array  # (2,)
    adversary_spawn_center: chex.Array  # (2,)
    
    # === Agent metadata ===
    agent_names: List[str] = struct.field(pytree_node=False)
    adversary_names: List[str] = struct.field(pytree_node=False)

    # === Extended fields for skills ===
    prev_p_pos: chex.Array  # (num_actors, 2) previous positions
    option_assignment: chex.Array  # skill assignments
    cum_subtask_obs: chex.Array  # (num_actors, num_subtasks) cumulative subtask observations


class SimpleAssemblyLine(SimpleMPE):
    """
    Assembly Line Environment.

    Two teams compete to assemble products. The assembly process requires:
    1. Pick up Part A from room A (can only pick up if not carrying anything)
    2. Pick up Part B from room B (can only pick up if already carrying Part A)
    3. Deliver parts to the central assembler zone
    4. Once both parts delivered, press and hold the assembly button for T steps

    Key Mechanics:
    - Parts spawn infinitely in their respective rooms
    - Agents must pick up parts in order (A first, then B)
    - Parts are delivered by entering the assembler zone while carrying them
    - Assembler is locked to first team that delivers a part
    - Only one team can finish assembly at a time
    - Completing assembly scores a point and resets the assembler

    Subtasks (8 total):
    0. near_part_a_room - Agent is near part A room
    1. near_part_b_room - Agent is near part B room
    2. carrying_part_a - Agent is carrying part A
    3. carrying_part_b - Agent is carrying part B (which means has both)
    4. at_assembler - Agent is at assembler zone
    5. pressing_button - Agent is pressing assembly button (stationary + in range)
    6. collide_teammate - Agent collided with teammate
    7. collide_opponent - Agent collided with opponent

    Parameters
    ----------
    num_good_agents : int
        Number of agents on Team A (default 3)
    num_adversaries : int  
        Number of agents on Team B (default 3)
    room_radius : float
        Radius of part rooms (default 3.0)
    assembler_radius : float
        Radius of assembler zone (default 4.0)
    button_radius : float
        Distance threshold for pressing the button (default 1.5)
    vel_eps : float
        Velocity magnitude considered stationary for button press (default 2.0)
    assembly_hold_time : int
        Number of steps to hold button to complete assembly (default 30)
    spawn_distance : float
        Distance between team spawn points (default 25.0)
    spawn_cluster_radius : float
        Radius of team spawn cluster (default 3.0)
    num_skills : int
        Number of discrete skills for option assignment (default 8)
    random_skills : bool
        Whether to use Dirichlet skill distributions (default False)
    assign_subtasks : bool
        Whether to assign specific subtasks and terminate on completion (default False)
    """

    def __init__(
        self,
        *,
        num_good_agents: int = 3,
        num_adversaries: int = 3,
        action_type=DISCRETE_ACT,
        room_radius: float = 3.0,
        assembler_radius: float = 4.0,
        button_radius: float = 1.5,
        vel_eps: float = 2.0,
        assembly_hold_time: int = 30,
        spawn_distance: float = 25.0,
        spawn_cluster_radius: float = 3.0,
        agent_size: float = 1.0,
        zero_sum: bool = True,
        random_start: bool = True,
        init_agent_everywhere: bool = False,
        num_skills: int = 8,
        random_skills: bool = False,
        assign_subtasks: bool = False,
        **kwargs,
    ):
        self.room_radius = room_radius
        self.assembler_radius = assembler_radius
        self.button_radius = button_radius
        self.vel_eps = vel_eps
        self.assembly_hold_time = assembly_hold_time
        self.spawn_distance = spawn_distance
        self.spawn_cluster_radius = spawn_cluster_radius
        self.agent_size = agent_size
        self.num_good_agents = num_good_agents
        self.num_adversaries = num_adversaries
        self.action_type = action_type
        self.max_steps = CTF_MAX_STEPS
        self.zero_sum = zero_sum
        self.random_start = random_start
        self.init_agent_everywhere = init_agent_everywhere

        # Skill parameters
        self.num_skills = num_skills
        self.random_skills = random_skills
        self.assign_subtasks = assign_subtasks
        self.num_subtasks = 8  # Number of subtask observations

        # Agent names
        self.good_agents = [f"agent_{i}" for i in range(num_good_agents)]
        self.adversaries = [f"adversary_{i}" for i in range(num_adversaries)]
        agents = self.good_agents + self.adversaries

        # Landmarks: part rooms (2) + assembler (1) + button (conceptual, not physical)
        # We'll use 3 landmarks: part_a_room, part_b_room, assembler
        landmarks = ["part_a_room", "part_b_room", "assembler"]

        if action_type == "Discrete":
            from ..spaces import Discrete
            self.action_spaces = {i: Discrete(5) for i in agents}
        elif action_type == "Continuous":
            self.action_spaces = {i: Box(-1, 1, (5,)) for i in agents}

        self.num_actors = num_good_agents + num_adversaries
        num_landmarks = 3  # 2 rooms + 1 assembler

        self.agent_range = jnp.arange(self.num_actors)

        # Collision setup: agents collide with each other but not with zones
        collides = jnp.concatenate(
            [
                jnp.full((self.num_actors,), True),  # agents collide
                jnp.full((num_landmarks,), False),   # landmarks don't collide
            ]
        )

        # Radii
        rad = jnp.concatenate(
            [
                jnp.full((self.num_actors,), agent_size),
                jnp.array([room_radius, room_radius, assembler_radius]),
            ]
        )

        # Mass
        mass = jnp.concatenate(
            [
                jnp.full((self.num_actors,), 1.0),
                jnp.full((num_landmarks,), 1.0),
            ]
        )

        # Moveable: only agents can move
        moveable = jnp.concatenate(
            [
                jnp.full((self.num_actors,), True),
                jnp.full((num_landmarks,), False),
            ]
        )

        # Max speed
        max_speed = jnp.concatenate(
            [
                jnp.full((self.num_actors,), -1.0),
                jnp.full((num_landmarks,), 0.0),
            ]
        )

        # Observation dimensions
        base_obs_dim = (
            2  # self_vel
            + (self.num_actors - 1) * 4  # other agents pos+vel
            + 2  # part A room rel pos
            + 2  # part B room rel pos
            + 2  # assembler rel pos
            + 2  # button rel pos
            + 2  # carrying status (part_a, part_b)
            + 4  # parts delivered (2 per team)
            + 1  # assembler locked by (-1, 0, 1)
            + 2  # assembly hold progress per team (normalized)
            + 2  # team scores (normalized)
        )
        option_obs_dim = 4  # additional option observations

        self.observation_spaces = {
            i: Box(-jnp.inf, jnp.inf, (base_obs_dim + option_obs_dim,)) for i in agents
        }

        if self.assign_subtasks:
            assert self.num_skills == 8, "Number of skills must be 8 when assigning subtasks"
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
        key, agent_key, adv_key, room_key, option_key = jax.random.split(key, 5)

        if self.random_start:
            # Random angle for room placement
            angle_key, key = jax.random.split(key)
            angle = jax.random.uniform(angle_key, (), minval=0, maxval=2 * jnp.pi)
            
            # Part rooms on opposite sides
            room_distance = self.spawn_distance * 0.8
            part_a_room_pos = jnp.array([
                room_distance / 2 * jnp.cos(angle),
                room_distance / 2 * jnp.sin(angle)
            ])
            part_b_room_pos = jnp.array([
                -room_distance / 2 * jnp.cos(angle),
                -room_distance / 2 * jnp.sin(angle)
            ])
            
            # Assembler at center
            assembler_pos = jnp.array([0.0, 0.0])
            
            # Button near assembler
            button_offset = self.assembler_radius + self.button_radius + 1.0
            assembly_button_pos = assembler_pos + jnp.array([button_offset, 0.0])
            
            # Both teams spawn in a neutral location perpendicular to the room axis
            # This ensures fairness - both teams are equidistant from both rooms
            perp_vec = jnp.array([-jnp.sin(angle), jnp.cos(angle)])  # perpendicular to room axis
            spawn_offset_from_center = self.assembler_radius + self.spawn_cluster_radius + 2.0
            
            # Both teams spawn on the same perpendicular line, close together
            agent_spawn_center = perp_vec * spawn_offset_from_center
            adversary_spawn_center = -perp_vec * spawn_offset_from_center
            
            if self.init_agent_everywhere:
                # Initialize anywhere in the map, inside the circle that encloses the spawn area
                agent_everywhere_radius = self.spawn_distance
                center_pos = jnp.array([0.0, 0.0])  # Use assembler position as center
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
                # Spawn agents clustered
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
        else:
            # Fixed positions - rooms on left and right
            part_a_room_pos = jnp.array([-self.spawn_distance / 2.5, 0.0])
            part_b_room_pos = jnp.array([self.spawn_distance / 2.5, 0.0])
            assembler_pos = jnp.array([0.0, 0.0])
            
            button_offset = self.assembler_radius + self.button_radius + 1.0
            assembly_button_pos = assembler_pos + jnp.array([0.0, button_offset])
            
            # Both teams spawn perpendicular to room axis (above/below center)
            # Equidistant from both rooms for fairness
            spawn_offset_from_center = self.assembler_radius + self.spawn_cluster_radius + 2.0
            agent_spawn_center = jnp.array([0.0, spawn_offset_from_center])
            adversary_spawn_center = jnp.array([0.0, -spawn_offset_from_center])
            
            # Fixed positions in cluster
            agent_angles = jnp.linspace(0, 2 * jnp.pi, self.num_good_agents, endpoint=False)
            agent_init_pos = agent_spawn_center + self.spawn_cluster_radius * 0.5 * jnp.stack(
                [jnp.cos(agent_angles), jnp.sin(agent_angles)], axis=1
            )
            
            adv_angles = jnp.linspace(0, 2 * jnp.pi, self.num_adversaries, endpoint=False)
            adversary_init_pos = adversary_spawn_center + self.spawn_cluster_radius * 0.5 * jnp.stack(
                [jnp.cos(adv_angles), jnp.sin(adv_angles)], axis=1
            )

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

        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_actors, 2)),
            done=jnp.full((self.num_actors,), False),
            step=0,
            part_a_room_pos=part_a_room_pos,
            part_b_room_pos=part_b_room_pos,
            assembler_pos=assembler_pos,
            assembly_button_pos=assembly_button_pos,
            carrying_part_a=jnp.zeros((self.num_actors,), dtype=bool),
            carrying_part_b=jnp.zeros((self.num_actors,), dtype=bool),
            parts_delivered=jnp.zeros((2, 2), dtype=jnp.int32),
            assembler_locked_by=jnp.array(-1, dtype=jnp.int32),
            assembly_hold_progress=jnp.zeros((2,), dtype=jnp.int32),
            team_scores=jnp.zeros((2,), dtype=jnp.int32),
            agent_spawn_center=agent_spawn_center,
            adversary_spawn_center=adversary_spawn_center,
            agent_names=self.good_agents,
            adversary_names=self.adversaries,
            prev_p_pos=p_pos,
            option_assignment=option_assignment,
            cum_subtask_obs=cum_subtask_obs,
        )

        return self.obs_fn(state), state

    @partial(jax.jit, static_argnums=[0])
    def _check_room_entry(self, state: State) -> Tuple[chex.Array, chex.Array]:
        """Check which agents are in which rooms.
        
        Returns:
            in_room_a: (num_actors,) boolean
            in_room_b: (num_actors,) boolean
        """
        dist_to_a = jnp.linalg.norm(state.p_pos - state.part_a_room_pos, axis=1)
        dist_to_b = jnp.linalg.norm(state.p_pos - state.part_b_room_pos, axis=1)
        
        in_room_a = dist_to_a <= self.room_radius
        in_room_b = dist_to_b <= self.room_radius
        
        return in_room_a, in_room_b

    @partial(jax.jit, static_argnums=[0])
    def _check_assembler_entry(self, state: State) -> chex.Array:
        """Check which agents are in the assembler zone.
        
        Returns:
            in_assembler: (num_actors,) boolean
        """
        dist_to_assembler = jnp.linalg.norm(state.p_pos - state.assembler_pos, axis=1)
        return dist_to_assembler <= self.assembler_radius

    @partial(jax.jit, static_argnums=[0])
    def _check_button_press(self, state: State) -> chex.Array:
        """Check which agents are pressing the assembly button.
        
        An agent is pressing if within button_radius and stationary (||v|| <= vel_eps).
        
        Returns:
            pressing: (num_actors,) boolean
        """
        dist_to_button = jnp.linalg.norm(state.p_pos - state.assembly_button_pos, axis=1)
        near_button = dist_to_button <= self.button_radius
        stationary = jnp.linalg.norm(state.p_vel, axis=-1) <= self.vel_eps
        
        return jnp.logical_and(near_button, stationary)

    @partial(jax.jit, static_argnums=[0])
    def _update_inventory(
        self, 
        state: State, 
        in_room_a: chex.Array, 
        in_room_b: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """Update agent inventory based on room entry.
        
        Rules:
        - Pick up Part A: in room A, not carrying anything
        - Pick up Part B: in room B, already carrying Part A (not B)
        
        Returns:
            new_carrying_a: (num_actors,) boolean
            new_carrying_b: (num_actors,) boolean
        """
        # Can pick up A if in room A and not carrying anything
        can_pickup_a = in_room_a & ~state.carrying_part_a & ~state.carrying_part_b
        new_carrying_a = state.carrying_part_a | can_pickup_a
        
        # Can pick up B if in room B and carrying A but not B
        can_pickup_b = in_room_b & state.carrying_part_a & ~state.carrying_part_b
        new_carrying_b = state.carrying_part_b | can_pickup_b
        
        return new_carrying_a, new_carrying_b

    @partial(jax.jit, static_argnums=[0])
    def _update_assembly(
        self,
        state: State,
        in_assembler: chex.Array,
        carrying_a: chex.Array,
        carrying_b: chex.Array,
        pressing: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """Update assembly state based on deliveries and button presses.
        
        Returns:
            new_carrying_a: Updated part A carrying status
            new_carrying_b: Updated part B carrying status
            new_parts_delivered: Updated parts delivered count
            new_assembler_locked_by: Updated assembler lock status
            new_hold_progress: Updated button hold progress
            new_team_scores: Updated team scores
        """
        # Determine team membership
        is_agent_team = jnp.arange(self.num_actors) < self.num_good_agents
        
        # Only allow delivery if assembler is not locked by other team
        can_agent_deliver = (state.assembler_locked_by == -1) | (state.assembler_locked_by == 0)
        can_adv_deliver = (state.assembler_locked_by == -1) | (state.assembler_locked_by == 1)
        
        # Find agents delivering BOTH parts (carrying_b implies carrying_a)
        # An agent must have both parts to make a valid delivery
        delivering_both = in_assembler & carrying_b
        
        # Count deliveries per team (each agent with both parts delivers one of each)
        agent_delivering = jnp.sum(delivering_both[:self.num_good_agents] & can_agent_deliver)
        adv_delivering = jnp.sum(delivering_both[self.num_good_agents:] & can_adv_deliver)
        
        # Update parts delivered - add both parts when agent delivers
        new_parts_delivered = state.parts_delivered.copy()
        new_parts_delivered = new_parts_delivered.at[0, 0].add(agent_delivering)  # Team A delivers Part A
        new_parts_delivered = new_parts_delivered.at[0, 1].add(agent_delivering)  # Team A delivers Part B
        new_parts_delivered = new_parts_delivered.at[1, 0].add(adv_delivering)    # Team B delivers Part A
        new_parts_delivered = new_parts_delivered.at[1, 1].add(adv_delivering)    # Team B delivers Part B
        
        # Clear inventory for agents who delivered - only when carrying BOTH parts
        # (carrying_b implies carrying_a due to pickup ordering)
        delivered_mask = in_assembler & carrying_b & jnp.where(is_agent_team, can_agent_deliver, can_adv_deliver)
        new_carrying_a = carrying_a & ~delivered_mask
        new_carrying_b = carrying_b & ~delivered_mask
        
        # Update assembler lock
        agent_delivered_any = agent_delivering > 0
        adv_delivered_any = adv_delivering > 0
        
        new_assembler_locked_by = jax.lax.select(
            state.assembler_locked_by == -1,
            jax.lax.select(
                agent_delivered_any & can_agent_deliver,
                jnp.array(0, dtype=jnp.int32),
                jax.lax.select(
                    adv_delivered_any & can_adv_deliver,
                    jnp.array(1, dtype=jnp.int32),
                    state.assembler_locked_by
                )
            ),
            state.assembler_locked_by
        )
        
        # Update button hold progress
        # Only count if team has delivered both parts (at least 1 of each)
        agent_ready = (new_parts_delivered[0, 0] >= 1) & (new_parts_delivered[0, 1] >= 1)
        adv_ready = (new_parts_delivered[1, 0] >= 1) & (new_parts_delivered[1, 1] >= 1)
        
        agent_pressing = jnp.any(pressing[:self.num_good_agents])
        adv_pressing = jnp.any(pressing[self.num_good_agents:])
        
        new_hold_progress = state.assembly_hold_progress.copy()
        new_hold_progress = new_hold_progress.at[0].set(
            jnp.where(
                agent_ready & agent_pressing & (new_assembler_locked_by == 0),
                state.assembly_hold_progress[0] + 1,
                state.assembly_hold_progress[0]
            )
        )
        new_hold_progress = new_hold_progress.at[1].set(
            jnp.where(
                adv_ready & adv_pressing & (new_assembler_locked_by == 1),
                state.assembly_hold_progress[1] + 1,
                state.assembly_hold_progress[1]
            )
        )
        
        # Check for assembly completion
        agent_completed = new_hold_progress[0] >= self.assembly_hold_time
        adv_completed = new_hold_progress[1] >= self.assembly_hold_time
        
        new_team_scores = state.team_scores.copy()
        new_team_scores = new_team_scores.at[0].add(agent_completed.astype(jnp.int32))
        new_team_scores = new_team_scores.at[1].add(adv_completed.astype(jnp.int32))
        
        # Reset assembler state if completed
        any_completed = agent_completed | adv_completed
        new_parts_delivered = jnp.where(any_completed, jnp.zeros((2, 2), dtype=jnp.int32), new_parts_delivered)
        new_assembler_locked_by = jnp.where(any_completed, jnp.array(-1, dtype=jnp.int32), new_assembler_locked_by)
        new_hold_progress = jnp.where(any_completed, jnp.zeros((2,), dtype=jnp.int32), new_hold_progress)
        
        return (new_carrying_a, new_carrying_b, new_parts_delivered, 
                new_assembler_locked_by, new_hold_progress, new_team_scores)

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        # Store previous positions
        prev_p_pos = state.p_pos

        # Create simple state for physics
        simple_state = SimpleState(
            p_pos=jnp.vstack(
                (state.p_pos, state.part_a_room_pos[None, :], 
                 state.part_b_room_pos[None, :], state.assembler_pos[None, :])
            ),
            p_vel=jnp.vstack(
                (state.p_vel, jnp.zeros((3, 2)))
            ),
            done=state.done,
            step=state.step,
            goal=None,
            c=jnp.zeros((self.num_actors, self.dim_c)),
        )

        # Step physics
        simple_obs, simple_state, simple_reward, simple_dones, simple_info = (
            SimpleMPE.step_env(self, key, simple_state, actions)
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

        # Check room entry
        in_room_a, in_room_b = self._check_room_entry(state)
        
        # Check assembler entry
        in_assembler = self._check_assembler_entry(state)
        
        # Check button press
        pressing = self._check_button_press(state)
        
        # Update inventory
        new_carrying_a, new_carrying_b = self._update_inventory(state, in_room_a, in_room_b)
        
        # Update assembly state
        (final_carrying_a, final_carrying_b, new_parts_delivered,
         new_assembler_locked_by, new_hold_progress, new_team_scores) = self._update_assembly(
            state, in_assembler, new_carrying_a, new_carrying_b, pressing
        )
        
        # Check if assembly was completed this step
        agent_completed = (new_team_scores[0] > state.team_scores[0])
        adv_completed = (new_team_scores[1] > state.team_scores[1])
        
        # Update state
        state = state.replace(
            carrying_part_a=final_carrying_a,
            carrying_part_b=final_carrying_b,
            parts_delivered=new_parts_delivered,
            assembler_locked_by=new_assembler_locked_by,
            assembly_hold_progress=new_hold_progress,
            team_scores=new_team_scores,
        )

        # Calculate rewards
        rewards = self.get_rewards(state, agent_completed, adv_completed)

        # Compute subtask observations
        subtask_obs = self.subtask_obs_fn(
            state, in_room_a, in_room_b, in_assembler, pressing
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
            "in_room_a": in_room_a,
            "in_room_b": in_room_b,
            "in_assembler": in_assembler,
            "pressing_button": pressing,
            "carrying_part_a": state.carrying_part_a,
            "carrying_part_b": state.carrying_part_b,
            "parts_delivered": state.parts_delivered,
            "assembler_locked_by": state.assembler_locked_by,
            "assembly_hold_progress": state.assembly_hold_progress,
            "team_scores": state.team_scores,
            "agent_completed": agent_completed,
            "adv_completed": adv_completed,
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

        # Update previous positions
        state = state.replace(prev_p_pos=prev_p_pos)

        obs = self.obs_fn(state)

        return obs, state, rewards, simple_dones, info

    @partial(jax.jit, static_argnums=[0])
    def subtask_obs_fn(
        self,
        state: State,
        in_room_a: chex.Array,
        in_room_b: chex.Array,
        in_assembler: chex.Array,
        pressing: chex.Array,
    ) -> Dict[str, chex.Array]:
        """
        Subtask observations for all agents.

        Subtasks (8 total):
        0. near_part_a_room - Agent is near part A room
        1. near_part_b_room - Agent is near part B room
        2. carrying_part_a - Agent is carrying part A
        3. carrying_part_b - Agent is carrying part B
        4. at_assembler - Agent is at assembler zone
        5. pressing_button - Agent is pressing assembly button
        6. collide_teammate - Agent collided with teammate
        7. collide_opponent - Agent collided with opponent
        """
        out = {}
        
        for i, name in enumerate(self.agents):
            is_agent = i < self.num_good_agents
            idx = i if is_agent else i - self.num_good_agents

            # Collision detection
            if is_agent:
                own_dists = jnp.linalg.norm(
                    state.p_pos[i] - jnp.delete(state.p_pos[:self.num_good_agents], idx, axis=0),
                    axis=1,
                )
                other_dists = jnp.linalg.norm(
                    state.p_pos[i] - state.p_pos[self.num_good_agents:], axis=1
                )
            else:
                own_dists = jnp.linalg.norm(
                    state.p_pos[i] - jnp.delete(state.p_pos[self.num_good_agents:], idx, axis=0),
                    axis=1,
                )
                other_dists = jnp.linalg.norm(
                    state.p_pos[i] - state.p_pos[:self.num_good_agents], axis=1
                )
            
            collide_teammate = jnp.any(own_dists <= 2 * self.agent_size)
            collide_opponent = jnp.any(other_dists <= 2 * self.agent_size)

            obs_vec = jnp.array(
                [
                    in_room_a[i],           # 0: near_part_a_room
                    in_room_b[i],           # 1: near_part_b_room
                    state.carrying_part_a[i],  # 2: carrying_part_a
                    state.carrying_part_b[i],  # 3: carrying_part_b
                    in_assembler[i],        # 4: at_assembler
                    pressing[i],            # 5: pressing_button
                    collide_teammate,       # 6: collide_teammate
                    collide_opponent,       # 7: collide_opponent
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

            # Other agents
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

            other_idx = jnp.concatenate([teammate_idx, opponent_idx])
            other_pos = state.p_pos[other_idx]
            other_vel = state.p_vel[other_idx]

            other_rel_pos = ((other_pos - state.p_pos[i]) @ rotmat.T).flatten()
            other_rel_vel = ((other_vel - state.p_vel[i]) @ rotmat.T).flatten()

            # Room and assembler positions
            room_a_rel = (state.part_a_room_pos - state.p_pos[i]) @ rotmat.T
            room_b_rel = (state.part_b_room_pos - state.p_pos[i]) @ rotmat.T
            assembler_rel = (state.assembler_pos - state.p_pos[i]) @ rotmat.T
            button_rel = (state.assembly_button_pos - state.p_pos[i]) @ rotmat.T

            # Carrying status
            carrying_status = jnp.array([
                state.carrying_part_a[i],
                state.carrying_part_b[i]
            ], dtype=jnp.float32)

            # Parts delivered (flattened)
            parts_delivered_flat = state.parts_delivered.flatten().astype(jnp.float32)

            # Assembler lock status (normalized to -1, 0, 1)
            assembler_lock = state.assembler_locked_by.astype(jnp.float32) / 1.0

            # Assembly hold progress (normalized)
            hold_progress = state.assembly_hold_progress.astype(jnp.float32) / self.assembly_hold_time

            # Team scores (normalized by max_steps as rough upper bound)
            team_scores_norm = state.team_scores.astype(jnp.float32) / 10.0

            # Option observations
            own_team = 0 if is_agent else 1
            own_parts_a = state.parts_delivered[own_team, 0]
            own_parts_b = state.parts_delivered[own_team, 1]
            own_ready = (own_parts_a >= 1) & (own_parts_b >= 1)
            own_hold = state.assembly_hold_progress[own_team]
            
            option_obs = jnp.array([
                own_parts_a / 1.0,  # Normalized
                own_parts_b / 1.0,
                own_ready.astype(jnp.float32),
                own_hold / self.assembly_hold_time,
            ])

            obs = jnp.concatenate([
                self_vel,
                other_rel_pos,
                other_rel_vel,
                room_a_rel,
                room_b_rel,
                assembler_rel,
                button_rel,
                carrying_status,
                parts_delivered_flat,
                jnp.array([assembler_lock]),
                hold_progress,
                team_scores_norm,
                option_obs,
            ])

            out[name] = obs

        return out

    @partial(jax.jit, static_argnums=[0])
    def get_rewards(
        self, 
        state: State, 
        agent_completed: chex.Array, 
        adv_completed: chex.Array
    ) -> Dict[str, float]:
        """
        Compute rewards for all agents.

        Reward structure:
        - +10 for completing an assembly
        - -10 for opponent completing (if zero_sum)
        - +0.1 for delivering a part
        - +0.01 for holding button (if ready)
        """
        r = jnp.zeros((self.num_actors,))

        # Big reward for completing assembly
        r = r.at[:self.num_good_agents].add(agent_completed.astype(float) * 10.0)
        r = r.at[self.num_good_agents:].add(adv_completed.astype(float) * 10.0)

        if self.zero_sum:
            r = r.at[:self.num_good_agents].subtract(adv_completed.astype(float) * 10.0)
            r = r.at[self.num_good_agents:].subtract(agent_completed.astype(float) * 10.0)

        return {a: r[i] for i, a in enumerate(self.agents)}

