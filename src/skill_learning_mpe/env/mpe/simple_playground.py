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
    Playground State - combines all game mechanics from:
    - Payload Escort: payload, push mechanics, zone unlock buttons
    - Assembly Line: part collection, assembly, assembly button
    - King of Hill: hills, melee combat, HP, respawn
    - CTF: flags, flag carrying
    
    All agents can participate in all activities simultaneously.
    """

    # === Agent state ===
    p_pos: chex.Array  # (num_actors, 2) agent positions
    p_vel: chex.Array  # (num_actors, 2) agent velocities
    done: chex.Array  # (num_actors,) done flags
    step: int  # current step
    agent_hp: chex.Array  # (num_actors,) HP for each agent (from KoH)

    # === Payload Escort ===
    payload_pos: chex.Array  # (2,) payload position
    payload_vel: chex.Array  # (2,) payload velocity
    agent_goal_zone: chex.Array  # (2,) Team A's goal zone
    adversary_goal_zone: chex.Array  # (2,) Team B's goal zone
    payload_button_pos: chex.Array  # (4, 2) payload zone unlock buttons
    payload_button_toggled: chex.Array  # (2,) whether zones are unlocked

    # === CTF ===
    flag_moving: chex.Array  # (2,) flags moving
    flag_carrier: chex.Array  # (2,) flag carriers
    flag_p_pos: chex.Array  # (2, 2) flag positions
    flag_p_vel: chex.Array  # (2, 2) flag velocities
    ctf_button_pos: chex.Array  # (2, 2) CTF zone gate buttons

    # === Assembly Line ===
    part_a_room_pos: chex.Array  # (2,) part A room center
    part_b_room_pos: chex.Array  # (2,) part B room center
    assembler_pos: chex.Array  # (2,) assembler zone center
    assembly_button_pos: chex.Array  # (2,) assembly button position
    carrying_part_a: chex.Array  # (num_actors,) carrying part A
    carrying_part_b: chex.Array  # (num_actors,) carrying part B
    parts_delivered: chex.Array  # (2, 2) parts delivered per team
    assembler_locked_by: chex.Array  # scalar: which team locked assembler
    assembly_hold_progress: chex.Array  # (2,) assembly hold progress per team
    team_scores: chex.Array  # (2,) assembly scores per team

    # === King of Hill ===
    hill_pos: chex.Array  # (2, 2) positions of two hills
    hill_owner: chex.Array  # (2,) hill ownership: -1=none, 0=agents, 1=adversaries

    # === Spawn locations ===
    agent_spawn_center: chex.Array  # (2,)
    adversary_spawn_center: chex.Array  # (2,)

    # === Static environment ===
    obs_pos: chex.Array  # (num_obstacles, 2) obstacle positions

    # === Agent metadata ===
    agent_names: List[str] = struct.field(pytree_node=False)
    adversary_names: List[str] = struct.field(pytree_node=False)

    # === Extended fields for skills ===
    prev_p_pos: chex.Array  # (num_actors, 2) previous positions
    prev_payload_pos: chex.Array  # (2,) previous payload position
    prev_hill_owner: chex.Array  # (2,) previous hill ownership
    option_assignment: chex.Array  # skill assignments
    cum_subtask_obs: chex.Array  # (num_actors, num_subtasks) cumulative subtask observations


class SimplePlayground(SimpleMPE):
    """
    Playground Environment combining all game mechanics.

    Combines:
    - Payload Escort: Push payload to opponent's zone (requires zone unlock buttons)
    - CTF: Capture opponent's flag and return to own zone (zone entry requires button)
    - Assembly Line: Collect parts A and B, deliver to assembler, hold assembly button
    - King of Hill: Capture and hold hills, melee combat with HP/respawn

    All mechanics operate simultaneously. Agents can participate in multiple activities.
    
    Subtasks (merged from all environments, total ~27):
    Payload Escort (7): pushing_payload, near_own_goal, near_opp_goal, contacting_payload, 
                        collide_teammate, collide_opponent, pressing_payload_button
    CTF (7): pressing_ctf_button, in_opp_zone, in_own_zone, carrying_opp_flag, 
             collide_obs, collide_own, collide_opp
    Assembly Line (8): near_part_a_room, near_part_b_room, carrying_part_a, carrying_part_b,
                       at_assembler, pressing_assembly_button, collide_teammate, collide_opponent
    King of Hill (6): near_own_hill, near_enemy_hill, near_neutral_hill, attacked_opponent,
                      collide_teammate, collide_opponent
    
    Note: Some subtasks overlap (collisions appear in multiple), so we deduplicate to ~27 total.
    """

    def __init__(
        self,
        *,
        num_good_agents: int = 3,
        num_adversaries: int = 3,
        num_obstacles: int = 2,
        action_type=DISCRETE_ACT,
        # Payload Escort params
        payload_mass: float = 10.0,
        payload_radius: float = 2.0,
        push_radius: float = 3.0,
        zone_size: float = 3.5,
        dist_between_zones: float = 25.0,
        payload_button_radius: float = 1.5,
        payload_button_offset: float = 3.0,
        # Assembly Line params
        room_radius: float = 3.0,
        assembler_radius: float = 2.5,
        assembly_button_radius: float = 1.5,
        assembly_hold_time: int = 30,
        spawn_distance: float = 25.0,
        spawn_cluster_radius: float = 3.0,
        # King of Hill params
        hill_radius: float = 2.0,
        melee_range: float = 2.5,
        max_hp: int = 3,
        # Common params
        agent_size: float = 0.7,
        obstacle_size: float = 1.0,
        vel_eps: float = 2.0,
        zero_sum: bool = True,
        random_start: bool = True,
        init_agent_everywhere: bool = False,
        zone_blocking: bool = False,  # Whether zones block entry (payload and CTF)
        num_skills: int = 27,
        random_skills: bool = False,
        assign_subtasks: bool = False,
        **kwargs,
    ):
        # Store all parameters
        self.zone_size = zone_size
        self.agent_size = agent_size
        self.obstacle_size = obstacle_size if num_obstacles > 0 else 0.001
        self.num_good_agents = num_good_agents
        self.num_adversaries = num_adversaries
        self.dist_between_zones = dist_between_zones
        self._user_num_obstacles = num_obstacles
        self.num_obstacles = max(num_obstacles, 1)
        self.action_type = action_type
        self.max_steps = CTF_MAX_STEPS
        self.zero_sum = zero_sum
        self.random_start = random_start
        self.init_agent_everywhere = init_agent_everywhere
        self.zone_blocking = zone_blocking
        self.vel_eps = vel_eps

        # Payload Escort params
        self.payload_mass = payload_mass
        self.payload_radius = payload_radius
        self.push_radius = push_radius
        self.payload_button_radius = payload_button_radius
        self.payload_button_offset = payload_button_offset

        # Assembly Line params
        self.room_radius = room_radius
        self.assembler_radius = assembler_radius
        self.assembly_button_radius = assembly_button_radius
        self.assembly_hold_time = assembly_hold_time
        self.spawn_distance = spawn_distance
        self.spawn_cluster_radius = spawn_cluster_radius

        # King of Hill params
        self.hill_radius = hill_radius
        self.melee_range = melee_range
        self.max_hp = max_hp

        # Skill parameters
        self.num_skills = num_skills
        self.random_skills = random_skills
        self.assign_subtasks = assign_subtasks
        # Combined subtasks: 7 (PE) + 7 (CTF) + 8 (AL) + 6 (KoH) - overlap = ~27
        # Actually we'll keep all distinct: PE(7) + CTF(7) + AL(8) + KoH(6) = 28
        # But collisions overlap, so let's make it: 
        # PE: pushing, near_own_goal, near_opp_goal, contacting_payload, pressing_payload_button (5)
        # CTF: pressing_ctf_button, in_opp_zone, in_own_zone, carrying_opp_flag (4)
        # AL: near_part_a, near_part_b, carrying_a, carrying_b, at_assembler, pressing_assembly_button (6)
        # KoH: near_own_hill, near_enemy_hill, near_neutral_hill, attacked_opponent (4)
        # Collisions: collide_teammate, collide_opponent, collide_obs (3)
        # Total: 22 subtasks
        self.num_subtasks = 22

        # Agent names
        self.good_agents = [f"agent_{i}" for i in range(num_good_agents)]
        self.adversaries = [f"adversary_{i}" for i in range(num_adversaries)]
        agents = self.good_agents + self.adversaries

        # Landmarks: payload + obstacles + zones + part rooms + assembler + hills
        obs = [f"obstacle_{i}" for i in range(self.num_obstacles)]
        landmarks = ["payload"] + obs + ["agent_zone", "adversary_zone", 
                                         "part_a_room", "part_b_room", "assembler"] + ["hill_0", "hill_1"]

        # Action space: 5 movement + 1 melee (same as KoH)
        if action_type == "Discrete":
            self.action_spaces = {i: Discrete(6) for i in agents}
        elif action_type == "Continuous":
            self.action_spaces = {i: Box(-1, 1, (6,)) for i in agents}

        self.num_actors = num_good_agents + num_adversaries
        # Landmarks: 1 payload + obstacles + 2 zones + 2 rooms + 1 assembler + 2 hills
        num_landmarks = 1 + self.num_obstacles + 2 + 3 + 2

        self.agent_range = jnp.arange(self.num_actors)

        # Collision setup
        collides = jnp.concatenate(
            [
                jnp.full((self.num_actors,), True),   # agents
                jnp.full((1,), True),                  # payload
                jnp.full((self.num_obstacles,), True), # obstacles
                jnp.full((2,), False),                 # zones don't collide
                jnp.full((3,), False),                 # rooms and assembler don't collide
                jnp.full((2,), False),                 # hills don't collide
            ]
        )

        # Radii
        rad = jnp.concatenate(
            [
                jnp.full((self.num_actors,), agent_size),
                jnp.full((1,), payload_radius),
                jnp.full((self.num_obstacles,), self.obstacle_size),
                jnp.full((2,), zone_size),
                jnp.array([room_radius, room_radius, assembler_radius]),
                jnp.full((2,), hill_radius),
            ]
        )

        # Mass
        mass = jnp.concatenate(
            [
                jnp.full((self.num_actors,), 1.0),
                jnp.full((1,), payload_mass),
                jnp.full((self.num_obstacles,), 1.0),
                jnp.full((7,), 1.0),  # zones + rooms + assembler + hills
            ]
        )

        # Moveable
        moveable = jnp.concatenate(
            [
                jnp.full((self.num_actors,), True),
                jnp.full((1,), True),                  # payload moves
                jnp.full((self.num_obstacles,), False),
                jnp.full((7,), False),                 # zones, rooms, assembler, hills don't move
            ]
        )

        # Max speed
        max_speed = jnp.concatenate(
            [
                jnp.full((self.num_actors,), -1.0),
                jnp.full((1,), -1.0),
                jnp.full((self.num_obstacles,), 0.0),
                jnp.full((7,), 0.0),
            ]
        )

        # Observation dimensions (very large!)
        # Base: self_vel (2) + other agents (4*(num_actors-1)) + payload (4) + 
        #       zones (4) + flags (4) + rooms (4) + assembler (2) + button (2) +
        #       hills (4) + obstacles (num_obstacles*2) + HP (num_actors+1) +
        #       carrying (2) + parts_delivered (4) + assembler_locked (1) + 
        #       hold_progress (2) + team_scores (2) + hill_owner (2) +
        #       button states (payload: 2, ctf: 2) = ...
        base_obs_dim = (
            2 +  # self_vel
            (self.num_actors - 1) * 4 +  # other agents pos+vel
            2 +  # payload rel pos
            2 +  # payload vel
            2 +  # own goal rel
            2 +  # opp goal rel
            2 * 2 +  # flags rel pos (2 flags)
            2 +  # part A room rel
            2 +  # part B room rel
            2 +  # assembler rel
            2 +  # assembly button rel
            2 * 2 +  # hills rel pos
            self._user_num_obstacles * 2 +  # obstacles rel
            1 +  # own HP
            self.num_actors +  # all HP
            2 +  # carrying status (part_a, part_b)
            4 +  # parts_delivered
            1 +  # assembler_locked
            2 +  # assembly_hold_progress
            2 +  # team_scores
            2 +  # hill_owner
            4 * 2 +  # payload button rel pos (4 buttons)
            2 * 2 +  # CTF button rel pos (2 buttons)
            2 +  # payload_button_toggled
            2  # CTF button pressed (any agent)
        )
        option_obs_dim = 10  # additional option observations

        self.observation_spaces = {
            i: Box(-jnp.inf, jnp.inf, (base_obs_dim + option_obs_dim,)) for i in agents
        }

        if self.assign_subtasks:
            assert self.num_skills == self.num_subtasks, f"Number of skills must be {self.num_subtasks} when assigning subtasks"
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
    def _place_payload_buttons(self, agent_zone: chex.Array, adversary_zone: chex.Array) -> chex.Array:
        """Place buttons for payload zone unlocking (from Payload Escort)."""
        vec = adversary_zone - agent_zone
        dist = jnp.linalg.norm(vec) + 1e-8
        u = vec / dist
        v = jnp.array([-u[1], u[0]])
        
        btn_a1 = agent_zone + v * (self.zone_size + self.payload_button_offset)
        btn_a2 = agent_zone - v * (self.zone_size + self.payload_button_offset)
        btn_b1 = adversary_zone + v * (self.zone_size + self.payload_button_offset)
        btn_b2 = adversary_zone - v * (self.zone_size + self.payload_button_offset)
        
        return jnp.vstack((btn_a1, btn_a2, btn_b1, btn_b2))

    @partial(jax.jit, static_argnums=[0])
    def _place_ctf_buttons(self, agent_zone: chex.Array, adversary_zone: chex.Array) -> chex.Array:
        """Place buttons for CTF zone entry (from CTF Buttons)."""
        vec = adversary_zone - agent_zone
        dist = jnp.linalg.norm(vec) + 1e-8
        u = vec / dist
        v = jnp.array([-u[1], u[0]])
        
        a_btn = agent_zone + v * (self.zone_size + 1.0)
        b_btn = adversary_zone - v * (self.zone_size + 1.0)
        return jnp.vstack((a_btn, b_btn))

    @partial(jax.jit, static_argnums=[0])
    def _check_collision(self, pos1: chex.Array, pos2: chex.Array, radius1: float, radius2: float) -> chex.Array:
        """Check if two circular entities overlap."""
        dist = jnp.linalg.norm(pos1 - pos2)
        return dist < (radius1 + radius2 + 0.5)  # 0.5 safety margin

    @partial(jax.jit, static_argnums=[0])
    def _resolve_agent_overlaps(self, key: chex.PRNGKey, p_pos: chex.Array, max_iterations: int = 20) -> chex.Array:
        """Resolve agent-agent overlaps by pushing overlapping agents apart."""
        min_separation = 2 * self.agent_size + 1.0  # Minimum distance between agent centers (with safety margin)
        
        def body_fn(iteration, pos):
            # Compute pairwise distances
            pos_diff = pos[:, None, :] - pos[None, :, :]  # (N, N, 2)
            dists = jnp.linalg.norm(pos_diff, axis=-1)  # (N, N)
            # Set diagonal to large value to ignore self-distances
            dists = dists + jnp.eye(self.num_actors) * 1e6
            
            # Find overlaps
            overlaps = dists < min_separation  # (N, N)
            
            # Normalize position differences (avoid division by zero)
            dists_safe = jnp.where(dists > 0, dists, 1.0)  # (N, N)
            pos_dirs = pos_diff / dists_safe[:, :, None]  # (N, N, 2) - normalized directions
            
            # Compute repulsion forces for each agent
            overlap_amounts = jnp.maximum(0, min_separation - dists)  # (N, N) - how much overlap
            repulsion_forces = pos_dirs * overlap_amounts[:, :, None] * overlaps[:, :, None]  # (N, N, 2)
            displacements = jnp.sum(repulsion_forces, axis=1) * 0.5  # (N, 2) - sum over overlapping agents, scale down
            
            return pos + displacements
        
        # Iteratively resolve overlaps
        resolved_pos = jax.lax.fori_loop(0, max_iterations, body_fn, p_pos)
        return resolved_pos

    @partial(jax.jit, static_argnums=[0])
    def _ensure_agent_no_static_overlap(
        self,
        key: chex.PRNGKey,
        p_pos: chex.Array,
        static_positions: chex.Array,
        static_radii: chex.Array,
        fallback_positions: chex.Array,
    ) -> chex.Array:
        """Ensure agents don't overlap with static entities by pushing them away."""
        num_static = static_positions.shape[0]
        min_separation = self.agent_size + static_radii + 1.0  # (num_static,)
        
        def resolve_agent_static(i, pos):
            agent_pos = pos[i]
            
            # Compute distances to all static entities
            agent_pos_expanded = agent_pos[None, :]  # (1, 2)
            static_pos_expanded = static_positions  # (num_static, 2)
            dists = jnp.linalg.norm(agent_pos_expanded - static_pos_expanded, axis=-1)  # (num_static,)
            
            # Find overlaps
            overlaps = dists < min_separation  # (num_static,)
            has_overlap = jnp.any(overlaps)
            
            # Compute repulsion from overlapping static entities
            pos_diffs = agent_pos_expanded - static_pos_expanded  # (num_static, 2)
            dists_safe = jnp.where(dists > 0, dists, 1.0)  # (num_static,)
            pos_dirs = pos_diffs / dists_safe[:, None]  # (num_static, 2) - normalized directions away from static
            
            # Repulsion force proportional to overlap amount
            overlap_amounts = jnp.maximum(0, min_separation - dists)  # (num_static,)
            repulsion_forces = pos_dirs * overlap_amounts[:, None] * overlaps[:, None]  # (num_static, 2)
            displacement = jnp.sum(repulsion_forces, axis=0) * 0.5  # (2,)
            
            new_pos = agent_pos + displacement
            
            # Only apply displacement if there's an overlap
            final_pos = jnp.where(has_overlap, new_pos, agent_pos)
            pos = pos.at[i].set(final_pos)
            
            return pos
        
        # Resolve overlaps for all agents iteratively
        safe_pos = jax.lax.fori_loop(0, self.num_actors, resolve_agent_static, p_pos)
        
        return safe_pos

    @partial(jax.jit, static_argnums=[0])
    def _resolve_static_entity_overlaps(
        self,
        key: chex.PRNGKey,
        entity_positions: chex.Array,
        entity_radii: chex.Array,
        avoid_positions: chex.Array,
        avoid_radii: chex.Array,
        bounds_min: float = -30.0,
        bounds_max: float = 30.0,
        max_attempts: int = 50,
    ) -> chex.Array:
        """Resolve overlaps between static entities by repositioning them."""
        num_entities = entity_positions.shape[0]
        
        def resolve_entity(i, state):
            resolved_pos, key_state = state
            entity_pos = entity_positions[i]
            entity_radius = entity_radii[i]
            
            # Check overlap with avoid positions
            entity_pos_expanded = entity_pos[None, :]  # (1, 2)
            avoid_pos_expanded = avoid_positions  # (num_avoid, 2)
            avoid_dists = jnp.linalg.norm(entity_pos_expanded - avoid_pos_expanded, axis=-1)  # (num_avoid,)
            avoid_min_dists = entity_radius + avoid_radii + 0.5  # (num_avoid,)
            overlaps_avoid = avoid_dists < avoid_min_dists  # (num_avoid,)
            
            # Check overlap with previously resolved entities (i > 0)
            # Use dynamic_slice to get entities before current index
            # For i == 0, we'll get empty slice which we handle
            prev_mask = jnp.arange(num_entities) < i  # (num_entities,) - True for indices < i
            
            # Check distances to all previously resolved entities
            all_dists_to_resolved = jnp.linalg.norm(entity_pos_expanded - resolved_pos, axis=-1)  # (num_entities,)
            all_min_dists = entity_radius + entity_radii + 0.5  # (num_entities,)
            all_overlaps = all_dists_to_resolved < all_min_dists  # (num_entities,)
            # Only consider overlaps with entities before current (mask out current and future)
            overlaps_other = jnp.any(all_overlaps & prev_mask)
            
            has_overlap = jnp.any(overlaps_avoid) | overlaps_other
            
            def find_valid_pos(attempt_idx, pos_and_key):
                pos, k = pos_and_key
                k, subk = jax.random.split(k)
                new_pos = jax.random.uniform(subk, (2,), minval=bounds_min, maxval=bounds_max)
                
                # Check validity
                new_pos_expanded = new_pos[None, :]  # (1, 2)
                new_avoid_dists = jnp.linalg.norm(new_pos_expanded - avoid_positions, axis=-1)  # (num_avoid,)
                new_avoid_overlaps = new_avoid_dists < avoid_min_dists
                
                # Check overlaps with previously resolved entities
                prev_mask = jnp.arange(num_entities) < i  # (num_entities,)
                new_all_dists = jnp.linalg.norm(new_pos_expanded - resolved_pos, axis=-1)  # (num_entities,)
                new_all_min_dists = entity_radius + entity_radii + 0.5  # (num_entities,)
                new_all_overlaps = new_all_dists < new_all_min_dists  # (num_entities,)
                new_other_overlaps = jnp.any(new_all_overlaps & prev_mask)
                
                valid = ~(jnp.any(new_avoid_overlaps) | jnp.any(new_other_overlaps))
                new_pos_final = jnp.where(valid, new_pos, pos)
                return (new_pos_final, k)
            
            key_state, subk = jax.random.split(key_state)
            final_pos = jax.lax.cond(
                has_overlap,
                lambda: jax.lax.fori_loop(0, max_attempts, find_valid_pos, (entity_pos, subk))[0],
                lambda: entity_pos
            )
            
            resolved_pos = resolved_pos.at[i].set(final_pos)
            return (resolved_pos, key_state)
        
        keys = jax.random.split(key, num_entities)
        final_key = keys[0]  # Start with first key
        resolved_pos, _ = jax.lax.fori_loop(0, num_entities, resolve_entity, (entity_positions, final_key))
        
        return resolved_pos

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Reset environment with random positions for all elements."""
        key, agent_key, adv_key, option_key = jax.random.split(key, 4)

        if self.random_start:
            # Random zone positions
            init_key, key = jax.random.split(key)
            agent_goal_zone = jax.random.uniform(
                init_key,
                (2,),
                minval=-self.dist_between_zones / 2,
                maxval=self.dist_between_zones / 2,
            )
            
            adv_angle_key, key = jax.random.split(key)
            adv_angle = jax.random.uniform(adv_angle_key, (1,), minval=0, maxval=2 * jnp.pi)
            adversary_goal_zone = (
                jnp.array([
                    self.dist_between_zones * jnp.cos(adv_angle),
                    self.dist_between_zones * jnp.sin(adv_angle),
                ]).flatten()
                + agent_goal_zone
            )
            
            # Payload at center
            payload_pos = (agent_goal_zone + adversary_goal_zone) / 2
            
            # Part rooms positioned perpendicular to zone axis
            room_angle_key, key = jax.random.split(key)
            angle = jax.random.uniform(room_angle_key, (), minval=0, maxval=2 * jnp.pi)
            room_distance = self.dist_between_zones * 0.6
            part_a_room_pos = jnp.array([
                room_distance / 2 * jnp.cos(angle),
                room_distance / 2 * jnp.sin(angle)
            ])
            part_b_room_pos = jnp.array([
                -room_distance / 2 * jnp.cos(angle),
                -room_distance / 2 * jnp.sin(angle)
            ])
            
            # Assembler at center (further from rooms)
            assembler_pos = jnp.array([0.0, 0.0])
            
            # Assembly button
            button_offset = self.assembler_radius + self.assembly_button_radius + 1.0
            assembly_button_pos = assembler_pos + jnp.array([button_offset, 0.0])
            
            # Spawn centers
            perp_vec = jnp.array([-jnp.sin(angle), jnp.cos(angle)])
            spawn_offset = self.assembler_radius + self.spawn_cluster_radius + 2.0
            agent_spawn_center = perp_vec * spawn_offset
            adversary_spawn_center = -perp_vec * spawn_offset
            
            # Hills - random positions
            hill_key, key = jax.random.split(key)
            hill_pos = jax.random.uniform(
                hill_key, (2, 2),
                minval=-self.dist_between_zones / 3, maxval=self.dist_between_zones / 3
            )
            
            # Obstacles
            if self._user_num_obstacles > 0:
                obs_key, key = jax.random.split(key)
                obs_pos = jax.random.uniform(
                    obs_key, (self._user_num_obstacles, 2),
                    minval=-self.dist_between_zones / 2, maxval=self.dist_between_zones / 2
                )
            else:
                obs_pos = jnp.array([[1e6, 1e6]])
        else:
            # Fixed positions
            agent_goal_zone = jnp.array([-self.dist_between_zones / 2, 0.0])
            adversary_goal_zone = jnp.array([self.dist_between_zones / 2, 0.0])
            payload_pos = jnp.array([0.0, 0.0])
            
            part_a_room_pos = jnp.array([-self.dist_between_zones / 2, self.dist_between_zones / 3])
            part_b_room_pos = jnp.array([self.dist_between_zones / 2, -self.dist_between_zones / 3])
            assembler_pos = jnp.array([0.0, 0.0])
            button_offset = self.assembler_radius + self.assembly_button_radius + 1.0
            assembly_button_pos = assembler_pos + jnp.array([0.0, button_offset])
            
            agent_spawn_center = jnp.array([0.0, self.dist_between_zones / 3])
            adversary_spawn_center = jnp.array([0.0, -self.dist_between_zones / 3])
            
            hill_pos = jnp.array([
                [-self.dist_between_zones / 6, self.dist_between_zones / 6],
                [self.dist_between_zones / 6, -self.dist_between_zones / 6]
            ])
            
            if self._user_num_obstacles > 0:
                obs_angles = jnp.linspace(0, 2 * jnp.pi, self._user_num_obstacles, endpoint=False)
                obs_pos = self.dist_between_zones / 4 * jnp.stack(
                    [jnp.cos(obs_angles), jnp.sin(obs_angles)], axis=1
                )
            else:
                obs_pos = jnp.array([[1e6, 1e6]])

        # Agent positions
        if self.init_agent_everywhere:
            agent_everywhere_radius = self.dist_between_zones
            center_pos = (agent_goal_zone + adversary_goal_zone) / 2
            ag_r_key, ag_a_key = jax.random.split(agent_key)
            agent_init_radius = jax.random.uniform(
                ag_r_key, (self.num_good_agents, 1),
                minval=self.agent_size, maxval=agent_everywhere_radius
            )
            agent_init_angle = jax.random.uniform(
                ag_a_key, (self.num_good_agents, 1),
                minval=0.0, maxval=2 * jnp.pi
            )
            agent_init_pos = (
                agent_init_radius * jnp.hstack((jnp.cos(agent_init_angle), jnp.sin(agent_init_angle)))
                + center_pos
            )
            adv_r_key, adv_a_key = jax.random.split(adv_key)
            adversary_init_radius = jax.random.uniform(
                adv_r_key, (self.num_adversaries, 1),
                minval=self.agent_size, maxval=agent_everywhere_radius
            )
            adversary_init_angle = jax.random.uniform(
                adv_a_key, (self.num_adversaries, 1),
                minval=0.0, maxval=2 * jnp.pi
            )
            adversary_init_pos = (
                adversary_init_radius * jnp.hstack((jnp.cos(adversary_init_angle), jnp.sin(adversary_init_angle)))
                + center_pos
            )
        else:
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

        p_pos = jnp.vstack((agent_init_pos, adversary_init_pos))

        # Place buttons (needed for collision checking)
        payload_button_pos = self._place_payload_buttons(agent_goal_zone, adversary_goal_zone)
        ctf_button_pos = self._place_ctf_buttons(agent_goal_zone, adversary_goal_zone)

        # Resolve agent-agent overlaps
        key, overlap_key = jax.random.split(key)
        p_pos = self._resolve_agent_overlaps(overlap_key, p_pos)

        # Collect all static entity positions and radii for overlap checking
        # Static entities: payload, buttons (6 total), rooms (2), assembler, hills (2), obstacles
        all_static_positions = jnp.vstack([
            payload_pos[None, :],  # payload
            payload_button_pos,  # 4 payload buttons
            ctf_button_pos,  # 2 CTF buttons
            part_a_room_pos[None, :],  # part A room
            part_b_room_pos[None, :],  # part B room
            assembler_pos[None, :],  # assembler
            assembly_button_pos[None, :],  # assembly button
            hill_pos,  # 2 hills
            obs_pos[:self._user_num_obstacles] if self._user_num_obstacles > 0 else jnp.array([[1e6, 1e6]]),  # obstacles
        ])
        all_static_radii = jnp.concatenate([
            jnp.array([self.payload_radius]),
            jnp.full((4,), self.payload_button_radius),
            jnp.full((2,), self.payload_button_radius),
            jnp.array([self.room_radius]),
            jnp.array([self.room_radius]),
            jnp.array([self.assembler_radius]),
            jnp.array([self.assembly_button_radius]),
            jnp.full((2,), self.hill_radius),
            jnp.full((self._user_num_obstacles if self._user_num_obstacles > 0 else 1,), self.obstacle_size),
        ])

        # Ensure agents don't overlap with static entities
        # Use original spawn positions as fallback
        fallback_pos = jnp.vstack((agent_init_pos, adversary_init_pos))
        key, static_overlap_key = jax.random.split(key)
        p_pos = self._ensure_agent_no_static_overlap(static_overlap_key, p_pos, all_static_positions, all_static_radii, fallback_pos)
        
        # Re-resolve agent-agent overlaps after static entity adjustment
        key, re_overlap_key = jax.random.split(key)
        p_pos = self._resolve_agent_overlaps(re_overlap_key, p_pos)

        # Resolve static entity overlaps (hills and obstacles with each other and key areas)
        # Key areas to avoid: zones, payload position, assembler
        key_areas = jnp.vstack([
            agent_goal_zone[None, :],
            adversary_goal_zone[None, :],
            payload_pos[None, :],
            assembler_pos[None, :],
            part_a_room_pos[None, :],
            part_b_room_pos[None, :],
        ])
        key_area_radii = jnp.array([
            self.zone_size,
            self.zone_size,
            self.payload_radius,
            self.assembler_radius,
            self.room_radius,
            self.room_radius,
        ])

        # Resolve hills and obstacles overlaps
        if self._user_num_obstacles > 0:
            hills_and_obs_pos = jnp.vstack([hill_pos, obs_pos[:self._user_num_obstacles]])
            hills_and_obs_radii = jnp.concatenate([
                jnp.full((2,), self.hill_radius),
                jnp.full((self._user_num_obstacles,), self.obstacle_size),
            ])
            key, static_resolve_key = jax.random.split(key)
            resolved_hills_obs = self._resolve_static_entity_overlaps(
                static_resolve_key,
                hills_and_obs_pos,
                hills_and_obs_radii,
                key_areas,
                key_area_radii,
                bounds_min=-self.dist_between_zones / 2,
                bounds_max=self.dist_between_zones / 2,
            )
            hill_pos = resolved_hills_obs[:2]
            obs_pos = resolved_hills_obs[2:]
        else:
            # Only resolve hills
            key, static_resolve_key = jax.random.split(key)
            hill_pos = self._resolve_static_entity_overlaps(
                static_resolve_key,
                hill_pos,
                jnp.full((2,), self.hill_radius),
                key_areas,
                key_area_radii,
                bounds_min=-self.dist_between_zones / 2,
                bounds_max=self.dist_between_zones / 2,
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

        # Buttons already placed above for collision checking

        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_actors, 2)),
            done=jnp.full((self.num_actors,), False),
            step=0,
            agent_hp=jnp.full((self.num_actors,), self.max_hp, dtype=jnp.int32),
            payload_pos=payload_pos,
            payload_vel=jnp.zeros(2),
            agent_goal_zone=agent_goal_zone,
            adversary_goal_zone=adversary_goal_zone,
            payload_button_pos=payload_button_pos,
            payload_button_toggled=jnp.array([False, False]),
            flag_moving=jnp.array([False, False]),
            flag_carrier=jnp.array([0, 0], dtype=jnp.int32),
            flag_p_pos=jnp.vstack([adversary_goal_zone, agent_goal_zone]),  # flags at opponent zones
            flag_p_vel=jnp.zeros((2, 2)),
            ctf_button_pos=ctf_button_pos,
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
            hill_pos=hill_pos,
            hill_owner=jnp.full((2,), -1, dtype=jnp.int32),
            agent_spawn_center=agent_spawn_center,
            adversary_spawn_center=adversary_spawn_center,
            obs_pos=obs_pos,
            agent_names=self.good_agents,
            adversary_names=self.adversaries,
            prev_p_pos=p_pos,
            prev_payload_pos=payload_pos,
            prev_hill_owner=jnp.full((2,), -1, dtype=jnp.int32),
            option_assignment=option_assignment,
            cum_subtask_obs=cum_subtask_obs,
        )

        return self.obs_fn(state), state

    # ========== HELPER METHODS ==========
    
    @partial(jax.jit, static_argnums=[0])
    def _pressed_payload_buttons_mask(self, state: State) -> chex.Array:
        """Returns (4,) boolean array - which payload buttons are pressed."""
        btn = state.payload_button_pos  # (4, 2)
        pos = state.p_pos  # (N, 2)
        vel = state.p_vel  # (N, 2)
        dists = jnp.linalg.norm(btn[:, None, :] - pos[None, :, :], axis=-1)  # (4, N)
        near = dists <= self.payload_button_radius
        still = jnp.linalg.norm(vel, axis=-1) <= self.vel_eps  # (N,)
        pressing = jnp.logical_and(near, still[None, :])  # (4, N)
        return jnp.any(pressing, axis=1)  # (4,)
    
    @partial(jax.jit, static_argnums=[0])
    def _update_payload_button_toggles(self, button_toggled: chex.Array, pressed: chex.Array) -> chex.Array:
        """Update payload button toggle states."""
        unlock_adversary = jnp.logical_or(pressed[0], pressed[1])
        unlock_agent = jnp.logical_or(pressed[2], pressed[3])
        return jnp.array([
            jnp.logical_or(button_toggled[0], unlock_agent),
            jnp.logical_or(button_toggled[1], unlock_adversary),
        ])
    
    @partial(jax.jit, static_argnums=[0])
    def _pressed_ctf_buttons_mask(self, state: State) -> chex.Array:
        """Returns (2, N) boolean array - which agents are pressing which CTF buttons."""
        btn = state.ctf_button_pos  # (2, 2)
        pos = state.p_pos  # (N, 2)
        vel = state.p_vel  # (N, 2)
        dists = jnp.linalg.norm(btn[:, None, :] - pos[None, :, :], axis=-1)  # (2, N)
        near = dists <= self.payload_button_radius  # Reuse button radius
        still = jnp.linalg.norm(vel, axis=-1) <= self.vel_eps  # (N,)
        return jnp.logical_and(near, still[None, :])  # (2, N)
    
    @partial(jax.jit, static_argnums=[0])
    def _compute_push_forces(self, state: State) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Compute forces on payload from all agents."""
        agent_dists = jnp.linalg.norm(state.p_pos[:self.num_good_agents] - state.payload_pos, axis=1)
        adv_dists = jnp.linalg.norm(state.p_pos[self.num_good_agents:] - state.payload_pos, axis=1)
        
        agent_pushing = agent_dists <= self.push_radius
        adv_pushing = adv_dists <= self.push_radius
        
        agent_dirs = state.payload_pos - state.p_pos[:self.num_good_agents]
        agent_dirs = agent_dirs / (jnp.linalg.norm(agent_dirs, axis=1, keepdims=True) + 1e-8)
        adv_dirs = state.payload_pos - state.p_pos[self.num_good_agents:]
        adv_dirs = adv_dirs / (jnp.linalg.norm(adv_dirs, axis=1, keepdims=True) + 1e-8)
        
        agent_vel_proj = jnp.sum(state.p_vel[:self.num_good_agents] * agent_dirs, axis=1)
        adv_vel_proj = jnp.sum(state.p_vel[self.num_good_agents:] * adv_dirs, axis=1)
        agent_vel_proj = jnp.maximum(agent_vel_proj, 0)
        adv_vel_proj = jnp.maximum(adv_vel_proj, 0)
        
        agent_force_mag = agent_pushing * (1.0 + agent_vel_proj)
        adv_force_mag = adv_pushing * (1.0 + adv_vel_proj)
        
        agent_push_dir = state.adversary_goal_zone - state.payload_pos
        agent_push_dir = agent_push_dir / (jnp.linalg.norm(agent_push_dir) + 1e-8)
        adv_push_dir = state.agent_goal_zone - state.payload_pos
        adv_push_dir = adv_push_dir / (jnp.linalg.norm(adv_push_dir) + 1e-8)
        
        agent_total_force = jnp.sum(agent_force_mag) * agent_push_dir
        adv_total_force = jnp.sum(adv_force_mag) * adv_push_dir
        net_force = agent_total_force + adv_total_force
        
        return net_force, agent_pushing, adv_pushing
    
    @partial(jax.jit, static_argnums=[0])
    def _apply_proximity_forces(self, payload_pos: chex.Array, payload_vel: chex.Array, net_force: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Apply proximity-based push forces to payload."""
        acceleration = net_force / self.payload_mass
        new_vel = payload_vel + acceleration * self.dt
        damping = 0.95
        new_vel = new_vel * damping
        max_payload_speed = 2.0
        speed = jnp.linalg.norm(new_vel)
        new_vel = jnp.where(speed > max_payload_speed, new_vel / speed * max_payload_speed, new_vel)
        new_pos = payload_pos + new_vel * self.dt
        return new_pos, new_vel
    
    @partial(jax.jit, static_argnums=[0])
    def _block_payload_entry(self, payload_pos: chex.Array, prev_payload_pos: chex.Array,
                             button_toggled: chex.Array, agent_goal_zone: chex.Array, adversary_goal_zone: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Block payload from entering locked zones."""
        # If zone blocking is disabled, never block
        if not self.zone_blocking:
            return payload_pos, jnp.array(False)
        
        prev_dist_agent = jnp.linalg.norm(prev_payload_pos - agent_goal_zone)
        curr_dist_agent = jnp.linalg.norm(payload_pos - agent_goal_zone)
        entering_agent = jnp.logical_and(prev_dist_agent >= self.zone_size, curr_dist_agent < self.zone_size)
        
        prev_dist_adv = jnp.linalg.norm(prev_payload_pos - adversary_goal_zone)
        curr_dist_adv = jnp.linalg.norm(payload_pos - adversary_goal_zone)
        entering_adv = jnp.logical_and(prev_dist_adv >= self.zone_size, curr_dist_adv < self.zone_size)
        
        block_agent = jnp.logical_and(entering_agent, ~button_toggled[0])
        block_adv = jnp.logical_and(entering_adv, ~button_toggled[1])
        should_block = jnp.logical_or(block_agent, block_adv)
        
        return jnp.where(should_block, prev_payload_pos, payload_pos), should_block
    
    @partial(jax.jit, static_argnums=[0])
    def _get_melee_actions(self, actions: dict) -> chex.Array:
        """Extract melee attack actions."""
        return jnp.array([
            actions[name].squeeze() == 5 if self.action_type == "Discrete" else actions[name][5] > 0.5
            for name in self.agents
        ])
    
    @partial(jax.jit, static_argnums=[0])
    def _process_combat(self, key: chex.PRNGKey, state: State, melee_actions: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Process melee combat."""
        pos = state.p_pos
        dists = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        in_range = dists <= self.melee_range
        is_agent_team = jnp.arange(self.num_actors) < self.num_good_agents
        
        valid_targets = jnp.zeros((self.num_actors, self.num_actors), dtype=bool)
        for i in range(self.num_actors):
            for j in range(self.num_actors):
                is_opponent = is_agent_team[i] != is_agent_team[j]
                valid_targets = valid_targets.at[i, j].set(melee_actions[i] & in_range[i, j] & is_opponent)
        
        attacks_received = jnp.sum(valid_targets, axis=0)
        attacked_mask = jnp.any(valid_targets, axis=1)
        got_attacked_mask = attacks_received > 0
        new_hp = jnp.maximum(state.agent_hp - attacks_received, 0)
        
        return new_hp, attacked_mask, got_attacked_mask
    
    @partial(jax.jit, static_argnums=[0])
    def _respawn_agents(self, key: chex.PRNGKey, state: State, new_hp: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Respawn agents with 0 HP."""
        needs_respawn = new_hp <= 0
        respawn_offsets = jax.random.uniform(key, (self.num_actors, 2), minval=-self.spawn_cluster_radius, maxval=self.spawn_cluster_radius)
        is_agent_team = jnp.arange(self.num_actors) < self.num_good_agents
        respawn_centers = jnp.where(is_agent_team[:, None], state.agent_spawn_center[None, :], state.adversary_spawn_center[None, :])
        respawn_pos = respawn_centers + respawn_offsets
        new_pos = jnp.where(needs_respawn[:, None], respawn_pos, state.p_pos)
        new_hp = jnp.where(needs_respawn, self.max_hp, new_hp)
        return new_pos, new_hp, needs_respawn
    
    @partial(jax.jit, static_argnums=[0])
    def _update_hill_ownership(self, state: State) -> chex.Array:
        """Update hill ownership based on agent proximity."""
        new_owner = state.hill_owner.copy()
        for hill_idx in range(2):
            hill_pos = state.hill_pos[hill_idx]
            dists = jnp.linalg.norm(state.p_pos - hill_pos, axis=1)
            touching = dists <= self.hill_radius + self.agent_size
            agent_touching = jnp.any(touching[:self.num_good_agents])
            adv_touching = jnp.any(touching[self.num_good_agents:])
            new_owner = new_owner.at[hill_idx].set(
                jax.lax.select(agent_touching & ~adv_touching, 0,
                    jax.lax.select(adv_touching & ~agent_touching, 1, new_owner[hill_idx]))
            )
        return new_owner
    
    @partial(jax.jit, static_argnums=[0])
    def _check_room_entry(self, state: State) -> Tuple[chex.Array, chex.Array]:
        """Check which agents are in which rooms."""
        dist_to_a = jnp.linalg.norm(state.p_pos - state.part_a_room_pos, axis=1)
        dist_to_b = jnp.linalg.norm(state.p_pos - state.part_b_room_pos, axis=1)
        return dist_to_a <= self.room_radius, dist_to_b <= self.room_radius
    
    @partial(jax.jit, static_argnums=[0])
    def _check_assembler_entry(self, state: State) -> chex.Array:
        """Check which agents are in assembler zone."""
        dist_to_assembler = jnp.linalg.norm(state.p_pos - state.assembler_pos, axis=1)
        return dist_to_assembler <= self.assembler_radius
    
    @partial(jax.jit, static_argnums=[0])
    def _check_assembly_button_press(self, state: State) -> chex.Array:
        """Check which agents are pressing assembly button."""
        dist_to_button = jnp.linalg.norm(state.p_pos - state.assembly_button_pos, axis=1)
        near_button = dist_to_button <= self.assembly_button_radius
        stationary = jnp.linalg.norm(state.p_vel, axis=-1) <= self.vel_eps
        return jnp.logical_and(near_button, stationary)
    
    @partial(jax.jit, static_argnums=[0])
    def _update_inventory(self, state: State, in_room_a: chex.Array, in_room_b: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Update agent inventory."""
        can_pickup_a = in_room_a & ~state.carrying_part_a & ~state.carrying_part_b
        new_carrying_a = state.carrying_part_a | can_pickup_a
        can_pickup_b = in_room_b & state.carrying_part_a & ~state.carrying_part_b
        new_carrying_b = state.carrying_part_b | can_pickup_b
        return new_carrying_a, new_carrying_b
    
    @partial(jax.jit, static_argnums=[0])
    def _update_assembly(self, state: State, in_assembler: chex.Array, carrying_a: chex.Array, 
                         carrying_b: chex.Array, pressing: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """Update assembly state."""
        is_agent_team = jnp.arange(self.num_actors) < self.num_good_agents
        can_agent_deliver = (state.assembler_locked_by == -1) | (state.assembler_locked_by == 0)
        can_adv_deliver = (state.assembler_locked_by == -1) | (state.assembler_locked_by == 1)
        
        delivering_both = in_assembler & carrying_b
        agent_delivering = jnp.sum(delivering_both[:self.num_good_agents] & can_agent_deliver)
        adv_delivering = jnp.sum(delivering_both[self.num_good_agents:] & can_adv_deliver)
        
        new_parts_delivered = state.parts_delivered.copy()
        new_parts_delivered = new_parts_delivered.at[0, 0].add(agent_delivering)
        new_parts_delivered = new_parts_delivered.at[0, 1].add(agent_delivering)
        new_parts_delivered = new_parts_delivered.at[1, 0].add(adv_delivering)
        new_parts_delivered = new_parts_delivered.at[1, 1].add(adv_delivering)
        
        delivered_mask = in_assembler & carrying_b & jnp.where(is_agent_team, can_agent_deliver, can_adv_deliver)
        new_carrying_a = carrying_a & ~delivered_mask
        new_carrying_b = carrying_b & ~delivered_mask
        
        agent_delivered_any = agent_delivering > 0
        adv_delivered_any = adv_delivering > 0
        new_assembler_locked_by = jax.lax.select(
            state.assembler_locked_by == -1,
            jax.lax.select(agent_delivered_any & can_agent_deliver, jnp.array(0, dtype=jnp.int32),
                jax.lax.select(adv_delivered_any & can_adv_deliver, jnp.array(1, dtype=jnp.int32), state.assembler_locked_by)),
            state.assembler_locked_by)
        
        agent_ready = (new_parts_delivered[0, 0] >= 1) & (new_parts_delivered[0, 1] >= 1)
        adv_ready = (new_parts_delivered[1, 0] >= 1) & (new_parts_delivered[1, 1] >= 1)
        agent_pressing = jnp.any(pressing[:self.num_good_agents])
        adv_pressing = jnp.any(pressing[self.num_good_agents:])
        
        new_hold_progress = state.assembly_hold_progress.copy()
        new_hold_progress = new_hold_progress.at[0].set(
            jnp.where(agent_ready & agent_pressing & (new_assembler_locked_by == 0),
                state.assembly_hold_progress[0] + 1, state.assembly_hold_progress[0]))
        new_hold_progress = new_hold_progress.at[1].set(
            jnp.where(adv_ready & adv_pressing & (new_assembler_locked_by == 1),
                state.assembly_hold_progress[1] + 1, state.assembly_hold_progress[1]))
        
        agent_completed = new_hold_progress[0] >= self.assembly_hold_time
        adv_completed = new_hold_progress[1] >= self.assembly_hold_time
        new_team_scores = state.team_scores.copy()
        new_team_scores = new_team_scores.at[0].add(agent_completed.astype(jnp.int32))
        new_team_scores = new_team_scores.at[1].add(adv_completed.astype(jnp.int32))
        
        any_completed = agent_completed | adv_completed
        new_parts_delivered = jnp.where(any_completed, jnp.zeros((2, 2), dtype=jnp.int32), new_parts_delivered)
        new_assembler_locked_by = jnp.where(any_completed, jnp.array(-1, dtype=jnp.int32), new_assembler_locked_by)
        new_hold_progress = jnp.where(any_completed, jnp.zeros((2,), dtype=jnp.int32), new_hold_progress)
        
        return new_carrying_a, new_carrying_b, new_parts_delivered, new_assembler_locked_by, new_hold_progress, new_team_scores
    
    @partial(jax.jit, static_argnums=[0])
    def _ctf_entry_blocks(self, state: State, new_pos: chex.Array, ctf_pressed: chex.Array) -> chex.Array:
        """Compute which agents must be blocked from entering CTF zones."""
        # If zone blocking is disabled, never block
        if not self.zone_blocking:
            return jnp.zeros((self.num_actors,), dtype=bool)
        
        N = self.num_actors
        pos_prev = state.prev_p_pos
        pos_curr = new_pos
        
        def entering(center):
            prev_in = jnp.linalg.norm(pos_prev - center, axis=1) < (self.zone_size - 1e-6)
            curr_in = jnp.linalg.norm(pos_curr - center, axis=1) < (self.zone_size - 1e-6)
            return jnp.logical_and(~prev_in, curr_in)
        
        enter_A = entering(state.agent_goal_zone)
        enter_B = entering(state.adversary_goal_zone)
        
        def per_i(i):
            others = jnp.arange(N) != i
            pressed_by_other_A = jnp.any(jnp.logical_and(ctf_pressed[1], others))
            pressed_by_other_B = jnp.any(jnp.logical_and(ctf_pressed[0], others))
            return jnp.array([pressed_by_other_A, pressed_by_other_B])
        
        pressed_by_other = jax.vmap(per_i)(jnp.arange(N))
        need_for_A = pressed_by_other[:, 1]
        need_for_B = pressed_by_other[:, 0]
        
        block_A = jnp.logical_and(enter_A, ~need_for_A)
        block_B = jnp.logical_and(enter_B, ~need_for_B)
        return jnp.logical_or(block_A, block_B)

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        """Step environment combining all game mechanics."""
        prev_p_pos = state.p_pos
        prev_payload_pos = state.payload_pos
        prev_hill_owner = state.hill_owner
        
        # Extract melee actions (action 5)
        melee_actions = self._get_melee_actions(actions)
        physics_actions = {
            name: jax.lax.select(actions[name].squeeze() == 5, 0, actions[name].squeeze())
            if self.action_type == "Discrete"
            else actions[name][:5]
            for name in self.agents
        }
        
        # ===== CTF Mechanics =====
        opp_flag_in = jnp.linalg.norm(state.flag_p_pos[1] - state.agent_goal_zone) <= (self.zone_size - self.agent_size)
        ego_flag_in = jnp.linalg.norm(state.flag_p_pos[0] - state.adversary_goal_zone) <= (self.zone_size - self.agent_size)
        reset_opp_flag = jax.lax.select(opp_flag_in, state.adversary_goal_zone, state.flag_p_pos[1])
        reset_ego_flag = jax.lax.select(ego_flag_in, state.agent_goal_zone, state.flag_p_pos[0])
        state = state.replace(flag_p_pos=jnp.vstack((reset_ego_flag, reset_opp_flag)))
        
        agent_carrying = jax.lax.select(state.flag_moving[1],
            jnp.full((self.num_good_agents,), False).at[state.flag_carrier[0]].set(True),
            jnp.full((self.num_good_agents,), False))
        adv_carrying = jax.lax.select(state.flag_moving[0],
            jnp.full((self.num_adversaries,), False).at[state.flag_carrier[1]].set(True),
            jnp.full((self.num_adversaries,), False))
        
        agent_can_drop = jnp.linalg.norm(state.p_pos[:self.num_good_agents] - state.agent_goal_zone, axis=1) <= (self.zone_size - self.agent_size)
        adv_can_drop = jnp.linalg.norm(state.p_pos[self.num_good_agents:] - state.adversary_goal_zone, axis=1) <= (self.zone_size - self.agent_size)
        agent_dropping_mask = jnp.logical_and(agent_carrying, agent_can_drop)
        adv_dropping_mask = jnp.logical_and(adv_carrying, adv_can_drop)
        
        flag_moving_after_drop = jnp.hstack((
            jax.lax.select(state.flag_moving[0], ~jnp.logical_and(jnp.any(adv_dropping_mask), state.flag_moving[0]), False),
            jax.lax.select(state.flag_moving[1], ~jnp.logical_and(jnp.any(agent_dropping_mask), state.flag_moving[1]), False),
        ))
        state = state.replace(flag_moving=flag_moving_after_drop)
        
        agent_can_pickup = jnp.linalg.norm(state.p_pos[:self.num_good_agents] - state.adversary_goal_zone, axis=1) <= (self.zone_size - self.agent_size)
        adv_can_pickup = jnp.linalg.norm(state.p_pos[self.num_good_agents:] - state.agent_goal_zone, axis=1) <= (self.zone_size - self.agent_size)
        
        state = state.replace(flag_carrier=jnp.hstack((
            jax.lax.select(state.flag_moving[1], state.flag_carrier[0], jnp.nonzero(agent_can_pickup, size=1, fill_value=0)[0][0]),
            jax.lax.select(state.flag_moving[0], state.flag_carrier[1], jnp.nonzero(adv_can_pickup, size=1, fill_value=0)[0][0]),
        )))
        state = state.replace(flag_moving=jnp.hstack((
            jnp.logical_or(jnp.any(adv_can_pickup), state.flag_moving[0]),
            jnp.logical_or(jnp.any(agent_can_pickup), state.flag_moving[1]),
        )))
        
        # ===== Physics Step =====
        # Order: agents, payload, obstacles, zones, rooms+assembler, hills
        simple_state = SimpleState(
            p_pos=jnp.vstack((
                state.p_pos, state.payload_pos[None, :], state.obs_pos,
                state.agent_goal_zone[None, :], state.adversary_goal_zone[None, :],
                state.part_a_room_pos[None, :], state.part_b_room_pos[None, :], state.assembler_pos[None, :],
                state.hill_pos
            )),
            p_vel=jnp.vstack((
                state.p_vel, state.payload_vel[None, :],
                jnp.zeros((self.num_obstacles + 2 + 3 + 2, 2))
            )),
            done=state.done, step=state.step, goal=None,
            c=jnp.zeros((self.num_actors, self.dim_c)),
        )
        
        simple_obs, simple_state, simple_reward, simple_dones, simple_info = SimpleMPE.step_env(self, key, simple_state, physics_actions)
        
        # Extract positions
        new_p_pos = simple_state.p_pos[:self.num_actors, :]
        new_p_vel = simple_state.p_vel[:self.num_actors, :]
        new_payload_pos = simple_state.p_pos[self.num_actors, :]
        new_payload_vel = simple_state.p_vel[self.num_actors, :]
        
        # ===== CTF Button Gate Blocking =====
        ctf_pressed = self._pressed_ctf_buttons_mask(state)
        ctf_must_block = self._ctf_entry_blocks(state, new_p_pos, ctf_pressed)
        new_p_pos = jnp.where(ctf_must_block[:, None], state.prev_p_pos, new_p_pos)
        new_p_vel = jnp.where(ctf_must_block[:, None], jnp.zeros_like(new_p_vel), new_p_vel)
        
        # ===== Combat and Respawn =====
        key, combat_key = jax.random.split(key)
        new_hp, attacked_mask, got_attacked_mask = self._process_combat(combat_key, state, melee_actions)
        key, respawn_key = jax.random.split(key)
        new_pos, new_hp, respawned_mask = self._respawn_agents(respawn_key, state.replace(p_pos=new_p_pos), new_hp)
        new_p_pos = new_pos
        state = state.replace(p_pos=new_p_pos, p_vel=new_p_vel, agent_hp=new_hp)
        
        # ===== Payload Mechanics =====
        net_force, agent_pushing, adv_pushing = self._compute_push_forces(state)
        new_payload_pos, new_payload_vel = self._apply_proximity_forces(state.payload_pos, state.payload_vel, net_force)
        payload_pressed = self._pressed_payload_buttons_mask(state)
        new_button_toggled = self._update_payload_button_toggles(state.payload_button_toggled, payload_pressed)
        blocked_payload_pos, payload_was_blocked = self._block_payload_entry(
            new_payload_pos, prev_payload_pos, new_button_toggled, state.agent_goal_zone, state.adversary_goal_zone)
        blocked_payload_vel = jnp.where(payload_was_blocked, jnp.zeros(2), new_payload_vel)
        
        # Reset payload if entered zone
        effective_zone_radius = self.zone_size - self.payload_radius
        payload_in_adv_zone = jnp.linalg.norm(blocked_payload_pos - state.adversary_goal_zone) <= effective_zone_radius
        payload_in_agent_zone = jnp.linalg.norm(blocked_payload_pos - state.agent_goal_zone) <= effective_zone_radius
        entered_adv = jnp.logical_and(jnp.linalg.norm(prev_payload_pos - state.adversary_goal_zone) > effective_zone_radius, payload_in_adv_zone)
        entered_agent = jnp.logical_and(jnp.linalg.norm(prev_payload_pos - state.agent_goal_zone) > effective_zone_radius, payload_in_agent_zone)
        entered_any = entered_adv | entered_agent
        starting_payload_pos = (state.agent_goal_zone + state.adversary_goal_zone) / 2
        final_payload_pos = jnp.where(entered_any, starting_payload_pos, blocked_payload_pos)
        final_payload_vel = jnp.where(entered_any, jnp.zeros(2), blocked_payload_vel)
        
        # ===== Assembly Line Mechanics =====
        in_room_a, in_room_b = self._check_room_entry(state)
        in_assembler = self._check_assembler_entry(state)
        assembly_pressing = self._check_assembly_button_press(state)
        new_carrying_a, new_carrying_b = self._update_inventory(state, in_room_a, in_room_b)
        (final_carrying_a, final_carrying_b, new_parts_delivered, new_assembler_locked_by, 
         new_hold_progress, new_team_scores) = self._update_assembly(
            state, in_assembler, new_carrying_a, new_carrying_b, assembly_pressing)
        
        # ===== Hill Ownership =====
        new_hill_owner = self._update_hill_ownership(state)
        
        # ===== Update Flags =====
        new_flag_p_pos = jnp.vstack((
            jax.lax.select(state.flag_moving[0], state.p_pos[state.flag_carrier[1] + self.num_good_agents, :], state.flag_p_pos[0, :]),
            jax.lax.select(state.flag_moving[1], state.p_pos[state.flag_carrier[0], :], state.flag_p_pos[1, :]),
        ))
        new_flag_p_vel = jnp.vstack((
            jax.lax.select(state.flag_moving[0], state.p_vel[state.flag_carrier[1] + self.num_good_agents, :], state.flag_p_vel[0, :]),
            jax.lax.select(state.flag_moving[1], state.p_vel[state.flag_carrier[0], :], state.flag_p_vel[1, :]),
        ))
        
        # ===== Final State Update =====
        state = state.replace(
            p_pos=new_p_pos, p_vel=new_p_vel, done=simple_state.done, step=simple_state.step,
            payload_pos=final_payload_pos, payload_vel=final_payload_vel,
            payload_button_toggled=new_button_toggled,
            flag_p_pos=new_flag_p_pos, flag_p_vel=new_flag_p_vel,
            carrying_part_a=final_carrying_a, carrying_part_b=final_carrying_b,
            parts_delivered=new_parts_delivered, assembler_locked_by=new_assembler_locked_by,
            assembly_hold_progress=new_hold_progress, team_scores=new_team_scores,
            hill_owner=new_hill_owner,
        )
        
        # ===== Rewards (return zeros as requested) =====
        rewards = {a: 0.0 for a in self.agents}
        
        # ===== Subtask Observations =====
        curr_agent_in_adv_zone = jnp.linalg.norm(state.p_pos[:self.num_good_agents] - state.adversary_goal_zone, axis=1) < self.zone_size
        curr_adv_in_agent_zone = jnp.linalg.norm(state.p_pos[self.num_good_agents:] - state.agent_goal_zone, axis=1) < self.zone_size
        curr_agent_in_own_zone = jnp.linalg.norm(state.p_pos[:self.num_good_agents] - state.agent_goal_zone, axis=1) < self.zone_size
        curr_adv_in_own_zone = jnp.linalg.norm(state.p_pos[self.num_good_agents:] - state.adversary_goal_zone, axis=1) < self.zone_size
        
        subtask_obs = self.subtask_obs_fn(
            state, agent_pushing, adv_pushing, payload_pressed, ctf_pressed, assembly_pressing,
            in_room_a, in_room_b, in_assembler, curr_agent_in_adv_zone, curr_adv_in_agent_zone,
            curr_agent_in_own_zone, curr_adv_in_own_zone, agent_carrying, adv_carrying,
            attacked_mask, prev_hill_owner)
        
        subtask_matrix = jnp.stack([subtask_obs[name] for name in self.agents], axis=0)
        new_cum_subtask_obs = state.cum_subtask_obs + subtask_matrix
        state = state.replace(cum_subtask_obs=new_cum_subtask_obs)
        normalized_cum_subtask_obs = new_cum_subtask_obs / self.max_steps
        cum_subtask_obs_dict = {name: normalized_cum_subtask_obs[i] for i, name in enumerate(self.agents)}
        
        info = {
            "agent_pushing": agent_pushing, "adv_pushing": adv_pushing,
            "payload_button_pressed": payload_pressed, "ctf_button_pressed": ctf_pressed,
            "assembly_button_pressing": assembly_pressing,
            "payload_in_adv_zone": payload_in_adv_zone, "payload_in_agent_zone": payload_in_agent_zone,
            "hill_owner": state.hill_owner, "agent_hp": state.agent_hp,
            "attacked": attacked_mask, "got_attacked": got_attacked_mask, "respawned": respawned_mask,
            "carrying_part_a": state.carrying_part_a, "carrying_part_b": state.carrying_part_b,
            "parts_delivered": state.parts_delivered, "team_scores": state.team_scores,
            "subtask_obs": subtask_obs, "cum_subtask_obs": cum_subtask_obs_dict,
            "option_assignment": state.option_assignment,
        }
        
        if not self.random_skills and self.assign_subtasks:
            assigned_indices = state.option_assignment[:, None]
            valid_assign = state.option_assignment >= 0
            assigned_values = jnp.take_along_axis(subtask_matrix, assigned_indices, axis=1).squeeze(axis=1)
            completion_mask = jnp.logical_and(assigned_values > 0.5, valid_assign)
            info["subtask_completed"] = completion_mask
            any_completed = jnp.any(completion_mask)
            state = state.replace(done=jnp.where(any_completed, jnp.ones_like(state.done), state.done))
            simple_dones = {name: jnp.logical_or(done_flag, any_completed) for name, done_flag in simple_dones.items()}
        
        state = state.replace(prev_p_pos=prev_p_pos, prev_payload_pos=prev_payload_pos, prev_hill_owner=prev_hill_owner)
        obs = self.obs_fn(state)
        
        return obs, state, rewards, simple_dones, info

    @partial(jax.jit, static_argnums=[0])
    def obs_fn(self, state: State) -> Dict[str, chex.Array]:
        """Generate observations combining all environment components."""
        out = {}
        
        for i, name in enumerate(self.agents):
            is_agent = i < self.num_good_agents
            idx = i if is_agent else i - self.num_good_agents
            
            # Rotation matrix (agents face toward origin)
            heading_vec = -state.p_pos[i]
            theta = jnp.arctan2(heading_vec[1], heading_vec[0])
            sin = jnp.sin(theta)
            cos = jnp.cos(theta)
            rotmat = jnp.array([[cos, sin], [-sin, cos]])
            
            # Self velocity
            self_vel = (state.p_vel[i] @ rotmat.T)
            
            # Other agents
            if is_agent:
                teammate_pos = jnp.delete(state.p_pos[:self.num_good_agents], idx, axis=0)
                teammate_vel = jnp.delete(state.p_vel[:self.num_good_agents], idx, axis=0)
                opponent_pos = state.p_pos[self.num_good_agents:]
                opponent_vel = state.p_vel[self.num_good_agents:]
                own_goal = state.agent_goal_zone
                opp_goal = state.adversary_goal_zone
            else:
                teammate_pos = jnp.delete(state.p_pos[self.num_good_agents:], idx, axis=0)
                teammate_vel = jnp.delete(state.p_vel[self.num_good_agents:], idx, axis=0)
                opponent_pos = state.p_pos[:self.num_good_agents]
                opponent_vel = state.p_vel[:self.num_good_agents]
                own_goal = state.adversary_goal_zone
                opp_goal = state.agent_goal_zone
            
            teammate_rel_pos = ((teammate_pos - state.p_pos[i]) @ rotmat.T).flatten()
            teammate_rel_vel = ((teammate_vel - state.p_vel[i]) @ rotmat.T).flatten()
            opponent_rel_pos = ((opponent_pos - state.p_pos[i]) @ rotmat.T).flatten()
            opponent_rel_vel = ((opponent_vel - state.p_vel[i]) @ rotmat.T).flatten()
            
            # Payload
            payload_rel_pos = (state.payload_pos - state.p_pos[i]) @ rotmat.T
            payload_rel_vel = state.payload_vel @ rotmat.T
            
            # Goals
            own_goal_rel = (own_goal - state.p_pos[i]) @ rotmat.T
            opp_goal_rel = (opp_goal - state.p_pos[i]) @ rotmat.T
            
            # Flags
            flags_rel_pos = ((state.flag_p_pos - state.p_pos[i]) @ rotmat.T).flatten()
            
            # Rooms and assembler
            room_a_rel = (state.part_a_room_pos - state.p_pos[i]) @ rotmat.T
            room_b_rel = (state.part_b_room_pos - state.p_pos[i]) @ rotmat.T
            assembler_rel = (state.assembler_pos - state.p_pos[i]) @ rotmat.T
            assembly_button_rel = (state.assembly_button_pos - state.p_pos[i]) @ rotmat.T
            
            # Hills
            hills_rel_pos = ((state.hill_pos - state.p_pos[i]) @ rotmat.T).flatten()
            
            # Obstacles
            if self._user_num_obstacles > 0:
                obs_rel_pos = ((state.obs_pos[:self._user_num_obstacles] - state.p_pos[i]) @ rotmat.T).flatten()
            else:
                obs_rel_pos = jnp.array([])
            
            # HP
            own_hp = state.agent_hp[i] / self.max_hp
            all_hp = state.agent_hp / self.max_hp
            
            # Carrying status
            carrying_status = jnp.array([state.carrying_part_a[i], state.carrying_part_b[i]], dtype=jnp.float32)
            
            # Parts delivered
            parts_delivered_flat = state.parts_delivered.flatten().astype(jnp.float32)
            
            # Assembler lock
            assembler_lock = state.assembler_locked_by.astype(jnp.float32) / 1.0
            
            # Assembly hold progress
            hold_progress = state.assembly_hold_progress.astype(jnp.float32) / self.assembly_hold_time
            
            # Team scores
            team_scores_norm = state.team_scores.astype(jnp.float32) / 10.0
            
            # Hill ownership
            hill_owner_obs = (state.hill_owner + 1) / 2.0  # Map -1,0,1 to 0,0.5,1
            
            # Payload buttons
            payload_button_rel_pos = ((state.payload_button_pos - state.p_pos[i]) @ rotmat.T).flatten()
            payload_button_toggled = state.payload_button_toggled.astype(jnp.float32)
            
            # CTF buttons
            ctf_button_rel_pos = ((state.ctf_button_pos - state.p_pos[i]) @ rotmat.T).flatten()
            ctf_pressed = self._pressed_ctf_buttons_mask(state)
            ctf_pressed_any = jnp.any(ctf_pressed, axis=1).astype(jnp.float32)  # (2,)
            
            # Option observations
            payload_to_opp = jnp.linalg.norm(state.payload_pos - opp_goal)
            total_dist = jnp.linalg.norm(own_goal - opp_goal)
            payload_progress = 1.0 - (payload_to_opp / (total_dist + 1e-8))
            payload_speed = jnp.linalg.norm(state.payload_vel)
            num_own_hills = jnp.sum(state.hill_owner == (0 if is_agent else 1))
            num_enemy_hills = jnp.sum(state.hill_owner == (1 if is_agent else 0))
            own_team = 0 if is_agent else 1
            own_parts_a = state.parts_delivered[own_team, 0]
            own_parts_b = state.parts_delivered[own_team, 1]
            own_ready = (own_parts_a >= 1) & (own_parts_b >= 1)
            own_hold = state.assembly_hold_progress[own_team]
            
            option_obs = jnp.array([
                payload_progress, payload_speed,
                num_own_hills / 2.0, num_enemy_hills / 2.0,
                own_hp,
                own_parts_a / 1.0, own_parts_b / 1.0,
                own_ready.astype(jnp.float32), own_hold / self.assembly_hold_time,
                0.0  # placeholder
            ])
            
            obs = jnp.concatenate([
                self_vel,
                teammate_rel_pos, teammate_rel_vel,
                opponent_rel_pos, opponent_rel_vel,
                payload_rel_pos, payload_rel_vel,
                own_goal_rel, opp_goal_rel,
                flags_rel_pos,
                room_a_rel, room_b_rel, assembler_rel, assembly_button_rel,
                hills_rel_pos,
                obs_rel_pos,
                jnp.array([own_hp]), all_hp,
                carrying_status,
                parts_delivered_flat,
                jnp.array([assembler_lock]),
                hold_progress,
                team_scores_norm,
                hill_owner_obs,
                payload_button_rel_pos,
                ctf_button_rel_pos,
                payload_button_toggled,
                ctf_pressed_any,
                option_obs,
            ])
            
            out[name] = obs
        
        return out

    @partial(jax.jit, static_argnums=[0])
    def subtask_obs_fn(self, state: State, agent_pushing: chex.Array, adv_pushing: chex.Array,
                       payload_pressed: chex.Array, ctf_pressed: chex.Array, assembly_pressing: chex.Array,
                       in_room_a: chex.Array, in_room_b: chex.Array, in_assembler: chex.Array,
                       curr_agent_in_adv_zone: chex.Array, curr_adv_in_agent_zone: chex.Array,
                       curr_agent_in_own_zone: chex.Array, curr_adv_in_own_zone: chex.Array,
                       agent_carrying: chex.Array, adv_carrying: chex.Array,
                       attacked_mask: chex.Array, prev_hill_owner: chex.Array) -> Dict[str, chex.Array]:
        """Generate combined subtask observations (22 total)."""
        # Compute collision distances
        pairwise_dists = jnp.linalg.norm(state.p_pos[:, None, :] - state.p_pos[None, :, :], axis=2)
        collision_threshold = 2 * self.agent_size
        collision_threshold_payload = self.agent_size + self.payload_radius
        
        # Payload pushing/contacting
        agent_dist_payload = jnp.linalg.norm(state.p_pos[:self.num_good_agents] - state.payload_pos, axis=1)
        adv_dist_payload = jnp.linalg.norm(state.p_pos[self.num_good_agents:] - state.payload_pos, axis=1)
        agent_colliding_payload = agent_dist_payload <= collision_threshold_payload
        adv_colliding_payload = adv_dist_payload <= collision_threshold_payload
        agent_in_push_range = agent_dist_payload <= self.push_radius
        adv_in_push_range = adv_dist_payload <= self.push_radius
        agent_pushing_only = agent_in_push_range & ~agent_colliding_payload
        adv_pushing_only = adv_in_push_range & ~adv_colliding_payload
        
        # Hills distances
        all_dists_to_hills = jnp.linalg.norm(state.p_pos[:, None, :] - state.hill_pos[None, :, :], axis=2)
        near_each_hill = all_dists_to_hills <= self.hill_radius + self.agent_size
        
        out = {}
        for i, name in enumerate(self.agents):
            is_agent = i < self.num_good_agents
            idx = i if is_agent else i - self.num_good_agents
            
            if is_agent:
                # Collisions
                own_dists = jnp.concatenate([
                    pairwise_dists[i, :idx],
                    pairwise_dists[i, idx+1:self.num_good_agents]
                ]) if self.num_good_agents > 1 else jnp.array([])
                other_dists = pairwise_dists[i, self.num_good_agents:]
                
                # Payload Escort subtasks (5)
                pushing_payload = agent_pushing_only[idx] if idx < len(agent_pushing_only) else False
                near_own_goal = jnp.linalg.norm(state.p_pos[i] - state.agent_goal_zone) <= self.zone_size
                near_opp_goal = jnp.linalg.norm(state.p_pos[i] - state.adversary_goal_zone) <= self.zone_size
                contacting_payload = agent_colliding_payload[idx] if idx < len(agent_colliding_payload) else False
                pressing_payload_button = jnp.logical_or(payload_pressed[0], payload_pressed[1])  # buttons near agent zone
                
                # CTF subtasks (4)
                pressing_ctf_button = ctf_pressed[0, i]
                in_opp_zone = curr_agent_in_adv_zone[idx]
                in_own_zone = curr_agent_in_own_zone[idx]
                carrying_opp_flag = agent_carrying[idx] if idx < len(agent_carrying) else False
                
                # Assembly Line subtasks (6)
                near_part_a = in_room_a[i]
                near_part_b = in_room_b[i]
                carrying_a = state.carrying_part_a[i]
                carrying_b = state.carrying_part_b[i]
                at_assembler = in_assembler[i]
                pressing_assembly_button = assembly_pressing[i]
                
                # King of Hill subtasks (4)
                agent_near_hill = near_each_hill[i]
                near_own_hill = jnp.any(agent_near_hill & (state.hill_owner == 0))
                near_enemy_hill = jnp.any(agent_near_hill & (state.hill_owner == 1))
                near_neutral_hill = jnp.any(agent_near_hill & (state.hill_owner == -1))
                attacked_opponent = attacked_mask[i]
                
                # Collision subtasks (3)
                collide_obs = jnp.any(jnp.linalg.norm(state.p_pos[i] - state.obs_pos, axis=1) <= (self.agent_size + self.obstacle_size)) if self._user_num_obstacles > 0 else False
                collide_teammate = jnp.any(own_dists <= collision_threshold) if len(own_dists) > 0 else False
                collide_opponent = jnp.any(other_dists <= collision_threshold)
            else:
                # Collisions
                adv_start = self.num_good_agents
                own_dists = jnp.concatenate([
                    pairwise_dists[i, adv_start:i],
                    pairwise_dists[i, i+1:]
                ]) if self.num_adversaries > 1 else jnp.array([])
                other_dists = pairwise_dists[i, :self.num_good_agents]
                
                # Payload Escort subtasks
                pushing_payload = adv_pushing_only[idx]
                near_own_goal = jnp.linalg.norm(state.p_pos[i] - state.adversary_goal_zone) <= self.zone_size
                near_opp_goal = jnp.linalg.norm(state.p_pos[i] - state.agent_goal_zone) <= self.zone_size
                contacting_payload = adv_colliding_payload[idx]
                pressing_payload_button = jnp.logical_or(payload_pressed[2], payload_pressed[3])  # buttons near adv zone
                
                # CTF subtasks
                pressing_ctf_button = ctf_pressed[1, i]
                in_opp_zone = curr_adv_in_agent_zone[idx]
                in_own_zone = curr_adv_in_own_zone[idx]
                carrying_opp_flag = adv_carrying[idx]
                
                # Assembly Line subtasks
                near_part_a = in_room_a[i]
                near_part_b = in_room_b[i]
                carrying_a = state.carrying_part_a[i]
                carrying_b = state.carrying_part_b[i]
                at_assembler = in_assembler[i]
                pressing_assembly_button = assembly_pressing[i]
                
                # King of Hill subtasks
                agent_near_hill = near_each_hill[i]
                near_own_hill = jnp.any(agent_near_hill & (state.hill_owner == 1))
                near_enemy_hill = jnp.any(agent_near_hill & (state.hill_owner == 0))
                near_neutral_hill = jnp.any(agent_near_hill & (state.hill_owner == -1))
                attacked_opponent = attacked_mask[i]
                
                # Collision subtasks
                collide_obs = jnp.any(jnp.linalg.norm(state.p_pos[i] - state.obs_pos, axis=1) <= (self.agent_size + self.obstacle_size)) if self._user_num_obstacles > 0 else False
                collide_teammate = jnp.any(own_dists <= collision_threshold) if len(own_dists) > 0 else False
                collide_opponent = jnp.any(other_dists <= collision_threshold)
            
            # Combine all 22 subtasks
            obs_vec = jnp.array([
                pushing_payload, near_own_goal, near_opp_goal, contacting_payload, pressing_payload_button,  # PE (5)
                pressing_ctf_button, in_opp_zone, in_own_zone, carrying_opp_flag,  # CTF (4)
                near_part_a, near_part_b, carrying_a, carrying_b, at_assembler, pressing_assembly_button,  # AL (6)
                near_own_hill, near_enemy_hill, near_neutral_hill, attacked_opponent,  # KoH (4)
                collide_obs, collide_teammate, collide_opponent,  # Collisions (3)
            ], dtype=jnp.float32)
            
            out[name] = obs_vec
        
        return out

    @partial(jax.jit, static_argnums=[0])
    def get_rewards(self, state: State) -> Dict[str, float]:
        """Return zero rewards for all agents."""
        return {a: 0.0 for a in self.agents}

