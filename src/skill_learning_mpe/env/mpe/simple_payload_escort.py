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

from ..spaces import Box


@struct.dataclass
class State:
    """
    Payload Escort State.

    Two teams compete to push a heavy neutral payload into the opponent's goal zone.
    The payload requires multiple agents pushing simultaneously to move effectively.
    
    Button mechanics: Each team has 2 buttons near their own goal zone. The opponent's
    goal zone is locked until at least one of the corresponding buttons has been pressed
    (toggle - press once to unlock, don't need to stay on the button).
    """

    # === Agent state ===
    p_pos: chex.Array  # (num_actors, 2) agent positions
    p_vel: chex.Array  # (num_actors, 2) agent velocities
    done: chex.Array  # (num_actors,) done flags
    step: int  # current step

    # === Payload state ===
    payload_pos: chex.Array  # (2,) payload position
    payload_vel: chex.Array  # (2,) payload velocity

    # === Static environment ===
    agent_goal_zone: chex.Array  # (2,) Team A's goal zone (where Team B wants to push payload)
    adversary_goal_zone: chex.Array  # (2,) Team B's goal zone (where Team A wants to push payload)
    obs_pos: chex.Array  # (num_obstacles, 2) obstacle positions

    # === Button state ===
    button_pos: chex.Array  # (4, 2) button positions [2 near agent zone, 2 near adversary zone]
    button_toggled: chex.Array  # (2,) whether each zone is unlocked [agent_zone_unlocked, adversary_zone_unlocked]

    # === Agent metadata ===
    agent_names: List[str] = struct.field(pytree_node=False)
    adversary_names: List[str] = struct.field(pytree_node=False)

    # === Extended fields for skills ===
    prev_p_pos: chex.Array  # (num_actors, 2) previous positions
    prev_payload_pos: chex.Array  # (2,) previous payload position
    option_assignment: chex.Array  # skill assignments
    cum_subtask_obs: chex.Array  # (num_actors, num_subtasks) cumulative subtask observations


class SimplePayloadEscort(SimpleMPE):
    """
    Payload Escort Environment.

    A heavy neutral payload spawns at the center of the map. Teams compete to push
    it into the opponent's goal zone.

    Key Mechanics:
    - The payload is heavy and requires multiple agents to push effectively
    - Agents within `push_radius` of the payload contribute to pushing
    - Net force from all contacting agents determines payload movement
    - Opponents can block or push in opposite direction
    - Each team has 2 buttons near their own goal zone. The opponent's goal zone
      is locked until at least one button is pressed (toggle mechanism).

    Parameters
    ----------
    num_good_agents : int
        Number of agents on Team A (default 3)
    num_adversaries : int  
        Number of agents on Team B (default 3)
    num_obstacles : int
        Number of obstacles (default 0)
    payload_mass : float
        Mass of the payload relative to agents (default 10.0)
    payload_radius : float
        Collision radius of the payload (default 2.0)
    push_radius : float
        Distance within which agents can push the payload (default 3.0)
    zone_size : float
        Radius of goal zones (default 5.0)
    dist_between_zones : float
        Distance between the two goal zones (default 20.0)
    button_radius : float
        Distance threshold for pressing a button (default 1.5)
    vel_eps : float
        Velocity magnitude considered stationary for button press (default 2.0)
    button_offset : float
        Distance to place buttons from goal zones perpendicular to zone line (default 3.0)
    num_skills : int
        Number of discrete skills for option assignment (default 10)
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
        num_obstacles: int = 0,
        action_type=DISCRETE_ACT,
        payload_mass: float = 10.0,
        payload_radius: float = 2.0,
        push_radius: float = 3.0,
        zone_size: float = 5.0,
        dist_between_zones: float = 20.0,
        agent_size: float = 1.0,
        obstacle_size: float = 1.5,
        zero_sum: bool = True,
        random_start: bool = True,
        init_agent_everywhere: bool = False,
        button_radius: float = 1.5,
        vel_eps: float = 2.0,
        button_offset: float = 3.0,
        num_skills: int = 8,
        random_skills: bool = False,
        assign_subtasks: bool = False,
        **kwargs,
    ):
        self.zone_size = zone_size
        self.agent_size = agent_size
        self.obstacle_size = obstacle_size if num_obstacles > 0 else 0.001  # Minimal size for dummy
        self.num_good_agents = num_good_agents
        self.num_adversaries = num_adversaries
        self.dist_between_zones = dist_between_zones
        # Always have at least 1 obstacle internally (dummy if none requested)
        self._user_num_obstacles = num_obstacles
        self.num_obstacles = max(num_obstacles, 1)  # At least 1 for array operations
        self.action_type = action_type
        self.max_steps = CTF_MAX_STEPS
        self.zero_sum = zero_sum
        self.random_start = random_start
        self.init_agent_everywhere = init_agent_everywhere

        # Payload parameters
        self.payload_mass = payload_mass
        self.payload_radius = payload_radius
        self.push_radius = push_radius

        # Button parameters
        self.button_radius = button_radius
        self.vel_eps = vel_eps
        self.button_offset = button_offset

        # Skill parameters
        self.num_skills = num_skills
        self.random_skills = random_skills
        self.assign_subtasks = assign_subtasks
        self.num_subtasks = 7  # Only directly controllable subtasks

        # Agent names
        self.good_agents = [f"agent_{i}" for i in range(num_good_agents)]
        self.adversaries = [f"adversary_{i}" for i in range(num_adversaries)]
        agents = self.good_agents + self.adversaries

        # Landmarks: payload + obstacles + zones
        # Payload is first landmark (index num_actors), then obstacles, then zones
        obs = [f"obstacle_{i}" for i in range(self.num_obstacles)]
        zones = ["agent_goal_zone", "adversary_goal_zone"]
        landmarks = ["payload"] + obs + zones

        if action_type == "Discrete":
            from ..spaces import Discrete
            self.action_spaces = {i: Discrete(5) for i in agents}
        elif action_type == "Continuous":
            self.action_spaces = {i: Box(-1, 1, (5,)) for i in agents}

        self.num_actors = num_good_agents + num_adversaries
        # Landmarks: 1 payload + obstacles + 2 zones
        num_landmarks = 1 + self.num_obstacles + 2

        self.agent_range = jnp.arange(self.num_actors)

        # Collision setup: agents, payload, and obstacles collide; zones don't
        collides = jnp.concatenate(
            [
                jnp.full((self.num_actors,), True),   # agents collide
                jnp.full((1,), True),                  # payload collides
                jnp.full((self.num_obstacles,), True), # obstacles collide
                jnp.full((2,), False),                 # zones don't collide
            ]
        )

        # Radii for collisions
        rad = jnp.concatenate(
            [
                jnp.full((self.num_actors,), agent_size),
                jnp.full((1,), payload_radius),        # payload radius
                jnp.full((self.num_obstacles,), self.obstacle_size),
                jnp.full((2,), zone_size),
            ]
        )

        # Mass: agents have mass 1, payload is heavy
        mass = jnp.concatenate(
            [
                jnp.full((self.num_actors,), 1.0),     # agent mass
                jnp.full((1,), payload_mass),          # payload mass
                jnp.full((self.num_obstacles,), 1.0),  # obstacle mass (immovable anyway)
                jnp.full((2,), 1.0),                   # zone mass (immovable)
            ]
        )

        # Moveable: agents and payload can move
        moveable = jnp.concatenate(
            [
                jnp.full((self.num_actors,), True),    # agents move
                jnp.full((1,), True),                  # payload moves
                jnp.full((self.num_obstacles,), False), # obstacles don't move
                jnp.full((2,), False),                 # zones don't move
            ]
        )

        # Max speed: -1 means no limit, 0 means clamped to 0
        max_speed = jnp.concatenate(
            [
                jnp.full((self.num_actors,), -1.0),    # agents have no speed limit
                jnp.full((1,), -1.0),                  # payload has no speed limit
                jnp.full((self.num_obstacles,), 0.0),  # obstacles don't move
                jnp.full((2,), 0.0),                   # zones don't move
            ]
        )

        # Observation dimensions:
        # self_vel (2) + teammate_pos_vel (2*(num_good-1)*2) + opponent_pos_vel (2*num_adv*2) +
        # payload_rel_pos (2) + payload_vel (2) + own_goal_rel (2) + opp_goal_rel (2) +
        # button_rel_pos (4*2) + button_toggled (2) + option_obs (2: payload_progress, payload_speed)
        base_obs_dim = (
            2  # self_vel
            + (self.num_actors - 1) * 4  # other agents pos+vel
            + 2  # payload rel pos
            + 2  # payload vel
            + 2  # own goal rel pos
            + 2  # opponent goal rel pos
            + 4 * 2  # button relative positions (4 buttons)
            + 2  # button toggled states (2 zones)
        )
        option_obs_dim = 2  # additional option observations (payload_progress, payload_speed)

        self.observation_spaces = {
            i: Box(-jnp.inf, jnp.inf, (base_obs_dim + option_obs_dim,)) for i in agents
        }

        if self.assign_subtasks:
            assert self.num_skills == 7, "Number of skills must be 7 when assigning subtasks"
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
    def _place_buttons(
        self, agent_goal_zone: chex.Array, adversary_goal_zone: chex.Array
    ) -> chex.Array:
        """Place two buttons near each goal zone, symmetrically positioned.

        Buttons are positioned perpendicular to the line connecting the zones.
        - Buttons 0, 1: near agent_goal_zone (Team A's zone) - pressing unlocks adversary_goal_zone
        - Buttons 2, 3: near adversary_goal_zone (Team B's zone) - pressing unlocks agent_goal_zone

        Returns:
            button_pos: (4, 2) array of button positions
        """
        vec = adversary_goal_zone - agent_goal_zone
        dist = jnp.linalg.norm(vec) + 1e-8
        u = vec / dist  # unit vector from A to B
        v = jnp.array([-u[1], u[0]])  # perpendicular (left of u)

        # Place buttons near each zone, offset perpendicular to the zone line
        # Buttons near agent zone (will unlock adversary zone when pressed)
        btn_a1 = agent_goal_zone + v * (self.zone_size + self.button_offset)
        btn_a2 = agent_goal_zone - v * (self.zone_size + self.button_offset)
        
        # Buttons near adversary zone (will unlock agent zone when pressed)
        btn_b1 = adversary_goal_zone + v * (self.zone_size + self.button_offset)
        btn_b2 = adversary_goal_zone - v * (self.zone_size + self.button_offset)

        return jnp.vstack((btn_a1, btn_a2, btn_b1, btn_b2))

    @partial(jax.jit, static_argnums=[0])
    def _pressed_mask(self, state: State) -> chex.Array:
        """
        Returns pressed[k] indicating whether any agent is currently pressing button k.
        A press is within button_radius and stationary (||v|| <= vel_eps).
        
        Returns:
            pressed: (4,) boolean array for each button
        """
        btn = state.button_pos  # (4, 2)
        pos = state.p_pos  # (N, 2)
        vel = state.p_vel  # (N, 2)
        
        # Distance from each agent to each button
        dists = jnp.linalg.norm(btn[:, None, :] - pos[None, :, :], axis=-1)  # (4, N)
        near = dists <= self.button_radius
        still = jnp.linalg.norm(vel, axis=-1) <= self.vel_eps  # (N,)
        
        # An agent presses a button if near and stationary
        pressing = jnp.logical_and(near, still[None, :])  # (4, N)
        
        # Any agent pressing each button
        return jnp.any(pressing, axis=1)  # (4,)

    @partial(jax.jit, static_argnums=[0])
    def _update_button_toggles(
        self, button_toggled: chex.Array, pressed: chex.Array
    ) -> chex.Array:
        """
        Update button toggle states based on current presses.
        
        Button mapping (opposite unlock):
        - Buttons 0, 1 (near agent zone): toggle unlock for adversary_goal_zone (index 1)
        - Buttons 2, 3 (near adversary zone): toggle unlock for agent_goal_zone (index 0)
        
        Args:
            button_toggled: (2,) current toggle states [agent_zone_unlocked, adversary_zone_unlocked]
            pressed: (4,) which buttons are currently being pressed
            
        Returns:
            new_toggled: (2,) updated toggle states
        """
        # If any button near agent zone pressed, unlock adversary zone
        unlock_adversary = jnp.logical_or(pressed[0], pressed[1])
        # If any button near adversary zone pressed, unlock agent zone
        unlock_agent = jnp.logical_or(pressed[2], pressed[3])
        
        new_toggled = jnp.array([
            jnp.logical_or(button_toggled[0], unlock_agent),  # agent zone unlocked
            jnp.logical_or(button_toggled[1], unlock_adversary),  # adversary zone unlocked
        ])
        
        return new_toggled

    @partial(jax.jit, static_argnums=[0])
    def _block_payload_entry(
        self, 
        payload_pos: chex.Array, 
        prev_payload_pos: chex.Array,
        button_toggled: chex.Array,
        agent_goal_zone: chex.Array,
        adversary_goal_zone: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Block payload from entering a zone if the corresponding button hasn't been toggled.
        
        Args:
            payload_pos: Current payload position
            prev_payload_pos: Previous payload position
            button_toggled: (2,) [agent_zone_unlocked, adversary_zone_unlocked]
            agent_goal_zone: Position of agent goal zone
            adversary_goal_zone: Position of adversary goal zone
            
        Returns:
            new_payload_pos: Potentially reverted position
            was_blocked: Whether the payload was blocked
        """
        # Check if payload is entering agent zone
        # Block at the outer zone boundary (zone_size) to keep payload fully outside
        prev_dist_to_agent = jnp.linalg.norm(prev_payload_pos - agent_goal_zone)
        curr_dist_to_agent = jnp.linalg.norm(payload_pos - agent_goal_zone)
        entering_agent_zone = jnp.logical_and(
            prev_dist_to_agent >= self.zone_size,
            curr_dist_to_agent < self.zone_size
        )
        
        # Check if payload is entering adversary zone
        prev_dist_to_adv = jnp.linalg.norm(prev_payload_pos - adversary_goal_zone)
        curr_dist_to_adv = jnp.linalg.norm(payload_pos - adversary_goal_zone)
        entering_adv_zone = jnp.logical_and(
            prev_dist_to_adv >= self.zone_size,
            curr_dist_to_adv < self.zone_size
        )
        
        # Block if entering a locked zone
        block_agent_entry = jnp.logical_and(entering_agent_zone, ~button_toggled[0])
        block_adv_entry = jnp.logical_and(entering_adv_zone, ~button_toggled[1])
        should_block = jnp.logical_or(block_agent_entry, block_adv_entry)
        
        # Revert to previous position if blocked
        new_payload_pos = jnp.where(should_block, prev_payload_pos, payload_pos)
        
        return new_payload_pos, should_block

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        key, agent_key, adv_key, obs_key, option_key = jax.random.split(key, 5)

        if self.random_start:
            # Random zone positions
            init_key, key = jax.random.split(key)
            agent_goal_zone = jax.random.uniform(
                init_key,
                (2,),
                minval=-self.dist_between_zones / 2,
                maxval=self.dist_between_zones / 2,
            )

            # Adversary goal zone at fixed distance
            adv_angle_key, key = jax.random.split(key)
            adv_angle = jax.random.uniform(adv_angle_key, (1,), minval=0, maxval=2 * jnp.pi)
            adversary_goal_zone = (
                jnp.array([
                    self.dist_between_zones * jnp.cos(adv_angle),
                    self.dist_between_zones * jnp.sin(adv_angle),
                ]).flatten()
                + agent_goal_zone
            )

            # Payload at center between zones
            payload_pos = (agent_goal_zone + adversary_goal_zone) / 2

            # Obstacles between zones
            between_zone_vec = adversary_goal_zone - agent_goal_zone
            if self._user_num_obstacles > 0:
                vecs = jnp.vstack([between_zone_vec] * self._user_num_obstacles)
                spacing = jnp.linspace(0.0, 1.0, self._user_num_obstacles + 2)[1:-1][:, None]
                obs_pos = agent_goal_zone + vecs * spacing
                shifts = jax.random.uniform(
                    obs_key,
                    (self._user_num_obstacles, 1),
                    minval=-self.zone_size / 2,
                    maxval=self.zone_size / 2,
                )
                perp = jnp.array([-between_zone_vec[1], between_zone_vec[0]])
                perp = perp / (jnp.linalg.norm(perp) + 1e-8)
                obs_pos = obs_pos + shifts * perp
            else:
                # Dummy obstacle placed far away (won't affect gameplay)
                obs_pos = jnp.array([[1e6, 1e6]])

            if self.init_agent_everywhere:
                # Initialize anywhere in the map, inside the circle that encloses both zones
                agent_everywhere_radius = jnp.linalg.norm(
                    adversary_goal_zone - agent_goal_zone
                )
                center_pos = (agent_goal_zone + adversary_goal_zone) / 2
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
                # Agents start near their goal zone
                agent_init_radius = self.zone_size / 2
                agent_angles = jax.random.uniform(
                    agent_key, (self.num_good_agents,), minval=0, maxval=2 * jnp.pi
                )
                agent_init_pos = agent_goal_zone + agent_init_radius * jnp.stack(
                    [jnp.cos(agent_angles), jnp.sin(agent_angles)], axis=1
                )

                # Adversaries start near their goal zone
                adv_angles = jax.random.uniform(
                    adv_key, (self.num_adversaries,), minval=0, maxval=2 * jnp.pi
                )
                adversary_init_pos = adversary_goal_zone + agent_init_radius * jnp.stack(
                    [jnp.cos(adv_angles), jnp.sin(adv_angles)], axis=1
                )
        else:
            # Fixed positions
            agent_goal_zone = jnp.array([-self.dist_between_zones / 2, 0.0])
            adversary_goal_zone = jnp.array([self.dist_between_zones / 2, 0.0])
            payload_pos = jnp.array([0.0, 0.0])

            # Fixed agent positions in a ring
            agent_angles = jnp.linspace(0, 2 * jnp.pi, self.num_good_agents, endpoint=False)
            agent_init_radius = self.zone_size / 2
            agent_init_pos = agent_goal_zone + agent_init_radius * jnp.stack(
                [jnp.cos(agent_angles), jnp.sin(agent_angles)], axis=1
            )

            adv_angles = jnp.linspace(-jnp.pi, jnp.pi, self.num_adversaries, endpoint=False)
            adversary_init_pos = adversary_goal_zone + agent_init_radius * jnp.stack(
                [jnp.cos(adv_angles), jnp.sin(adv_angles)], axis=1
            )

            # Obstacles
            if self._user_num_obstacles > 0:
                between_zone_vec = adversary_goal_zone - agent_goal_zone
                vecs = jnp.vstack([between_zone_vec] * self._user_num_obstacles)
                spacing = jnp.linspace(0.0, 1.0, self._user_num_obstacles + 2)[1:-1][:, None]
                obs_pos = agent_goal_zone + vecs * spacing
            else:
                # Dummy obstacle placed far away (won't affect gameplay)
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

        # Place buttons
        button_pos = self._place_buttons(agent_goal_zone, adversary_goal_zone)
        
        # Initialize button toggled states (both zones locked initially)
        button_toggled = jnp.array([False, False])

        p_pos = jnp.vstack((agent_init_pos, adversary_init_pos))

        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_actors, 2)),
            done=jnp.full((self.num_actors,), False),
            step=0,
            payload_pos=payload_pos,
            payload_vel=jnp.zeros(2),
            agent_goal_zone=agent_goal_zone,
            adversary_goal_zone=adversary_goal_zone,
            obs_pos=obs_pos,
            button_pos=button_pos,
            button_toggled=button_toggled,
            agent_names=self.good_agents,
            adversary_names=self.adversaries,
            prev_p_pos=p_pos,
            prev_payload_pos=payload_pos,
            option_assignment=option_assignment,
            cum_subtask_obs=cum_subtask_obs,
        )

        return self.obs_fn(state), state

    @partial(jax.jit, static_argnums=[0])
    def _compute_push_forces(self, state: State) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Compute forces on the payload from all agents.

        Returns:
            net_force: (2,) net force on payload
            agent_pushing: (num_good_agents,) boolean mask of agents pushing
            adv_pushing: (num_adversaries,) boolean mask of adversaries pushing
        """
        # Distance from each agent to payload center
        agent_dists = jnp.linalg.norm(
            state.p_pos[:self.num_good_agents] - state.payload_pos, axis=1
        )
        adv_dists = jnp.linalg.norm(
            state.p_pos[self.num_good_agents:] - state.payload_pos, axis=1
        )

        # Which agents are close enough to push (within push_radius)
        agent_pushing = agent_dists <= self.push_radius
        adv_pushing = adv_dists <= self.push_radius

        # Direction from each agent to the payload (agents push in this direction)
        agent_dirs = state.payload_pos - state.p_pos[:self.num_good_agents]
        agent_dirs = agent_dirs / (jnp.linalg.norm(agent_dirs, axis=1, keepdims=True) + 1e-8)

        adv_dirs = state.payload_pos - state.p_pos[self.num_good_agents:]
        adv_dirs = adv_dirs / (jnp.linalg.norm(adv_dirs, axis=1, keepdims=True) + 1e-8)

        # Weight force contribution by velocity component towards payload
        # Agents moving toward payload contribute more
        agent_vel_proj = jnp.sum(state.p_vel[:self.num_good_agents] * agent_dirs, axis=1)
        adv_vel_proj = jnp.sum(state.p_vel[self.num_good_agents:] * adv_dirs, axis=1)

        # Only count positive velocity (moving toward payload)
        agent_vel_proj = jnp.maximum(agent_vel_proj, 0)
        adv_vel_proj = jnp.maximum(adv_vel_proj, 0)

        # Force magnitude based on velocity and proximity
        agent_force_mag = agent_pushing * (1.0 + agent_vel_proj)
        adv_force_mag = adv_pushing * (1.0 + adv_vel_proj)

        # Net force: agents push toward adversary goal, adversaries push toward agent goal
        # Direction to opponent's goal zone
        agent_push_dir = state.adversary_goal_zone - state.payload_pos
        agent_push_dir = agent_push_dir / (jnp.linalg.norm(agent_push_dir) + 1e-8)

        adv_push_dir = state.agent_goal_zone - state.payload_pos
        adv_push_dir = adv_push_dir / (jnp.linalg.norm(adv_push_dir) + 1e-8)

        # Sum up forces from each team
        agent_total_force = jnp.sum(agent_force_mag) * agent_push_dir
        adv_total_force = jnp.sum(adv_force_mag) * adv_push_dir

        net_force = agent_total_force + adv_total_force

        return net_force, agent_pushing, adv_pushing

    @partial(jax.jit, static_argnums=[0])
    def _apply_proximity_forces(
        self, 
        payload_pos: chex.Array, 
        payload_vel: chex.Array, 
        net_force: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Apply proximity-based push forces to the payload.
        
        This is in addition to the collision physics from the base class.
        Agents within push_radius contribute forces even without direct collision.
        
        Args:
            payload_pos: Current payload position (2,)
            payload_vel: Current payload velocity (2,)
            net_force: Net force from all pushing agents (2,)
            
        Returns:
            new_payload_pos: Updated position (2,)
            new_payload_vel: Updated velocity (2,)
        """
        # Apply force: F = ma, so a = F/m
        acceleration = net_force / self.payload_mass
        
        # Update velocity
        new_vel = payload_vel + acceleration * self.dt
        
        # Apply damping to prevent excessive speeds
        damping = 0.95
        new_vel = new_vel * damping
        
        # Clamp maximum speed
        max_payload_speed = 2.0
        speed = jnp.linalg.norm(new_vel)
        new_vel = jnp.where(
            speed > max_payload_speed,
            new_vel / speed * max_payload_speed,
            new_vel
        )
        
        # Update position
        new_pos = payload_pos + new_vel * self.dt
        
        return new_pos, new_vel

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        # Store previous positions
        prev_p_pos = state.p_pos
        prev_payload_pos = state.payload_pos

        # Create simple state for physics
        # Order: agents, payload, obstacles, zones
        simple_state = SimpleState(
            p_pos=jnp.vstack(
                (state.p_pos, state.payload_pos[None, :], state.obs_pos, 
                 state.agent_goal_zone[None, :], state.adversary_goal_zone[None, :])
            ),
            p_vel=jnp.vstack(
                (state.p_vel, state.payload_vel[None, :], 
                 jnp.zeros((self.num_obstacles + 2, 2)))
            ),
            done=state.done,
            step=state.step,
            goal=None,
            c=jnp.zeros((self.num_actors, self.dim_c)),
        )

        # Step physics (includes collision between agents and payload via base class)
        simple_obs, simple_state, simple_reward, simple_dones, simple_info = (
            SimpleMPE.step_env(self, key, simple_state, actions)
        )

        # Extract updated positions
        # Order in simple_state: agents (0:num_actors), payload (num_actors), obstacles, zones
        new_p_pos = simple_state.p_pos[:self.num_actors, :]
        new_p_vel = simple_state.p_vel[:self.num_actors, :]
        new_payload_pos = simple_state.p_pos[self.num_actors, :]
        new_payload_vel = simple_state.p_vel[self.num_actors, :]

        # Update state with base class physics results
        state = state.replace(
            p_pos=new_p_pos,
            p_vel=new_p_vel,
            payload_pos=new_payload_pos,
            payload_vel=new_payload_vel,
            done=simple_state.done,
            step=simple_state.step,
        )

        # Compute push forces from agents within push_radius
        net_force, agent_pushing, adv_pushing = self._compute_push_forces(state)
        
        # Apply proximity-based forces (in addition to collision physics)
        new_payload_pos, new_payload_vel = self._apply_proximity_forces(
            state.payload_pos, state.payload_vel, net_force
        )
        
        # Detect button presses and update toggle states
        pressed = self._pressed_mask(state)
        new_button_toggled = self._update_button_toggles(state.button_toggled, pressed)
        
        # Block payload from entering locked zones
        blocked_payload_pos, payload_was_blocked = self._block_payload_entry(
            new_payload_pos,
            prev_payload_pos,
            new_button_toggled,
            state.agent_goal_zone,
            state.adversary_goal_zone,
        )
        
        # If blocked, also zero out velocity
        blocked_payload_vel = jnp.where(payload_was_blocked, jnp.zeros(2), new_payload_vel)

        # Update state with proximity force effects and button toggles
        state = state.replace(
            payload_pos=blocked_payload_pos,
            payload_vel=blocked_payload_vel,
            button_toggled=new_button_toggled,
        )

        # Check win conditions after step (payload must be fully inside zone)
        # Fully inside means: distance from center <= (zone_size - payload_radius)
        effective_zone_radius = self.zone_size - self.payload_radius
        payload_in_adv_zone_new = jnp.linalg.norm(
            state.payload_pos - state.adversary_goal_zone
        ) <= effective_zone_radius
        payload_in_agent_zone_new = jnp.linalg.norm(
            state.payload_pos - state.agent_goal_zone
        ) <= effective_zone_radius
        
        # Check if payload was in zones in previous step
        payload_in_adv_zone_prev = jnp.linalg.norm(
            prev_payload_pos - state.adversary_goal_zone
        ) <= effective_zone_radius
        payload_in_agent_zone_prev = jnp.linalg.norm(
            prev_payload_pos - state.agent_goal_zone
        ) <= effective_zone_radius
        
        # Detect entry: was outside, now inside
        entered_adv_zone = jnp.logical_and(~payload_in_adv_zone_prev, payload_in_adv_zone_new)
        entered_agent_zone = jnp.logical_and(~payload_in_agent_zone_prev, payload_in_agent_zone_new)
        entered_any_zone = jnp.logical_or(entered_adv_zone, entered_agent_zone)

        # Calculate rewards (only on entry)
        rewards = self.get_rewards_on_entry(entered_adv_zone, entered_agent_zone)
        
        # Reset payload to center position if reward was given
        starting_payload_pos = (state.agent_goal_zone + state.adversary_goal_zone) / 2
        reset_payload_pos = jnp.where(entered_any_zone, starting_payload_pos, state.payload_pos)
        reset_payload_vel = jnp.where(entered_any_zone, jnp.zeros(2), state.payload_vel)
        
        # Update state with reset payload if needed
        state = state.replace(
            payload_pos=reset_payload_pos,
            payload_vel=reset_payload_vel,
        )
        
        # Update zone status after reset (payload won't be in zone after reset)
        # If we entered and reset, payload is at center (not in zone)
        # Otherwise, use the current zone status
        payload_in_adv_zone_final = jnp.logical_and(
            payload_in_adv_zone_new, ~entered_any_zone
        )
        payload_in_agent_zone_final = jnp.logical_and(
            payload_in_agent_zone_new, ~entered_any_zone
        )

        # Compute subtask observations
        subtask_obs = self.subtask_obs_fn(
            agent_pushing,
            adv_pushing,
            state,
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
            "payload_in_adv_zone": payload_in_adv_zone_final,
            "payload_in_agent_zone": payload_in_agent_zone_final,
            "agent_pushing": agent_pushing,
            "adv_pushing": adv_pushing,
            "subtask_obs": subtask_obs,
            "cum_subtask_obs": cum_subtask_obs_dict,
            "option_assignment": state.option_assignment,
            "button_pos": state.button_pos,
            "button_pressed": pressed,
            "button_toggled": state.button_toggled,
            "payload_blocked": payload_was_blocked,
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
        state = state.replace(
            prev_p_pos=prev_p_pos,
            prev_payload_pos=prev_payload_pos,
        )

        obs = self.obs_fn(state)

        return obs, state, rewards, simple_dones, info

    @partial(jax.jit, static_argnums=[0])
    def subtask_obs_fn(
        self,
        agent_pushing: chex.Array,
        adv_pushing: chex.Array,
        state: State,
    ) -> Dict[str, chex.Array]:
        """
        Subtask observations for all agents.

        Only includes directly controllable subtasks (7 total):
        0. pushing_payload - agent is within push_radius but NOT colliding with payload
        1. near_own_goal - agent is near their own goal zone (defending)
        2. near_opp_goal - agent is near opponent's goal zone (attacking)
        3. contacting_payload - agent is colliding with the payload
        4. collide_teammate - agent collides with a teammate
        5. collide_opponent - agent collides with an opponent
        6. pressing_own_button - agent is pressing a button near their own goal zone
        """
        # Agent distances to zones
        agent_dist_own_zone = jnp.linalg.norm(
            state.p_pos[:self.num_good_agents] - state.agent_goal_zone, axis=1
        )
        agent_dist_opp_zone = jnp.linalg.norm(
            state.p_pos[:self.num_good_agents] - state.adversary_goal_zone, axis=1
        )
        adv_dist_own_zone = jnp.linalg.norm(
            state.p_pos[self.num_good_agents:] - state.adversary_goal_zone, axis=1
        )
        adv_dist_opp_zone = jnp.linalg.norm(
            state.p_pos[self.num_good_agents:] - state.agent_goal_zone, axis=1
        )

        # Distance to payload for each agent
        agent_dist_payload = jnp.linalg.norm(
            state.p_pos[:self.num_good_agents] - state.payload_pos, axis=1
        )
        adv_dist_payload = jnp.linalg.norm(
            state.p_pos[self.num_good_agents:] - state.payload_pos, axis=1
        )
        
        # Collision threshold (physical contact)
        collision_threshold = self.agent_size + self.payload_radius
        
        # Contacting = colliding with payload (within collision threshold)
        agent_colliding = agent_dist_payload <= collision_threshold
        adv_colliding = adv_dist_payload <= collision_threshold
        
        # Pushing = within push_radius but NOT colliding
        agent_in_push_range = agent_dist_payload <= self.push_radius
        adv_in_push_range = adv_dist_payload <= self.push_radius
        agent_pushing_only = agent_in_push_range & ~agent_colliding
        adv_pushing_only = adv_in_push_range & ~adv_colliding

        # Button press detection for subtasks
        btn = state.button_pos  # (4, 2)
        pos = state.p_pos  # (N, 2)
        vel = state.p_vel  # (N, 2)
        dists_to_btns = jnp.linalg.norm(btn[:, None, :] - pos[None, :, :], axis=-1)  # (4, N)
        near_btn = dists_to_btns <= self.button_radius
        still = jnp.linalg.norm(vel, axis=-1) <= self.vel_eps  # (N,)
        pressing_btn = jnp.logical_and(near_btn, still[None, :])  # (4, N)

        out = {}
        for i, name in enumerate(self.agents):
            is_agent = i < self.num_good_agents
            idx = i if is_agent else i - self.num_good_agents

            if is_agent:
                # pushing_payload: within push_radius but NOT colliding
                pushing = agent_pushing_only[idx]
                near_own = agent_dist_own_zone[idx] <= self.zone_size
                near_opp = agent_dist_opp_zone[idx] <= self.zone_size
                # contacting_payload: colliding with payload
                contacting = agent_colliding[idx]

                # Collisions with teammates
                own_dists = jnp.linalg.norm(
                    state.p_pos[idx] - jnp.delete(state.p_pos[:self.num_good_agents], idx, axis=0),
                    axis=1,
                )
                other_dists = jnp.linalg.norm(
                    state.p_pos[idx] - state.p_pos[self.num_good_agents:], axis=1
                )
                collide_teammate = jnp.any(own_dists <= 2 * self.agent_size)
                collide_opponent = jnp.any(other_dists <= 2 * self.agent_size)
                
                # Button subtasks for agents: buttons 0,1 are near agent zone
                pressing_own_btn = jnp.logical_or(pressing_btn[0, i], pressing_btn[1, i])
            else:
                # pushing_payload: within push_radius but NOT colliding
                pushing = adv_pushing_only[idx]
                near_own = adv_dist_own_zone[idx] <= self.zone_size
                near_opp = adv_dist_opp_zone[idx] <= self.zone_size
                # contacting_payload: colliding with payload
                contacting = adv_colliding[idx]

                # Collisions
                own_dists = jnp.linalg.norm(
                    state.p_pos[i] - jnp.delete(state.p_pos[self.num_good_agents:], idx, axis=0),
                    axis=1,
                )
                other_dists = jnp.linalg.norm(
                    state.p_pos[i] - state.p_pos[:self.num_good_agents], axis=1
                )
                collide_teammate = jnp.any(own_dists <= 2 * self.agent_size)
                collide_opponent = jnp.any(other_dists <= 2 * self.agent_size)
                
                # Button subtasks for adversaries: buttons 2,3 are near adversary zone
                pressing_own_btn = jnp.logical_or(pressing_btn[2, i], pressing_btn[3, i])

            obs_vec = jnp.array(
                [
                    pushing,
                    near_own,
                    near_opp,
                    contacting,
                    collide_teammate,
                    collide_opponent,
                    pressing_own_btn,
                ],
                dtype=jnp.float32,
            )
            out[name] = obs_vec

        return out

    @partial(jax.jit, static_argnums=[0])
    def obs_fn(self, state: State) -> Dict[str, chex.Array]:
        """
        Generate observations for all agents.

        Observations include:
        - Self velocity (2)
        - Teammate relative positions and velocities (rotated to local frame)
        - Opponent relative positions and velocities (rotated to local frame)
        - Payload relative position and velocity (rotated to local frame)
        - Own goal zone relative position (rotated to local frame)
        - Opponent goal zone relative position (rotated to local frame)
        - Button relative positions (4 buttons, rotated to local frame)
        - Button toggle states (2 zones)
        - Option observations (payload_progress, payload_speed)
        """
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
                # Teammates (other agents)
                teammate_pos = jnp.delete(state.p_pos[:self.num_good_agents], idx, axis=0)
                teammate_vel = jnp.delete(state.p_vel[:self.num_good_agents], idx, axis=0)
                # Opponents (adversaries)
                opponent_pos = state.p_pos[self.num_good_agents:]
                opponent_vel = state.p_vel[self.num_good_agents:]
                # Goals
                own_goal = state.agent_goal_zone
                opp_goal = state.adversary_goal_zone
            else:
                # Teammates (other adversaries)
                teammate_pos = jnp.delete(state.p_pos[self.num_good_agents:], idx, axis=0)
                teammate_vel = jnp.delete(state.p_vel[self.num_good_agents:], idx, axis=0)
                # Opponents (agents)
                opponent_pos = state.p_pos[:self.num_good_agents]
                opponent_vel = state.p_vel[:self.num_good_agents]
                # Goals
                own_goal = state.adversary_goal_zone
                opp_goal = state.agent_goal_zone

            # Transform to local frame
            teammate_rel_pos = ((teammate_pos - state.p_pos[i]) @ rotmat.T).flatten()
            teammate_rel_vel = ((teammate_vel - state.p_vel[i]) @ rotmat.T).flatten()
            opponent_rel_pos = ((opponent_pos - state.p_pos[i]) @ rotmat.T).flatten()
            opponent_rel_vel = ((opponent_vel - state.p_vel[i]) @ rotmat.T).flatten()

            payload_rel_pos = (state.payload_pos - state.p_pos[i]) @ rotmat.T
            payload_rel_vel = state.payload_vel @ rotmat.T

            own_goal_rel = (own_goal - state.p_pos[i]) @ rotmat.T
            opp_goal_rel = (opp_goal - state.p_pos[i]) @ rotmat.T

            # Button relative positions (4 buttons, transformed to local frame)
            button_rel_pos = ((state.button_pos - state.p_pos[i]) @ rotmat.T).flatten()
            
            # Button toggle states
            button_toggled = state.button_toggled.astype(jnp.float32)

            # Additional option observations
            # [payload_progress, payload_speed] - goal_rel already included in base obs
            payload_to_opp_goal = jnp.linalg.norm(state.payload_pos - opp_goal)
            total_dist = jnp.linalg.norm(own_goal - opp_goal)
            payload_progress = 1.0 - (payload_to_opp_goal / (total_dist + 1e-8))
            payload_speed = jnp.linalg.norm(state.payload_vel)

            option_obs = jnp.array([payload_progress, payload_speed])

            obs = jnp.concatenate([
                self_vel,
                teammate_rel_pos,
                teammate_rel_vel,
                opponent_rel_pos,
                opponent_rel_vel,
                payload_rel_pos,
                payload_rel_vel,
                own_goal_rel,
                opp_goal_rel,
                button_rel_pos,
                button_toggled,
                option_obs,
            ])

            out[name] = obs

        return out

    @partial(jax.jit, static_argnums=[0])
    def get_rewards_on_entry(
        self, entered_adv_zone: chex.Array, entered_agent_zone: chex.Array
    ) -> Dict[str, float]:
        """
        Compute rewards for all agents when payload enters a zone.
        
        Reward structure (zero-sum):
        - +1 for all agents when payload enters opponent's zone
        - -1 for all agents when payload enters own zone
        """
        r = jnp.zeros((self.num_actors,))

        # Team A (agents) wins when payload enters adversary zone
        r = r.at[:self.num_good_agents].add(entered_adv_zone.astype(float))
        # Team B (adversaries) wins when payload enters agent zone
        r = r.at[self.num_good_agents:].add(entered_agent_zone.astype(float))

        if self.zero_sum:
            # Subtract opponent's score
            r = r.at[:self.num_good_agents].subtract(entered_agent_zone.astype(float))
            r = r.at[self.num_good_agents:].subtract(entered_adv_zone.astype(float))

        return {a: r[i] for i, a in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=[0])
    def get_rewards(self, state: State) -> Dict[str, float]:
        """
        Compute rewards for all agents.

        Reward structure (zero-sum):
        - +1 for all agents when payload enters opponent's zone
        - -1 for all agents when payload enters own zone
        - Small reward for payload progress toward opponent's zone
        """
        r = jnp.zeros((self.num_actors,))

        # Check if payload in zones (payload must be fully inside zone)
        # Fully inside means: distance from center <= (zone_size - payload_radius)
        effective_zone_radius = self.zone_size - self.payload_radius
        payload_in_adv_zone = jnp.linalg.norm(
            state.payload_pos - state.adversary_goal_zone
        ) <= effective_zone_radius
        payload_in_agent_zone = jnp.linalg.norm(
            state.payload_pos - state.agent_goal_zone
        ) <= effective_zone_radius

        # Team A (agents) wins when payload in adversary zone
        r = r.at[:self.num_good_agents].add(payload_in_adv_zone.astype(float))
        # Team B (adversaries) wins when payload in agent zone
        r = r.at[self.num_good_agents:].add(payload_in_agent_zone.astype(float))

        if self.zero_sum:
            # Subtract opponent's score
            r = r.at[:self.num_good_agents].subtract(payload_in_agent_zone.astype(float))
            r = r.at[self.num_good_agents:].subtract(payload_in_adv_zone.astype(float))

        return {a: r[i] for i, a in enumerate(self.agents)}

