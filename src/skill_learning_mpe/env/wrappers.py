import os
from functools import partial
from typing import Callable, Dict, Iterable, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from flax import struct
from flax.traverse_util import flatten_dict, unflatten_dict
from safetensors.flax import load_file, save_file

from .multi_agent_env import MultiAgentEnv, State


def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
    flattened_dict = flatten_dict(params, sep=",")
    save_file(flattened_dict, filename)


def load_params(filename: Union[str, os.PathLike]) -> Dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")


class Wrapper:
    def __init__(self, env: MultiAgentEnv):
        self._env = env

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def _batchify_agents(self, x: dict, agents: Iterable[str]):
        return jnp.stack([x[a] for a in agents])


@struct.dataclass
class AdversarialLogEnvState:
    env_state: State
    ag_episode_returns: chex.Array
    ag_episode_lengths: chex.Array
    ag_returned_episode_returns: chex.Array
    ag_returned_episode_lengths: chex.Array
    adv_episode_returns: chex.Array
    adv_episode_lengths: chex.Array
    adv_returned_episode_returns: chex.Array
    adv_returned_episode_lengths: chex.Array


class AdversarialLogWrapper(Wrapper):
    """Track separate episode returns for the two teams."""

    def __init__(self, env: MultiAgentEnv, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, AdversarialLogEnvState]:
        obs, env_state = self._env.reset(key)
        state = AdversarialLogEnvState(
            env_state=env_state,
            ag_episode_returns=jnp.zeros((self._env.num_good_agents,)),
            ag_episode_lengths=jnp.zeros((self._env.num_good_agents,)),
            ag_returned_episode_returns=jnp.zeros((self._env.num_good_agents,)),
            ag_returned_episode_lengths=jnp.zeros((self._env.num_good_agents,)),
            adv_episode_returns=jnp.zeros((self._env.num_adversaries,)),
            adv_episode_lengths=jnp.zeros((self._env.num_adversaries,)),
            adv_returned_episode_returns=jnp.zeros((self._env.num_adversaries,)),
            adv_returned_episode_lengths=jnp.zeros((self._env.num_adversaries,)),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: AdversarialLogEnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, AdversarialLogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        ep_done = done["__all__"]
        ag_reward = self._batchify_agents(reward, self._env.good_agents)
        adv_reward = self._batchify_agents(reward, self._env.adversaries)
        ag_episode_returns = state.ag_episode_returns + ag_reward
        adv_episode_returns = state.adv_episode_returns + adv_reward
        ag_episode_lengths = state.ag_episode_lengths + 1
        adv_episode_lengths = state.adv_episode_lengths + 1

        state = AdversarialLogEnvState(
            env_state=env_state,
            ag_episode_returns=ag_episode_returns * (1 - ep_done),
            ag_episode_lengths=ag_episode_lengths * (1 - ep_done),
            ag_returned_episode_returns=state.ag_returned_episode_returns
            * (1 - ep_done)
            + ag_episode_returns * ep_done,
            ag_returned_episode_lengths=state.ag_returned_episode_lengths
            * (1 - ep_done)
            + ag_episode_lengths * ep_done,
            adv_episode_returns=adv_episode_returns * (1 - ep_done),
            adv_episode_lengths=adv_episode_lengths * (1 - ep_done),
            adv_returned_episode_returns=state.adv_returned_episode_returns
            * (1 - ep_done)
            + adv_episode_returns * ep_done,
            adv_returned_episode_lengths=state.adv_returned_episode_lengths
            * (1 - ep_done)
            + adv_episode_lengths * ep_done,
        )

        if self.replace_info:
            info = {}
        info["ag_returned_episode_returns"] = state.ag_returned_episode_returns
        info["ag_returned_episode_lengths"] = state.ag_returned_episode_lengths
        info["ag_returned_episode"] = jnp.full((self._env.num_good_agents,), ep_done)
        info["adv_returned_episode_returns"] = state.adv_returned_episode_returns
        info["adv_returned_episode_lengths"] = state.adv_returned_episode_lengths
        info["adv_returned_episode"] = jnp.full((self._env.num_adversaries,), ep_done)
        return obs, state, reward, done, info


class TeamWorldStateWrapper(Wrapper):
    """Add centralized critic observations for each team."""

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        obs, env_state = self._env.reset(key)
        return self._add_world_state(obs), env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, state: State, actions: dict):
        obs, env_state, reward, done, info = self._env.step(key, state, actions)
        return self._add_world_state(obs), env_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _add_world_state(self, obs: Dict[str, chex.Array]):
        obs["agent_world_state"] = self.agent_world_state(obs)
        obs["adversary_world_state"] = self.adversary_world_state(obs)
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def agent_world_state(self, obs: Dict[str, chex.Array]):
        all_obs = jnp.array([obs[a] for a in self._env.good_agents]).flatten()
        return jnp.expand_dims(all_obs, axis=0).repeat(
            self._env.num_good_agents, axis=0
        )

    @partial(jax.jit, static_argnums=(0,))
    def adversary_world_state(self, obs: Dict[str, chex.Array]):
        all_obs = jnp.array([obs[a] for a in self._env.adversaries]).flatten()
        return jnp.expand_dims(all_obs, axis=0).repeat(
            self._env.num_adversaries, axis=0
        )

    def agent_world_state_size(self):
        return sum(
            self._env.observation_space(agent).shape[-1]
            for agent in self._env.good_agents
        )

    def adversary_world_state_size(self):
        return sum(
            self._env.observation_space(agent).shape[-1]
            for agent in self._env.adversaries
        )


WRAPPER_REGISTRY: Dict[str, Callable[[MultiAgentEnv], Wrapper]] = {
    "TeamWorldStateWrapper": TeamWorldStateWrapper,
    "AdversarialLogWrapper": AdversarialLogWrapper,
}


def apply_wrappers(env: MultiAgentEnv, wrapper_names: Iterable[str]) -> MultiAgentEnv:
    for name in wrapper_names:
        if name not in WRAPPER_REGISTRY:
            names = ", ".join(sorted(WRAPPER_REGISTRY))
            raise ValueError(f"Unknown wrapper '{name}'. Available wrappers: {names}")
        env = WRAPPER_REGISTRY[name](env)
    return env
