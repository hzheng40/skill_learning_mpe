import jax

from skill_learning_mpe.env import DEFAULT_ENV_IDS, make_env
from skill_learning_mpe.env.wrappers import apply_wrappers


def test_default_envs_reset_and_step():
    for env_id in DEFAULT_ENV_IDS.values():
        env = apply_wrappers(
            make_env(env_id), ["TeamWorldStateWrapper", "AdversarialLogWrapper"]
        )
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        assert "agent_world_state" in obs
        assert "adversary_world_state" in obs
        actions = {agent: env.action_space(agent).sample(key) for agent in env.agents}
        obs, state, reward, done, info = env.step(key, state, actions)
        assert "__all__" in done
        assert "ag_returned_episode_returns" in info
        assert "adv_returned_episode_returns" in info
