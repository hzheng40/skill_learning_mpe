
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from skill_learning_mpe.baselines.mappo import ActorFF, batchify, unbatchify
from skill_learning_mpe.env import DEFAULT_ENV_IDS, make_env, resolve_env_id
from skill_learning_mpe.env.mpe.visualizer import (
    render_batch_trajectory_html,
    write_eval_html,
)
from skill_learning_mpe.env.wrappers import apply_wrappers, load_params


DEFAULT_WRAPPERS = ["TeamWorldStateWrapper", "AdversarialLogWrapper"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MAPPO and write a static HTML viewer"
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--actor-path", type=Path, default=None)
    parser.add_argument("--adv-actor-path", type=Path, default=None)
    parser.add_argument("--env-key", default="ctf_buttons", choices=sorted(DEFAULT_ENV_IDS))
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--num-evals", type=int, default=4)
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Initial rollout selected in the generated batch viewer.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/eval/index.html"))
    parser.add_argument("--random-policy", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_eval_config(args: argparse.Namespace) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "SEED": 0,
        "FC_DIM_SIZE": 128,
        "ACTIVATION": "tanh",
        "ENV_WRAPPERS": DEFAULT_WRAPPERS,
    }
    if args.run_dir is not None and (args.run_dir / "config.yaml").exists():
        with (args.run_dir / "config.yaml").open("r", encoding="utf-8") as f:
            config.update(yaml.safe_load(f) or {})
    config["ENV_NAME"] = resolve_env_id(
        args.env_key, args.env_id or config.get("ENV_NAME")
    )
    config["ENV_KEY"] = args.env_key
    if args.seed is not None:
        config["SEED"] = args.seed
    return config


def _policy_actions(env, config, actor, params, obs, agents, num_evals, rng):
    obs_batch = batchify(obs, agents, num_evals * len(agents))
    pi = actor.apply(params, obs_batch[np.newaxis, :])
    action = pi.sample(seed=rng)
    return unbatchify(action, agents, num_evals, len(agents))


def rollout(config: Dict[str, Any], args: argparse.Namespace):
    rng = jax.random.PRNGKey(config["SEED"])
    base_env = make_env(config["ENV_NAME"])
    env = apply_wrappers(base_env, config.get("ENV_WRAPPERS", DEFAULT_WRAPPERS))
    scripted_attackers = bool(getattr(env, "scripted_attackers_enabled", False))
    policy_agents = env.controlled_agents if scripted_attackers else env.agents
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, args.num_evals)
    obs, env_state = jax.vmap(env.reset)(reset_rngs)

    actor = ActorFF(env.action_space(policy_agents[0]).n, config=config)
    if args.random_policy:
        rng, init_rng = jax.random.split(rng)
        init_x = jnp.zeros(
            (1, args.num_evals, env.observation_space(policy_agents[0]).shape[0])
        )
        actor_params = actor.init(init_rng, init_x)
        adv_actor_params = None if scripted_attackers else actor_params
    else:
        actor_path = args.actor_path or (
            args.run_dir / "actor.safetensors" if args.run_dir else None
        )
        if actor_path is None:
            raise ValueError("Provide --run-dir, --actor-path, or use --random-policy")
        adv_actor_path = args.adv_actor_path or actor_path
        actor_params = load_params(actor_path)
        adv_actor_params = None if scripted_attackers else load_params(adv_actor_path)

    def _env_step(runner_state, unused):
        env_state, last_obs, rng = runner_state
        rng, ag_rng, adv_rng, step_rng = jax.random.split(rng, 4)
        ag_act = _policy_actions(
            env,
            config,
            actor,
            actor_params,
            last_obs,
            env.good_agents,
            args.num_evals,
            ag_rng,
        )
        if scripted_attackers:
            env_actions = ag_act
        else:
            adv_act = _policy_actions(
                env,
                config,
                actor,
                adv_actor_params,
                last_obs,
                env.adversaries,
                args.num_evals,
                adv_rng,
            )
            env_actions = ag_act | adv_act
        step_rngs = jax.random.split(step_rng, args.num_evals)
        obs, env_state, reward, done, info = jax.vmap(env.step)(
            step_rngs, env_state, env_actions
        )
        transition = (env_state.env_state, reward, done, info)
        return (env_state, obs, rng), transition

    _, rollout_data = jax.lax.scan(
        _env_step, (env_state, obs, rng), None, base_env.max_steps
    )
    traj_state, reward_traj, done_traj, info_traj = rollout_data
    return base_env, traj_state, reward_traj, done_traj, info_traj


def compute_batch_stats(env, reward_traj, done_traj) -> list[dict[str, Any]]:
    agent_rewards = np.stack(
        [np.asarray(reward_traj[agent]) for agent in env.good_agents], axis=0
    )
    adversary_rewards = np.stack(
        [np.asarray(reward_traj[agent]) for agent in env.adversaries], axis=0
    )
    agent_returns = agent_rewards.sum(axis=(0, 1))
    adversary_returns = adversary_rewards.sum(axis=(0, 1))
    done_all = np.asarray(done_traj["__all__"], dtype=bool)
    if done_all.ndim == 1:
        done_all = done_all[:, None]
    episode_lengths = np.where(
        done_all.any(axis=0), done_all.argmax(axis=0) + 1, done_all.shape[0]
    )
    stats = []
    for idx, (agent_return, adversary_return, length) in enumerate(
        zip(agent_returns, adversary_returns, episode_lengths)
    ):
        margin = float(agent_return - adversary_return)
        winner = "tie"
        if margin > 0:
            winner = "agent"
        elif margin < 0:
            winner = "adversary"
        stats.append(
            {
                "rollout": int(idx),
                "agent_return": float(agent_return),
                "adversary_return": float(adversary_return),
                "margin": margin,
                "winner": winner,
                "episode_length": int(length),
            }
        )
    return stats


def main() -> None:
    args = parse_args()
    config = load_eval_config(args)
    env, traj_state, reward_traj, done_traj, _ = rollout(config, args)
    episode_stats = compute_batch_stats(env, reward_traj, done_traj)
    title = f"{config['ENV_KEY']} MAPPO evaluation"
    html = render_batch_trajectory_html(
        traj_state,
        env,
        episode_stats=episode_stats,
        selected_episode_index=args.episode_index,
        title=title,
    )
    out_path = write_eval_html(
        html,
        args.output,
        metadata={
            "env_key": config["ENV_KEY"],
            "env_id": config["ENV_NAME"],
            "num_evals": args.num_evals,
            "initial_episode_index": args.episode_index,
            "policy": "random" if args.random_policy else "checkpoint",
            "scripted_attackers": bool(getattr(env, "scripted_attackers_enabled", False)),
            "attacker_controller_mode": getattr(env, "attacker_controller_mode", None),
        },
        title=title,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
