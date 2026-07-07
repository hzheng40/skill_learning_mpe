#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import chex
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax.training.train_state import TrainState
from tqdm.auto import tqdm

from skill_learning_mpe.baselines.mappo import (
    ActorFF,
    CriticFF,
    Transition,
    batchify,
    unbatchify,
)
from skill_learning_mpe.env import DEFAULT_ENV_IDS, make_env, resolve_env_id
from skill_learning_mpe.env.wrappers import apply_wrappers, save_params


DEFAULT_CONFIG: Dict[str, Any] = {
    "LR": 1.0e-4,
    "NUM_ENVS": 16,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 131_072,
    "FC_DIM_SIZE": 128,
    "UPDATE_EPOCHS": 2,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "WINSORIZE_ADV": False,
    "WINSORIZE_K": 3.0,
    "SCALE_CLIP_EPS": False,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",
    "SEED": 0,
    "ANNEAL_LR": False,
    "ENV_WRAPPERS": ["TeamWorldStateWrapper", "AdversarialLogWrapper"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train feed-forward self-play MAPPO")
    parser.add_argument("--env-key", default="ctf_buttons", choices=sorted(DEFAULT_ENV_IDS))
    parser.add_argument("--env-id", default=None, help="Full env id; overrides --env-key")
    parser.add_argument("--run-name", default="mappo_run")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as f:
            config.update(yaml.safe_load(f) or {})
    config["ENV_NAME"] = resolve_env_id(args.env_key, args.env_id)
    config["ENV_KEY"] = args.env_key
    config["RUN_NAME"] = args.run_name
    if args.total_timesteps is not None:
        config["TOTAL_TIMESTEPS"] = args.total_timesteps
    if args.num_envs is not None:
        config["NUM_ENVS"] = args.num_envs
    if args.num_steps is not None:
        config["NUM_STEPS"] = args.num_steps
    if args.seed is not None:
        config["SEED"] = args.seed
    return config


def _json_ready(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return arr.tolist()


def make_train(config: Dict[str, Any], metrics_path: Path, use_wandb: bool = False):
    env = apply_wrappers(make_env(config["ENV_NAME"]), config["ENV_WRAPPERS"])
    scripted_attackers = bool(getattr(env, "scripted_attackers_enabled", False))
    train_agents = env.controlled_agents if scripted_attackers else env.agents
    config["SCRIPTED_ATTACKERS"] = scripted_attackers
    config["NUM_ACTORS"] = len(train_agents) * config["NUM_ENVS"]
    config["AG_NUM_ACTORS"] = env.num_good_agents * config["NUM_ENVS"]
    config["ADV_NUM_ACTORS"] = (
        0 if scripted_attackers else env.num_adversaries * config["NUM_ENVS"]
    )
    config["NUM_UPDATES"] = max(
        1, config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )

    if config["NUM_ACTORS"] % config["NUM_MINIBATCHES"] != 0:
        raise ValueError("NUM_ACTORS must be divisible by NUM_MINIBATCHES")

    pbar = tqdm(total=config["NUM_UPDATES"], desc=config["RUN_NAME"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_path.open("a", encoding="utf-8")

    if use_wandb:
        import wandb
    else:
        wandb = None

    def linear_schedule(count):
        frac = 1.0 - (
            count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
        ) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng: chex.PRNGKey):
        actor_network = ActorFF(env.action_space(train_agents[0]).n, config=config)
        critic_network = CriticFF(config=config)

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        ac_init_x = jnp.zeros(
            (1, config["NUM_ENVS"], env.observation_space(train_agents[0]).shape[0])
        )
        actor_params = actor_network.init(actor_rng, ac_init_x)
        cr_init_x = jnp.zeros((1, config["NUM_ENVS"], env.agent_world_state_size()))
        critic_params = critic_network.init(critic_rng, cr_init_x)

        tx_builder = lambda lr: optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(lr, eps=1e-5),
        )
        actor_tx = tx_builder(linear_schedule if config["ANNEAL_LR"] else config["LR"])
        critic_tx = tx_builder(linear_schedule if config["ANNEAL_LR"] else config["LR"])
        actor_state = TrainState.create(
            apply_fn=actor_network.apply, params=actor_params, tx=actor_tx
        )
        critic_state = TrainState.create(
            apply_fn=critic_network.apply, params=critic_params, tx=critic_tx
        )

        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
        obs, env_state = jax.vmap(env.reset)(reset_rngs)

        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, ag_last_done, adv_last_done, rng = runner_state
                actor_ts, critic_ts = train_states

                rng, ag_rng, adv_rng, step_rng = jax.random.split(rng, 4)
                ag_obs = batchify(last_obs, env.good_agents, config["AG_NUM_ACTORS"])

                ag_pi = actor_network.apply(actor_ts.params, ag_obs[np.newaxis, :])
                ag_action = ag_pi.sample(seed=ag_rng)
                ag_log_prob = ag_pi.log_prob(ag_action)
                ag_action_flat = ag_action.squeeze()
                ag_log_prob_flat = ag_log_prob.squeeze()
                ag_env_act = unbatchify(
                    ag_action, env.good_agents, config["NUM_ENVS"], env.num_good_agents
                )

                ag_world_state = last_obs["agent_world_state"].swapaxes(0, 1)
                ag_world_state = ag_world_state.reshape((config["AG_NUM_ACTORS"], -1))
                ag_value = critic_network.apply(critic_ts.params, ag_world_state[None, :])

                if scripted_attackers:
                    adv_action_flat = jnp.zeros((0,), dtype=ag_action_flat.dtype)
                    adv_log_prob_flat = jnp.zeros((0,), dtype=ag_log_prob_flat.dtype)
                    adv_value = jnp.zeros((0,), dtype=ag_value.dtype)
                    adv_obs = jnp.zeros((0, ag_obs.shape[-1]), dtype=ag_obs.dtype)
                    adv_world_state = jnp.zeros(
                        (0, ag_world_state.shape[-1]), dtype=ag_world_state.dtype
                    )
                    env_actions = ag_env_act
                else:
                    adv_obs = batchify(
                        last_obs, env.adversaries, config["ADV_NUM_ACTORS"]
                    )
                    adv_pi = actor_network.apply(actor_ts.params, adv_obs[np.newaxis, :])
                    adv_action = adv_pi.sample(seed=adv_rng)
                    adv_log_prob = adv_pi.log_prob(adv_action)
                    adv_action_flat = adv_action.squeeze()
                    adv_log_prob_flat = adv_log_prob.squeeze()
                    adv_env_act = unbatchify(
                        adv_action,
                        env.adversaries,
                        config["NUM_ENVS"],
                        env.num_adversaries,
                    )
                    adv_world_state = last_obs["adversary_world_state"].swapaxes(0, 1)
                    adv_world_state = adv_world_state.reshape(
                        (config["ADV_NUM_ACTORS"], -1)
                    )
                    adv_value = critic_network.apply(
                        critic_ts.params, adv_world_state[None, :]
                    )
                    env_actions = ag_env_act | adv_env_act

                step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                next_obs, next_env_state, reward, done, info = jax.vmap(env.step)(
                    step_rngs, env_state, env_actions
                )
                info = jax.tree.map(lambda x: x.flatten(), info)
                ag_done = batchify(done, env.good_agents, config["AG_NUM_ACTORS"]).squeeze()
                adv_done = (
                    jnp.zeros((0,), dtype=ag_done.dtype)
                    if scripted_attackers
                    else batchify(
                        done, env.adversaries, config["ADV_NUM_ACTORS"]
                    ).squeeze()
                )
                adv_reward = (
                    jnp.zeros((0,), dtype=jnp.asarray(reward[env.good_agents[0]]).dtype)
                    if scripted_attackers
                    else batchify(
                        reward, env.adversaries, config["ADV_NUM_ACTORS"]
                    ).squeeze()
                )

                transition = Transition(
                    jnp.tile(done["__all__"], len(train_agents)),
                    jnp.concatenate((ag_last_done, adv_last_done)),
                    jnp.concatenate((ag_action_flat, adv_action_flat)),
                    jnp.concatenate((ag_value.squeeze(), adv_value.squeeze())),
                    jnp.concatenate(
                        (
                            batchify(reward, env.good_agents, config["AG_NUM_ACTORS"]).squeeze(),
                            adv_reward,
                        )
                    ),
                    jnp.concatenate((ag_log_prob_flat, adv_log_prob_flat)),
                    jnp.vstack((ag_obs, adv_obs)),
                    jnp.vstack((ag_world_state, adv_world_state)),
                    info,
                )
                runner_state = (
                    train_states,
                    next_env_state,
                    next_obs,
                    ag_done,
                    adv_done,
                    rng,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            train_states, env_state, last_obs, ag_last_done, adv_last_done, rng = runner_state
            actor_ts, critic_ts = train_states

            ag_last_world_state = last_obs["agent_world_state"].swapaxes(0, 1)
            ag_last_world_state = ag_last_world_state.reshape((config["AG_NUM_ACTORS"], -1))
            ag_last_val = critic_network.apply(
                critic_ts.params, ag_last_world_state[None, :]
            ).squeeze()
            if scripted_attackers:
                last_val = ag_last_val
            else:
                adv_last_world_state = last_obs["adversary_world_state"].swapaxes(0, 1)
                adv_last_world_state = adv_last_world_state.reshape(
                    (config["ADV_NUM_ACTORS"], -1)
                )
                last_val = jnp.concatenate(
                    (
                        ag_last_val,
                        critic_network.apply(
                            critic_ts.params, adv_last_world_state[None, :]
                        ).squeeze(),
                    )
                )

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    delta = (
                        transition.reward
                        + config["GAMMA"] * next_value * (1 - transition.global_done)
                        - transition.value
                    )
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - transition.global_done) * gae
                    return (gae, transition.value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                train_states, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                batch = (traj_batch, advantages.squeeze(), targets.squeeze())
                permutation = jax.random.permutation(perm_rng, config["NUM_ACTORS"])
                shuffled = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled,
                )

                def _update_minibatch(train_states, batch_info):
                    actor_ts, critic_ts = train_states
                    traj, adv, target = batch_info

                    def actor_loss_fn(params, obs, action, old_log_prob, gae):
                        pi = actor_network.apply(params, obs)
                        log_prob = pi.log_prob(action)
                        logratio = log_prob - old_log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        if config["WINSORIZE_ADV"]:
                            gae = jnp.clip(gae, -config["WINSORIZE_K"], config["WINSORIZE_K"])
                        loss_actor = -jnp.minimum(
                            ratio * gae,
                            jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae,
                        ).mean()
                        entropy = pi.entropy().mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        return loss_actor - config["ENT_COEF"] * entropy, (
                            loss_actor,
                            entropy,
                            approx_kl,
                            clip_frac,
                        )

                    def critic_loss_fn(params, world_state, old_value, target):
                        value = critic_network.apply(params, world_state)
                        value_clipped = old_value + (value - old_value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_loss = jnp.maximum(
                            jnp.square(value - target),
                            jnp.square(value_clipped - target),
                        ).mean() * 0.5
                        return config["VF_COEF"] * value_loss, value_loss

                    (actor_loss, actor_aux), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
                        actor_ts.params, traj.obs, traj.action, traj.log_prob, adv
                    )
                    (critic_loss, value_loss), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
                        critic_ts.params, traj.world_state, traj.value, target
                    )
                    actor_ts = actor_ts.apply_gradients(grads=actor_grads)
                    critic_ts = critic_ts.apply_gradients(grads=critic_grads)
                    loss_info = {
                        "total_loss": actor_loss + critic_loss,
                        "actor_loss": actor_aux[0],
                        "value_loss": value_loss,
                        "entropy": actor_aux[1],
                        "approx_kl": actor_aux[2],
                        "clip_frac": actor_aux[3],
                    }
                    return (actor_ts, critic_ts), loss_info

                train_states, loss_info = jax.lax.scan(
                    _update_minibatch, train_states, minibatches
                )
                return (train_states, traj_batch, advantages, targets, rng), loss_info

            update_state = (train_states, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_states = update_state[0]
            rng = update_state[-1]
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric = traj_batch.info
            metric["loss"] = loss_info
            metric["update_steps"] = update_steps

            def callback(metric):
                env_step = int(metric["update_steps"]) * config["NUM_ENVS"] * config["NUM_STEPS"]
                row = {
                    "update": int(metric["update_steps"]),
                    "env_step": env_step,
                    "agent_returns": float(np.asarray(metric["ag_returned_episode_returns"][-1]).mean()),
                    "adversary_returns": float(np.asarray(metric["adv_returned_episode_returns"][-1]).mean()),
                }
                row.update({k: _json_ready(v) for k, v in metric["loss"].items()})
                metrics_file.write(json.dumps(row) + "\n")
                metrics_file.flush()
                if wandb is not None:
                    wandb.log(row)
                pbar.update(1)

            jax.experimental.io_callback(callback, None, metric)
            runner_state = (
                train_states,
                env_state,
                last_obs,
                ag_last_done,
                adv_last_done,
                rng,
            )
            return (runner_state, update_steps + 1), metric

        rng, loop_rng = jax.random.split(rng)
        runner_state = (
            (actor_state, critic_state),
            env_state,
            obs,
            jnp.zeros((config["AG_NUM_ACTORS"],), dtype=bool),
            jnp.zeros((config["ADV_NUM_ACTORS"],), dtype=bool),
            loop_rng,
        )
        runner_state, _ = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        final_runner_state, _ = runner_state
        return {"runner_state": final_runner_state}

    return train, metrics_file, pbar


def main() -> None:
    args = parse_args()
    config = load_config(args)
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=True)

    if args.wandb:
        import wandb

        wandb.init(
            project="skill_learning_mpe",
            config=config,
            tags=["MAPPO", "FF", config["ENV_KEY"], "SIMULTANEOUS_SELF_PLAY"],
        )

    rng = jax.random.PRNGKey(config["SEED"])
    train, metrics_file, pbar = make_train(config, run_dir / "metrics.jsonl", args.wandb)
    try:
        out = jax.jit(train)(rng)
    finally:
        metrics_file.close()
        pbar.close()

    final_train_states = out["runner_state"][0]
    actor_state, critic_state = final_train_states
    save_params(actor_state.params, run_dir / "actor.safetensors")
    save_params(critic_state.params, run_dir / "critic.safetensors")
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
