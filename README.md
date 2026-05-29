# skill_learning_mpe

Minimal JAX MAPPO baseline for five two-team MPE-style environments. The package exposes JAX-compatible environments, a feed-forward simultaneous self-play MAPPO trainer, and a static HTML evaluator with batched rollout selection and summary stats.

## Install

This project uses `uv` and installs JAX with CUDA 13 support.

```bash
uv sync --extra dev
```

Optional WandB logging is kept behind an extra:

```bash
uv sync --extra dev --extra wandb
```

Check that JAX can see the expected device(s):

```bash
uv run python -c "import jax; print(jax.devices())"
```

## Repository Layout

- `src/skill_learning_mpe/env/`: environment registry, spaces, wrappers, and MPE env implementations.
- `src/skill_learning_mpe/baselines/mappo.py`: small feed-forward actor/critic modules and batching helpers.
- `scripts/train_mappo.py`: train a shared-policy two-team MAPPO baseline.
- `scripts/eval_mappo.py`: run batched policy rollouts and write a static HTML viewer.
- `configs/mappo.yaml`: editable default training config.
- `tests/`: smoke tests for env stepping and HTML rendering.

## Environments

Use the short `--env-key` names for normal runs. Each key maps to a default full environment id.

| Key | Default env id | Description |
| --- | --- | --- |
| `ctf_buttons` | `simple_ctfbuttons_3v3_1obs_discrete_easy_0s_localobsv_random_v0` | Capture-the-flag with gated zone access. Agents must coordinate around buttons, flags, obstacles, and opposing-team pressure. |
| `assembly_line` | `simple_assemblyline_3v3_discrete_0s_random_v0` | Teams collect parts from rooms, deliver them to an assembler, and use a button/hold mechanic to complete assembly. |
| `king_of_hill` | `simple_kingofhill_3v3_discrete_0s_random_v0` | Teams contest hill ownership while using discrete movement/combat actions and respawn-style health mechanics. |
| `payload_escort` | `simple_payload_3v3_discrete_0s_random_v0` | Teams push a shared payload toward their scoring side while button state and nearby agents affect movement. |
| `playground` | `simple_playground_3v3_discrete_v0` | Combined task playground with payload, CTF, hill, combat, and assembly-style mechanics in one environment. |

You can override the default with `--env-id` when you want a supported variant. Supported id patterns are:

```text
simple_ctfbuttons_{A}v{B}_{N}obs_discrete_{easy|noobseasy|medium|hard}_{0s|gs}_{absobsv|relobsv|localobsv}_{random|static|everywhere}_v0
simple_payload_{A}v{B}_discrete_{0s|gs}_{random|static|everywhere}_v0
simple_kingofhill_{A}v{B}_discrete_{0s|gs}_{random|static|everywhere}_v0
simple_assemblyline_{A}v{B}_discrete_{0s|gs}_{random|static|everywhere}_v0
simple_playground_{A}v{B}_discrete_v0
```

`A` is the number of good agents and `B` is the number of adversaries. The current MAPPO scripts assume two teams and discrete actions.

## Environment API

Create environments from Python through the registry:

```python
import jax
from skill_learning_mpe.env import DEFAULT_ENV_IDS, make_env
from skill_learning_mpe.env.wrappers import apply_wrappers

env = make_env(DEFAULT_ENV_IDS["playground"])
env = apply_wrappers(env, ["TeamWorldStateWrapper", "AdversarialLogWrapper"])

key = jax.random.PRNGKey(0)
obs, state = env.reset(key)
actions = {agent: env.action_space(agent).sample(key) for agent in env.agents}
obs, state, reward, done, info = env.step(key, state, actions)
```

The trainer uses two wrappers by default:

- `TeamWorldStateWrapper`: adds `agent_world_state` and `adversary_world_state` observations for the centralized critic.
- `AdversarialLogWrapper`: logs separate good-team and adversary-team episode returns and lengths.

## Training

Run a quick smoke training job:

```bash
uv run python scripts/train_mappo.py   --env-key playground   --run-name smoke   --total-timesteps 1024   --num-envs 2   --num-steps 8
```

Run with the config file:

```bash
uv run python scripts/train_mappo.py   --config configs/mappo.yaml   --env-key ctf_buttons   --run-name ctf_buttons_mappo
```

Outputs are written to `runs/<run-name>/`:

- `config.yaml`: resolved config used for the run.
- `actor.safetensors`: final actor parameters.
- `critic.safetensors`: final critic parameters.
- `metrics.jsonl`: per-update loss and return metrics.

Enable WandB logging after installing the extra:

```bash
uv run python scripts/train_mappo.py --env-key playground --run-name playground_wandb --wandb
```

## Training CLI Knobs

| Flag | Meaning |
| --- | --- |
| `--env-key` | Short environment key. Defaults to `ctf_buttons`. |
| `--env-id` | Full environment id. Overrides `--env-key`. |
| `--run-name` | Output directory name under `runs/`. |
| `--config` | YAML file whose values override built-in defaults. |
| `--total-timesteps` | Total environment steps used to derive update count. |
| `--num-envs` | Number of parallel environments. |
| `--num-steps` | Rollout horizon per update. |
| `--seed` | JAX PRNG seed. |
| `--wandb` | Enable WandB logging. WandB is not imported unless this flag is set. |

## Training Config Knobs

These keys can be edited in `configs/mappo.yaml` or any YAML passed with `--config`.

| Key | Default | Meaning |
| --- | ---: | --- |
| `LR` | `1.0e-4` | Adam learning rate. |
| `NUM_ENVS` | `16` | Parallel rollout environments. More envs improve throughput but increase memory. |
| `NUM_STEPS` | `128` | Steps per rollout before each PPO update. |
| `TOTAL_TIMESTEPS` | `131072` | Total environment steps. Updates are `TOTAL_TIMESTEPS // NUM_STEPS // NUM_ENVS`. |
| `FC_DIM_SIZE` | `128` | Hidden width for actor and critic MLPs. |
| `UPDATE_EPOCHS` | `2` | PPO epochs over each collected batch. |
| `NUM_MINIBATCHES` | `4` | Number of PPO minibatches. `num_agents * NUM_ENVS` must be divisible by this. |
| `GAMMA` | `0.99` | Discount factor. |
| `GAE_LAMBDA` | `0.95` | GAE lambda. |
| `CLIP_EPS` | `0.2` | PPO policy and value clipping epsilon. |
| `WINSORIZE_ADV` | `false` | Clip normalized advantages before PPO loss. |
| `WINSORIZE_K` | `3.0` | Advantage clipping threshold when winsorization is enabled. |
| `SCALE_CLIP_EPS` | `false` | Divide `CLIP_EPS` by number of agents. |
| `ENT_COEF` | `0.01` | Entropy bonus coefficient. |
| `VF_COEF` | `0.5` | Value loss coefficient. |
| `MAX_GRAD_NORM` | `0.5` | Global gradient clipping norm. |
| `ACTIVATION` | `tanh` | MLP activation, `tanh` or `relu`. |
| `SEED` | `0` | Default random seed. |
| `ANNEAL_LR` | `false` | Linearly anneal learning rate over updates. |
| `ENV_WRAPPERS` | `TeamWorldStateWrapper`, `AdversarialLogWrapper` | Wrappers used by train/eval. |

## Evaluation and Web Output

Evaluate a trained run and write a static HTML page:

```bash
uv run python scripts/eval_mappo.py   --run-dir runs/smoke   --num-evals 8   --episode-index 0   --output outputs/eval/smoke/index.html
```

Render without a checkpoint using a random policy:

```bash
uv run python scripts/eval_mappo.py   --env-key playground   --random-policy   --num-evals 4   --output outputs/eval/random_playground/index.html
```

The generated page is self-contained HTML. It embeds rollout trajectory data and renders playback in a JavaScript canvas, so larger `--num-evals` values create larger files. The viewer includes:

- a rollout selector dropdown for batched rollouts;
- batch summary cards;
- a per-rollout stats table;
- fixed-size 720 by 720 canvas playback;
- equal-aspect world coordinates;
- labels for every actor and task entity;
- no GIF output.

## Evaluation CLI Knobs

| Flag | Meaning |
| --- | --- |
| `--run-dir` | Directory containing `config.yaml` and `actor.safetensors`. |
| `--actor-path` | Explicit actor checkpoint. Useful when not using `--run-dir`. |
| `--adv-actor-path` | Optional adversary actor checkpoint. Defaults to the same actor. |
| `--env-key` | Short environment key. Used unless overridden by `--env-id` or run config. |
| `--env-id` | Full environment id override. |
| `--num-evals` | Number of parallel rollout episodes to render and summarize. |
| `--episode-index` | Initial rollout selected in the HTML viewer. |
| `--output` | HTML output path. |
| `--random-policy` | Initialize random actor parameters instead of loading a checkpoint. |
| `--seed` | Evaluation seed. |

## Viewing HTML Outputs

Open an HTML file directly if you are on the same machine:

```bash
xdg-open outputs/eval/smoke/index.html
```

Or serve a directory locally:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory outputs/eval
```

Then open:

```text
http://127.0.0.1:8000/smoke/index.html
```

For a remote host on the same network or Tailscale tailnet, bind to all interfaces:

```bash
python3 -m http.server 8000 --bind 0.0.0.0 --directory outputs/eval
```

Find the remote address:

```bash
hostname -I
# or, on Tailscale:
tailscale ip -4
```

Then open this from your local browser:

```text
http://<remote-ip>:8000/<output-subdir>/index.html
```

For example:

```text
http://100.70.66.46:8000/playground/index.html
```

If you cannot expose a port directly, use SSH port forwarding instead:

```bash
ssh -L 8000:127.0.0.1:8000 user@remote-host
python3 -m http.server 8000 --bind 127.0.0.1 --directory outputs/eval
```

Then open `http://127.0.0.1:8000/<output-subdir>/index.html` locally.

## Tests

Run all smoke tests:

```bash
uv run pytest
```

The tests reset/step every default environment and verify that batched random-policy eval writes an HTML page with rollout selector and stats, without GIFs.
