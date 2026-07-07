from pathlib import Path
import json
import re
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTED_AREA_ID = "simple_areadenial_3v3_0obs_discrete_0s_static_ctrl-direct_v0"


def test_scripted_area_denial_train_smoke(tmp_path):
    run_dir = tmp_path / "scripted_area_train"
    cmd = [
        sys.executable,
        "scripts/train_mappo.py",
        "--env-key",
        "area_denial",
        "--env-id",
        SCRIPTED_AREA_ID,
        "--run-name",
        str(run_dir),
        "--total-timesteps",
        "16",
        "--num-envs",
        "4",
        "--num-steps",
        "4",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    assert (run_dir / "actor.safetensors").exists()
    assert (run_dir / "critic.safetensors").exists()
    assert (run_dir / "metrics.jsonl").exists()


def test_scripted_area_denial_random_eval_smoke(tmp_path):
    out = tmp_path / "scripted_area_eval.html"
    cmd = [
        sys.executable,
        "scripts/eval_mappo.py",
        "--env-key",
        "area_denial",
        "--env-id",
        SCRIPTED_AREA_ID,
        "--random-policy",
        "--num-evals",
        "1",
        "--output",
        str(out),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    html = out.read_text(encoding="utf-8")
    assert "area_denial MAPPO evaluation" in html
    assert "scripted_attackers" in html
    assert "area_pos" in html
    assert "protected_area" in html
    assert "adversary_0" in html
    match = re.search(
        r'<script id="rollout-data" type="application/json">(.*?)</script>',
        html,
        re.S,
    )
    assert match is not None
    payload = json.loads(match.group(1))
    axis = payload["rollouts"][0]["axis"]
    assert max(abs(value) for value in axis) < 100.0
