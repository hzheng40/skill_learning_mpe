from pathlib import Path
import subprocess
import sys

from skill_learning_mpe.env import DEFAULT_ENV_IDS


def test_random_policy_eval_writes_html_without_gifs(tmp_path):
    env_key = "playground"
    out = tmp_path / "index.html"
    cmd = [
        sys.executable,
        "scripts/eval_mappo.py",
        "--env-key",
        env_key,
        "--random-policy",
        "--num-evals",
        "2",
        "--output",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "<html" in html
    assert "rollout-select" in html
    assert "rollout-canvas" in html
    assert "Agent mean return" in html
    assert "720px" in html
    assert "agent_0" in html
    assert "adversary_0" in html
    assert "payload" in html
    assert "agent_goal_zone" in html
    assert "adversary_goal_zone" in html
    assert "agent_spawn_center" in html
    assert "adversary_spawn_center" in html
    assert "assembly_button" in html
    assert "_anim_img" not in html
    assert not list(tmp_path.rglob("*.gif"))
