from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import numpy as np


TEAM_COLORS = {"agent": "#2563eb", "adversary": "#dc2626"}
PLOT_SIZE_PX = 720

POSITION_FIELDS = [
    "agent_zone",
    "adversary_zone",
    "agent_goal_zone",
    "adversary_goal_zone",
    "agent_spawn_center",
    "adversary_spawn_center",
    "obs_pos",
    "payload_pos",
    "payload_button_pos",
    "ctf_button_pos",
    "button_pos",
    "flag_p_pos",
    "hill_pos",
    "part_a_room_pos",
    "part_b_room_pos",
    "assembler_pos",
    "assembly_button_pos",
]
STATE_FIELDS = POSITION_FIELDS + ["p_pos", "payload_button_toggled", "button_toggled"]


def _array(value: Any) -> np.ndarray:
    return np.asarray(value)


def _has(state: Any, name: str) -> bool:
    return hasattr(state, name) and getattr(state, name) is not None


def _select_episode(value: Any, episode_index: int) -> np.ndarray:
    arr = _array(value)
    if arr.ndim >= 2:
        return arr[:, episode_index]
    return arr


def _position_fields(state: Any, episode_index: int) -> list[np.ndarray]:
    fields = []
    for name in POSITION_FIELDS + ["p_pos"]:
        if not _has(state, name):
            continue
        try:
            arr = _select_episode(getattr(state, name), episode_index)
        except Exception:
            continue
        if arr.ndim >= 2 and arr.shape[-1] == 2:
            fields.append(arr.reshape(-1, 2))
    return fields


def _axis_limits(state: Any, episode_index: int) -> tuple[float, float, float, float]:
    fields = _position_fields(state, episode_index)
    if not fields:
        return -30.0, 30.0, -30.0, 30.0
    points = np.concatenate(fields, axis=0)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return -30.0, 30.0, -30.0, 30.0
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = (lo + hi) / 2.0
    span = max(float(np.max(hi - lo)), 1.0)
    half_span = span / 2.0 + max(span * 0.15, 4.0)
    return (
        center[0] - half_span,
        center[0] + half_span,
        center[1] - half_span,
        center[1] + half_span,
    )


def _jsonable_array(value: Any) -> Any:
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.round(arr, 4)
    return arr.tolist()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable_array(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _rollout_payload(traj_state: Any, env: Any, episode_index: int) -> dict[str, Any]:
    fields = {}
    for name in STATE_FIELDS:
        if _has(traj_state, name):
            fields[name] = _jsonable_array(_select_episode(getattr(traj_state, name), episode_index))
    x0, x1, y0, y1 = _axis_limits(traj_state, episode_index)
    return {
        "index": episode_index,
        "axis": [float(x0), float(x1), float(y0), float(y1)],
        "fields": fields,
    }


def _format_stat(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _batch_summary(episode_stats: list[dict[str, Any]]) -> dict[str, Any]:
    if not episode_stats:
        return {}
    agent_returns = np.array([s["agent_return"] for s in episode_stats], dtype=float)
    adv_returns = np.array([s["adversary_return"] for s in episode_stats], dtype=float)
    lengths = np.array([s["episode_length"] for s in episode_stats], dtype=float)
    winners = [s["winner"] for s in episode_stats]
    count = len(episode_stats)
    return {
        "rollouts": count,
        "agent_mean_return": float(agent_returns.mean()),
        "adversary_mean_return": float(adv_returns.mean()),
        "agent_win_rate": winners.count("agent") / count,
        "adversary_win_rate": winners.count("adversary") / count,
        "tie_rate": winners.count("tie") / count,
        "mean_episode_length": float(lengths.mean()),
    }


def _stats_cards_html(summary: dict[str, Any]) -> str:
    if not summary:
        return ""
    cards = [
        ("Rollouts", summary["rollouts"], 0),
        ("Agent mean return", summary["agent_mean_return"], 3),
        ("Adversary mean return", summary["adversary_mean_return"], 3),
        ("Agent win rate", summary["agent_win_rate"], 3),
        ("Adversary win rate", summary["adversary_win_rate"], 3),
        ("Tie rate", summary["tie_rate"], 3),
        ("Mean length", summary["mean_episode_length"], 1),
    ]
    return "".join(
        f'<div class="stat-card"><span>{escape(label)}</span><strong>{escape(_format_stat(value, digits))}</strong></div>'
        for label, value, digits in cards
    )


def _stats_table_html(episode_stats: list[dict[str, Any]]) -> str:
    if not episode_stats:
        return ""
    rows = []
    for stat in episode_stats:
        rollout = int(stat["rollout"])
        rows.append(
            '<tr data-rollout-row="{rollout}">'
            "<td>{rollout}</td>"
            "<td>{winner}</td>"
            "<td>{agent_return}</td>"
            "<td>{adversary_return}</td>"
            "<td>{margin}</td>"
            "<td>{episode_length}</td>"
            "</tr>".format(
                rollout=rollout,
                winner=escape(str(stat["winner"])),
                agent_return=escape(_format_stat(stat["agent_return"])),
                adversary_return=escape(_format_stat(stat["adversary_return"])),
                margin=escape(_format_stat(stat["margin"])),
                episode_length=escape(_format_stat(stat["episode_length"], 0)),
            )
        )
    return (
        '<div class="stats-table-wrap"><table class="stats-table">'
        "<thead><tr>"
        "<th>Rollout</th><th>Winner</th><th>Agent return</th>"
        "<th>Adversary return</th><th>Margin</th><th>Length</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )


def _viewer_payload(
    traj_state: Any,
    env: Any,
    episode_stats: list[dict[str, Any]],
    selected_episode_index: int,
    fps: int,
) -> dict[str, Any]:
    p_pos = _array(traj_state.p_pos)
    num_rollouts = int(p_pos.shape[1]) if p_pos.ndim >= 4 else 1
    selected_episode_index = max(0, min(selected_episode_index, num_rollouts - 1))
    rollouts = [_rollout_payload(traj_state, env, idx) for idx in range(num_rollouts)]
    return {
        "plotSize": PLOT_SIZE_PX,
        "fps": fps,
        "selected": selected_episode_index,
        "env": {
            "numGood": int(env.num_good_agents),
            "numAdversaries": int(env.num_adversaries),
            "agentLabels": [f"agent_{idx}" for idx in range(int(env.num_good_agents))],
            "adversaryLabels": [f"adversary_{idx}" for idx in range(int(env.num_adversaries))],
            "agentSize": float(getattr(env, "agent_size", 1.0)),
            "zoneSize": float(getattr(env, "zone_size", 4.0)),
            "obstacleSize": float(getattr(env, "obstacle_size", 1.0)),
            "payloadRadius": float(getattr(env, "payload_radius", 1.5)),
            "hillRadius": float(getattr(env, "hill_radius", 2.0)),
            "roomRadius": float(getattr(env, "room_radius", 3.0)),
            "assemblerRadius": float(getattr(env, "assembler_radius", 4.0)),
            "buttonRadius": float(getattr(env, "button_radius", 1.5)),
        },
        "stats": episode_stats,
        "rollouts": rollouts,
    }


def render_trajectory_html(
    traj_state: Any,
    env: Any,
    episode_index: int = 0,
    title: str = "MAPPO evaluation",
    fps: int = 10,
) -> str:
    return render_batch_trajectory_html(
        traj_state,
        env,
        episode_stats=None,
        selected_episode_index=episode_index,
        title=title,
        fps=fps,
    )


def render_batch_trajectory_html(
    traj_state: Any,
    env: Any,
    episode_stats: list[dict[str, Any]] | None = None,
    selected_episode_index: int = 0,
    title: str = "MAPPO evaluation",
    fps: int = 10,
) -> str:
    p_pos = _array(traj_state.p_pos)
    num_rollouts = int(p_pos.shape[1]) if p_pos.ndim >= 4 else 1
    selected_episode_index = max(0, min(selected_episode_index, num_rollouts - 1))
    episode_stats = episode_stats or []
    payload = _viewer_payload(traj_state, env, episode_stats, selected_episode_index, fps)
    payload_json = (
        json.dumps(payload, separators=(",", ":"), default=_json_default)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    options = "".join(
        f'<option value="{idx}" {"selected" if idx == selected_episode_index else ""}>Rollout {idx}</option>'
        for idx in range(num_rollouts)
    )
    summary = _batch_summary(episode_stats)
    cards = _stats_cards_html(summary)
    table = _stats_table_html(episode_stats)
    return f"""
<section class="batch-panel">
  <div class="batch-controls">
    <label for="rollout-select">Rollout</label>
    <select id="rollout-select">{options}</select>
    <button id="play-toggle" type="button">Play</button>
    <label for="frame-slider">Frame</label>
    <input id="frame-slider" type="range" min="0" max="0" value="0">
    <span id="frame-current" class="frame-current"></span>
    <span id="rollout-current" class="rollout-current"></span>
  </div>
  <div class="stats-grid">{cards}</div>
  {table}
</section>
<section class="canvas-panel" aria-label="{escape(title)}">
  <canvas id="rollout-canvas" width="{PLOT_SIZE_PX}" height="{PLOT_SIZE_PX}"></canvas>
</section>
<script id="rollout-data" type="application/json">{payload_json}</script>
<script>
(() => {{
  const data = JSON.parse(document.getElementById("rollout-data").textContent);
  const canvas = document.getElementById("rollout-canvas");
  const ctx = canvas.getContext("2d");
  const select = document.getElementById("rollout-select");
  const playToggle = document.getElementById("play-toggle");
  const frameSlider = document.getElementById("frame-slider");
  const frameCurrent = document.getElementById("frame-current");
  const rolloutCurrent = document.getElementById("rollout-current");
  const rows = Array.from(document.querySelectorAll("[data-rollout-row]"));
  let rolloutIndex = Number(select.value || data.selected || 0);
  let frameIndex = 0;
  let playing = false;
  let timer = null;

  const colors = {{
    agent: "#2563eb",
    adversary: "#dc2626",
    obstacle: "#525252",
    payload: "#f59e0b",
    ctfButton: "#8b5cf6",
    spawn: "#7c3aed",
    payloadButtonOn: "#16a34a",
    payloadButtonOff: "#a3a3a3",
    flagAgent: "#1d4ed8",
    flagAdversary: "#b91c1c",
    hill: "#eab308",
    partA: "#0ea5e9",
    partB: "#f97316",
    assembler: "#22c55e",
    assemblyButton: "#14b8a6",
    grid: "#e5e7eb",
    text: "#111827"
  }};

  function rollout() {{ return data.rollouts[rolloutIndex]; }}
  function fields() {{ return rollout().fields; }}
  function frameCount() {{ return fields().p_pos.length; }}
  function fieldAt(name, step = frameIndex) {{
    const field = fields()[name];
    return field ? field[step] : null;
  }}
  function asPoints(value) {{
    if (!value) return [];
    if (Array.isArray(value[0]) && typeof value[0][0] === "number") return value;
    if (typeof value[0] === "number") return [value];
    return [];
  }}
  function asList(value) {{
    if (value === null || value === undefined) return [];
    return Array.isArray(value) ? value : [value];
  }}
  function toCanvas(xy) {{
    const [x0, x1, y0, y1] = rollout().axis;
    const x = ((xy[0] - x0) / (x1 - x0)) * canvas.width;
    const y = canvas.height - ((xy[1] - y0) / (y1 - y0)) * canvas.height;
    return [x, y];
  }}
  function worldRadius(radius) {{
    const [x0, x1] = rollout().axis;
    return radius * (canvas.width / (x1 - x0));
  }}
  function drawLabel(text, xy, dy = -12) {{
    const [x, y] = toCanvas(xy);
    ctx.save();
    ctx.font = "12px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const width = ctx.measureText(text).width + 8;
    const height = 16;
    const lx = x;
    const ly = y + dy;
    ctx.fillStyle = "rgba(255,255,255,0.88)";
    ctx.strokeStyle = "rgba(24,24,27,0.28)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(lx - width / 2, ly - height / 2, width, height, 4);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = colors.text;
    ctx.fillText(text, lx, ly + 0.5);
    ctx.restore();
  }}
  function drawCircle(xy, radius, fill, label, alpha = 1, stroke = fill, labelDy = -14) {{
    const [x, y] = toCanvas(xy);
    const r = Math.max(3, worldRadius(radius));
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = fill;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
    if (label) drawLabel(label, xy, labelDy);
  }}
  function drawFlag(xy, fill, label) {{
    const [x, y] = toCanvas(xy);
    ctx.save();
    ctx.strokeStyle = fill;
    ctx.fillStyle = fill;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, y - 24);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y - 24);
    ctx.lineTo(x + 24, y - 18);
    ctx.lineTo(x, y - 12);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
    drawLabel(label, xy, -36);
  }}
  function drawGrid() {{
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = colors.grid;
    ctx.lineWidth = 1;
    const lines = 10;
    for (let i = 0; i <= lines; i++) {{
      const p = (i / lines) * canvas.width;
      ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, canvas.height); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(canvas.width, p); ctx.stroke();
    }}
    ctx.strokeStyle = "#a1a1aa";
    ctx.strokeRect(0.5, 0.5, canvas.width - 1, canvas.height - 1);
  }}
  function drawFrame() {{
    const env = data.env;
    drawGrid();

    const agentZone = fieldAt("agent_zone");
    if (agentZone) drawCircle(agentZone, env.zoneSize, colors.agent, "agent_zone", 0.08, colors.agent, 18);
    const adversaryZone = fieldAt("adversary_zone");
    if (adversaryZone) drawCircle(adversaryZone, env.zoneSize, colors.adversary, "adversary_zone", 0.08, colors.adversary, 18);
    const agentGoalZone = fieldAt("agent_goal_zone");
    if (agentGoalZone) drawCircle(agentGoalZone, env.zoneSize, colors.agent, "agent_goal_zone", 0.08, colors.agent, 18);
    const adversaryGoalZone = fieldAt("adversary_goal_zone");
    if (adversaryGoalZone) drawCircle(adversaryGoalZone, env.zoneSize, colors.adversary, "adversary_goal_zone", 0.08, colors.adversary, 18);
    const agentSpawn = fieldAt("agent_spawn_center");
    if (agentSpawn) drawCircle(agentSpawn, env.agentSize * 1.4, colors.spawn, "agent_spawn_center", 0.18, colors.spawn, 18);
    const adversarySpawn = fieldAt("adversary_spawn_center");
    if (adversarySpawn) drawCircle(adversarySpawn, env.agentSize * 1.4, colors.spawn, "adversary_spawn_center", 0.18, colors.spawn, 18);

    asPoints(fieldAt("obs_pos")).forEach((xy, i) => drawCircle(xy, env.obstacleSize, colors.obstacle, `obstacle_${{i}}`, 0.25));

    const payload = fieldAt("payload_pos");
    const hasPayload = Boolean(payload);
    if (payload) drawCircle(payload, env.payloadRadius, colors.payload, "payload", 0.45);

    const payloadButtonSource = fieldAt("payload_button_pos") || (hasPayload ? fieldAt("button_pos") : null);
    const payloadButtons = asPoints(payloadButtonSource);
    const payloadToggleSource = fieldAt("payload_button_toggled") ?? fieldAt("button_toggled");
    const toggled = asList(payloadToggleSource);
    payloadButtons.forEach((xy, i) => {{
      const state = Boolean(toggled[Math.min(i, Math.max(toggled.length - 1, 0))] ?? false);
      drawCircle(xy, 0.7, state ? colors.payloadButtonOn : colors.payloadButtonOff, `payload_button_${{i}}`, 0.55);
    }});

    const ctfButtonSource = fieldAt("ctf_button_pos") || (!hasPayload ? fieldAt("button_pos") : null);
    const ctfButtons = asPoints(ctfButtonSource);
    ctfButtons.forEach((xy, i) => drawCircle(xy, 0.7, colors.ctfButton, `ctf_button_${{i}}`, 0.5));

    const flags = asPoints(fieldAt("flag_p_pos"));
    if (flags[0]) drawFlag(flags[0], colors.flagAgent, "agent_flag");
    if (flags[1]) drawFlag(flags[1], colors.flagAdversary, "adversary_flag");

    asPoints(fieldAt("hill_pos")).forEach((xy, i) => drawCircle(xy, env.hillRadius, colors.hill, `hill_${{i}}`, 0.32));

    const partA = fieldAt("part_a_room_pos");
    if (partA) drawCircle(partA, env.roomRadius, colors.partA, "part_a_room", 0.2, colors.partA, 18);
    const partB = fieldAt("part_b_room_pos");
    if (partB) drawCircle(partB, env.roomRadius, colors.partB, "part_b_room", 0.2, colors.partB, 18);
    const assembler = fieldAt("assembler_pos");
    if (assembler) drawCircle(assembler, env.assemblerRadius, colors.assembler, "assembler", 0.2, colors.assembler, 18);
    const assemblyButton = fieldAt("assembly_button_pos");
    if (assemblyButton) drawCircle(assemblyButton, env.buttonRadius, colors.assemblyButton, "assembly_button", 0.55);

    const pos = fieldAt("p_pos") || [];
    const adversaryStart = env.numGood;
    const adversaryEnd = env.numGood + env.numAdversaries;
    pos.slice(0, adversaryStart).forEach((xy, i) => drawCircle(xy, env.agentSize, colors.agent, env.agentLabels?.[i] || `agent_${{i}}`, 0.9));
    pos.slice(adversaryStart, adversaryEnd).forEach((xy, i) => drawCircle(xy, env.agentSize, colors.adversary, env.adversaryLabels?.[i] || `adversary_${{i}}`, 0.9));
    pos.slice(adversaryEnd).forEach((xy, i) => drawCircle(xy, env.agentSize, "#7c3aed", `entity_${{i}}`, 0.75));

    frameSlider.value = String(frameIndex);
    frameCurrent.textContent = `Frame ${{frameIndex + 1}} / ${{frameCount()}}`;
  }}
  function updateRolloutLabel() {{
    const stats = data.stats || [];
    const stat = stats.find((item) => Number(item.rollout) === rolloutIndex);
    if (!stat) {{
      rolloutCurrent.textContent = `Showing rollout ${{rolloutIndex}}`;
      return;
    }}
    const fmt = (value) => typeof value === "number" ? value.toFixed(3) : value;
    rolloutCurrent.textContent = `Showing rollout ${{rolloutIndex}} | ${{stat.winner}} | agent ${{fmt(stat.agent_return)}} vs adversary ${{fmt(stat.adversary_return)}}`;
  }}
  function syncControls() {{
    frameSlider.max = String(Math.max(0, frameCount() - 1));
    frameSlider.value = String(frameIndex);
    rows.forEach((row) => row.classList.toggle("selected", Number(row.dataset.rolloutRow) === rolloutIndex));
    updateRolloutLabel();
    drawFrame();
  }}
  function stop() {{
    playing = false;
    playToggle.textContent = "Play";
    if (timer) window.clearInterval(timer);
    timer = null;
  }}
  function play() {{
    playing = true;
    playToggle.textContent = "Pause";
    timer = window.setInterval(() => {{
      frameIndex = (frameIndex + 1) % frameCount();
      drawFrame();
    }}, 1000 / data.fps);
  }}

  select.addEventListener("change", (event) => {{
    rolloutIndex = Number(event.target.value);
    frameIndex = 0;
    stop();
    syncControls();
  }});
  frameSlider.addEventListener("input", (event) => {{
    frameIndex = Number(event.target.value);
    drawFrame();
  }});
  playToggle.addEventListener("click", () => {{ playing ? stop() : play(); }});

  syncControls();
}})();
</script>
"""


def write_eval_html(
    html: str,
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
    title: str = "MAPPO evaluation",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = metadata or {}
    meta_items = "".join(
        f"<dt>{escape(str(key))}</dt><dd>{escape(str(value))}</dd>"
        for key, value in metadata.items()
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #f7f7f8; color: #18181b; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 24px; margin: 0 0 16px; }}
    dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 6px 14px; margin: 0 0 20px; }}
    dt {{ font-weight: 700; color: #52525b; }}
    dd {{ margin: 0; }}
    .viewer {{ background: white; border: 1px solid #dedee3; border-radius: 8px; padding: 16px; overflow-x: auto; }}
    .batch-panel {{ display: grid; gap: 14px; margin-bottom: 16px; }}
    .batch-controls {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
    .batch-controls label {{ font-weight: 700; color: #3f3f46; }}
    button, select, input[type="range"] {{ accent-color: #2563eb; }}
    button, select {{ border: 1px solid #c7c7d1; border-radius: 6px; padding: 6px 10px; background: white; }}
    button {{ cursor: pointer; }}
    #frame-slider {{ width: 220px; }}
    .rollout-current, .frame-current {{ color: #52525b; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 8px; }}
    .stat-card {{ border: 1px solid #e4e4e7; border-radius: 8px; padding: 10px; background: #fafafa; }}
    .stat-card span {{ display: block; color: #71717a; font-size: 12px; }}
    .stat-card strong {{ display: block; font-size: 18px; margin-top: 4px; }}
    .stats-table-wrap {{ overflow-x: auto; }}
    .stats-table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    .stats-table th, .stats-table td {{ border-bottom: 1px solid #e4e4e7; padding: 8px; text-align: right; }}
    .stats-table th:first-child, .stats-table td:first-child, .stats-table th:nth-child(2), .stats-table td:nth-child(2) {{ text-align: left; }}
    .stats-table tr.selected {{ background: #eff6ff; }}
    .canvas-panel {{ width: {PLOT_SIZE_PX}px; min-width: {PLOT_SIZE_PX}px; }}
    #rollout-canvas {{ width: {PLOT_SIZE_PX}px; height: {PLOT_SIZE_PX}px; display: block; border: 1px solid #d4d4d8; background: white; }}
  </style>
</head>
<body>
  <main>
    <h1>{escape(title)}</h1>
    <dl>{meta_items}</dl>
    <section class="viewer">{html}</section>
  </main>
</body>
</html>
"""
    output_path.write_text(page, encoding="utf-8")
    return output_path
