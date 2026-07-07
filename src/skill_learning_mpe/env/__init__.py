from .mpe.simple_area_denial import SimpleAreaDenial
from .mpe.simple_assembly_line import SimpleAssemblyLine
from .mpe.simple_ctf_button_gate import SimpleCTFButtons
from .mpe.simple_grid_ctf_button_gate import SimpleGridCTFButtons
from .mpe.simple_king_of_hill import SimpleKingOfHill
from .mpe.simple_payload_escort import SimplePayloadEscort
from .mpe.simple_playground import SimplePlayground
from .multi_agent_env import MultiAgentEnv

DEFAULT_ENV_IDS = {
    "ctf_buttons": "simple_ctfbuttons_3v3_1obs_discrete_easy_0s_localobsv_random_v0",
    "grid_ctf_buttons": "simple_gridctfbuttons_3v3_1obs_15x9_discrete_random_v0",
    "area_denial": "simple_areadenial_3v3_0obs_discrete_0s_random_v0",
    "assembly_line": "simple_assemblyline_3v3_discrete_0s_random_v0",
    "king_of_hill": "simple_kingofhill_3v3_discrete_0s_random_v0",
    "payload_escort": "simple_payload_3v3_discrete_0s_random_v0",
    "playground": "simple_playground_3v3_discrete_v0",
}

SUPPORTED_ENV_KEYS = tuple(DEFAULT_ENV_IDS.keys())


def resolve_env_id(env_key: str | None = None, env_id: str | None = None) -> str:
    if env_id is not None:
        return env_id
    key = env_key or "ctf_buttons"
    if key not in DEFAULT_ENV_IDS:
        valid = ", ".join(SUPPORTED_ENV_KEYS)
        raise ValueError(f"Unknown env key '{key}'. Available keys: {valid}")
    return DEFAULT_ENV_IDS[key]


def make_env(env_id: str) -> MultiAgentEnv:
    args = env_id.split("_")
    if len(args) < 2:
        raise ValueError(f"Invalid environment id: {env_id}")

    env_core = args[0] + args[1]

    if env_core == "simplectfbuttons":
        # simple_ctfbuttons_{num_agents}v{num_adversaries}_{n}obs_{acttype}_{zonesize}_{0s}_{obsv}_{randomstart}_v0
        num_ag, num_adv = args[2].split("v")
        num_obs = int(args[3].replace("obs", ""))
        act_type = args[4].capitalize()

        if args[5] == "easy":
            zone_size = 5.0
            zone_dist = 12.0
            obs_size = 1.5
        elif args[5] == "noobseasy":
            zone_size = 5.0
            zone_dist = 15.0
            obs_size = 0.001
        elif args[5] == "medium":
            zone_size = 5.0
            zone_dist = 15.0
            obs_size = 1.5
        elif args[5] == "hard":
            zone_size = 1.0
            zone_dist = 10.0
            obs_size = 1.5
        else:
            raise ValueError(f"Unknown CTF difficulty segment in env id: {env_id}")

        zero_sum = args[6] == "0s"
        obs_type = args[7]
        if obs_type == "absobsv":
            abs_obsv = True
            transform = False
        elif obs_type == "relobsv":
            abs_obsv = False
            transform = False
        elif obs_type == "localobsv":
            abs_obsv = False
            transform = True
        else:
            raise ValueError(f"Unknown observation segment in env id: {env_id}")

        random_start = args[8] != "static"
        init_agent_everywhere = random_start and args[8] == "everywhere"
        return SimpleCTFButtons(
            num_good_agents=int(num_ag),
            num_adversaries=int(num_adv),
            num_obstacles=num_obs,
            action_type=act_type,
            zone_size=zone_size,
            dist_between_zones=zone_dist,
            obstacle_size=obs_size,
            zero_sum=zero_sum,
            abs_obs=abs_obsv,
            random_start=random_start,
            transform=transform,
            init_agent_everywhere=init_agent_everywhere,
        )

    if env_core == "simplegridctfbuttons":
        # simple_gridctfbuttons_{num_agents}v{num_adversaries}_{n}obs_{width}x{height}_{acttype}_{randomstart}_v0
        num_ag, num_adv = args[2].split("v")
        num_obs = int(args[3].replace("obs", ""))
        width, height = (int(part) for part in args[4].split("x"))
        act_type = args[5].capitalize()
        random_start = args[6] != "static"
        return SimpleGridCTFButtons(
            num_good_agents=int(num_ag),
            num_adversaries=int(num_adv),
            num_obstacles=num_obs,
            width=width,
            height=height,
            action_type=act_type,
            random_start=random_start,
        )

    if env_core == "simpleareadenial":
        # simple_areadenial_{num_agents}v{num_adversaries}_{n}obs_{acttype}_{0s}_{randomstart}[_ctrl-{mode}]_v0
        num_ag, num_adv = args[2].split("v")
        num_obs = int(args[3].replace("obs", ""))
        act_type = args[4].capitalize()
        zero_sum = args[5] == "0s"
        random_start = args[6] != "static"
        init_agent_everywhere = random_start and args[6] == "everywhere"
        attacker_controller_mode = None
        if len(args) == 9:
            if not args[7].startswith("ctrl-"):
                raise ValueError(f"Unknown area-denial controller segment in env id: {env_id}")
            attacker_controller_mode = args[7].replace("ctrl-", "", 1)
        elif len(args) != 8:
            raise ValueError(f"Invalid area-denial environment id: {env_id}")
        return SimpleAreaDenial(
            num_good_agents=int(num_ag),
            num_adversaries=int(num_adv),
            num_obstacles=num_obs,
            action_type=act_type,
            area_radius=4.0,
            spawn_distance=20.0,
            spawn_cluster_radius=3.0,
            agent_size=1.0,
            zero_sum=zero_sum,
            random_start=random_start,
            init_agent_everywhere=init_agent_everywhere,
            attacker_controller_mode=attacker_controller_mode,
        )

    if env_core == "simplepayload":
        # simple_payload_{num_agents}v{num_adversaries}_{acttype}_{0s}_{randomstart}_v0
        num_ag, num_adv = args[2].split("v")
        zero_sum = args[4] == "0s"
        random_start = args[5] != "static"
        init_agent_everywhere = random_start and args[5] == "everywhere"
        return SimplePayloadEscort(
            num_good_agents=int(num_ag),
            num_adversaries=int(num_adv),
            num_obstacles=0,
            payload_mass=8.0,
            payload_radius=1.5,
            push_radius=6.0,
            zone_size=4.0,
            dist_between_zones=15.0,
            agent_size=1.0,
            zero_sum=zero_sum,
            random_start=random_start,
            init_agent_everywhere=init_agent_everywhere,
        )

    if env_core == "simplekingofhill":
        # simple_kingofhill_{num_agents}v{num_adversaries}_{acttype}_{0s}_{randomstart}_v0
        num_ag, num_adv = args[2].split("v")
        zero_sum = args[4] == "0s"
        random_start = args[5] != "static"
        init_agent_everywhere = random_start and args[5] == "everywhere"
        return SimpleKingOfHill(
            num_good_agents=int(num_ag),
            num_adversaries=int(num_adv),
            num_obstacles=0,
            hill_radius=2.0,
            melee_range=2.5,
            max_hp=3,
            spawn_distance=20.0,
            spawn_cluster_radius=3.0,
            agent_size=1.0,
            zero_sum=zero_sum,
            random_start=random_start,
            init_agent_everywhere=init_agent_everywhere,
        )

    if env_core == "simpleassemblyline":
        # simple_assemblyline_{num_agents}v{num_adversaries}_{acttype}_{0s}_{randomstart}_v0
        num_ag, num_adv = args[2].split("v")
        zero_sum = args[4] == "0s"
        random_start = args[5] != "static"
        init_agent_everywhere = random_start and args[5] == "everywhere"
        return SimpleAssemblyLine(
            num_good_agents=int(num_ag),
            num_adversaries=int(num_adv),
            room_radius=3.0,
            assembler_radius=4.0,
            button_radius=1.5,
            vel_eps=2.0,
            assembly_hold_time=30,
            spawn_distance=25.0,
            spawn_cluster_radius=3.0,
            agent_size=1.0,
            zero_sum=zero_sum,
            random_start=random_start,
            init_agent_everywhere=init_agent_everywhere,
        )

    if env_core == "simpleplayground":
        # simple_playground_{num_agents}v{num_adversaries}_{acttype}_v0
        num_ag, num_adv = args[2].split("v")
        act_type = args[3].capitalize()
        return SimplePlayground(
            num_good_agents=int(num_ag),
            num_adversaries=int(num_adv),
            action_type=act_type,
        )

    valid = ", ".join(DEFAULT_ENV_IDS.values())
    raise ValueError(f"Unknown environment '{env_id}'. Example env ids: {valid}")


__all__ = [
    "DEFAULT_ENV_IDS",
    "SUPPORTED_ENV_KEYS",
    "MultiAgentEnv",
    "make_env",
    "resolve_env_id",
]
