from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.single_drone.quad_obstacle_single_baseline import QUAD_BASELINE_CLI_1

_params = ParamGrid(
    [
        ("seed", [0000, 3333]),
        ("quads_obstacle_tof_resolution", [8]),
        ("quads_obst_grid_size_random", [True]),
        ("normalize_input", [False, True]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_1 + (
    ' --quads_obst_grid_size=0.5 --quads_obst_spawn_center=False --quads_obst_grid_size_range 0.5 0.8 '
    '--quads_mode=o_static_same_goal --quads_obs_rel_rot=False --quads_dynamic_goal=True '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    '--wandb_group=darren_sd_mo_static_c_rew_v3'
)

_experiment = Experiment(
    "darren_sd_mo_static_c_rew_v3",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("single_drone_multi_obst", experiments=[_experiment])