from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.quad_obstacle_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 1111, 2222, 3333]),
        ("quads_num_agents", [1]),
        ("quads_obs_rel_rot", [True]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    ' --anneal_collision_steps=0 --replay_buffer_sample_prob=0.0 --num_workers=36 '
    '--num_envs_per_worker=4 --rnn_size=16 --quads_obs_repr=xyz_vxyz_R_omega '
    '--quads_obst_hidden_size=4 --quads_neighbor_visible_num=-1 --quads_neighbor_obs_type=none '
    '--quads_obst_density=0.1 --quads_obstacle_obs_type=ToFs '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    '--wandb_group=refactor_test_rel_rot_v1'
)

_experiment = Experiment(
    "refactor_test_rel_rot_v1",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("single_quad_multi_obst", experiments=[_experiment])