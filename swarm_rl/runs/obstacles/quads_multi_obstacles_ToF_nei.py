from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.quad_obstacle_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 1111, 2222, 3333]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    ' --async_rl=False --serial_mode=True --batch_size=2048 --num_workers=4 --num_envs_per_worker=4 --rollout=128 '
    '--quads_obst_hidden_size=4 --rnn_size=16 --quads_neighbor_hidden_size=16 --quads_obs_repr=xyz_vxyz_R_omega '
    '--quads_obstacle_obs_type=ToFs --with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    '--wandb_group=ToFs-xxxx '
)

_experiment = Experiment(
    "ToFs-xxxx",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("ToFs-xxxx", experiments=[_experiment])