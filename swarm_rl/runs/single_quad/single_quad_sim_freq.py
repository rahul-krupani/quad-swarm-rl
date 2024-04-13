from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.single_quad.baseline import QUAD_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000, 3333]),
    ('quads_obst_density', [0.05, 0.1]),
])

SINGLE_CLI = QUAD_BASELINE_CLI + (
    ' --quads_sim_freq=200 --rnn_size=16 --with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_group=tofs_corl_repo_dummy --wandb_user=multi-drones'
    '--quads_use_obstacles=True --quads_obst_spawn_area 8 8 --quads_obst_density=0.2 --quads_obst_size=0.6 '
    '--quads_obst_collision_reward=5.0 --quads_obstacle_obs_type=ToFs'
)

_experiment = Experiment(
    'tofs_corl_repo_dummy',
    SINGLE_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('tofs_corl_repo_dummy', experiments=[_experiment])
