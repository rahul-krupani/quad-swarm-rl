python -m sample_factory.launcher.run \
--run=swarm_rl.runs.hybrid.quads_hybrid_search_acc_agg_v1 \
--backend=slurm --slurm_workdir=slurm_output \
--experiment_suffix=slurm --pause_between=1 \
--slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 \
--slurm_sbatch_template=/home/rkrupani/slurm/swarm_rl_sbatch_timeout.sh \
--slurm_print_only=False