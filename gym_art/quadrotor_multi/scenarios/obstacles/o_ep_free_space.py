import numpy as np

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base


class Scenario_o_ep_free_space(Scenario_o_base):
    # Based on https://mathcurve.com/courbes3d.gb/lissajous3d/lissajous3d.shtml

    def step(self, infos, rewards):
        control_freq = self.envs[0].control_freq
        tick = self.envs[0].tick / control_freq
        x, y, z = self.generate_pos_obst_map()
        goal_x, goal_y, goal_z = self.goals[0]
        x_new, y_new, z_new = x + goal_x, y + goal_y, z + goal_z
        self.goals = np.array([[x_new, y_new, z_new] for _ in range(self.num_agents)])

        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

        return infos, rewards

    def update_formation_size(self, new_formation_size):
        pass

    def reset(self, obst_map=None):
        # Update duration time
        duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

        self.obstacle_map = obst_map
        if obst_map is None:
            raise NotImplementedError

        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))

        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Generate goals
        self.formation_center = self.generate_pos_obst_map() # prevent drones from crashing into the wall
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                         layer_dist=0.0)
