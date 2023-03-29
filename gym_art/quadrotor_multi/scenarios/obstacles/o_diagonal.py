import numpy as np
import copy

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base


class Scenario_o_diagonal(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                 quads_formation_size):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                         quads_formation_size)
        self.free_x = [-self.room_dims[0] / 2 + 2, self.room_dims[0] / 2 - 2,
                       -self.room_dims[0] / 2 + 2, self.room_dims[0] / 2 - 2]

        self.free_y = [-self.room_dims[1] / 2 + 2, -self.room_dims[1] / 2 + 2,
                       self.room_dims[1] / 2 - 2, self.room_dims[1] / 2 - 2]

    def update_formation_size(self, new_formation_size):
        pass

    def step(self, infos, rewards):
        tick = self.envs[0].tick

        if tick <= int(self.duration_time * self.envs[0].control_freq):
            return infos, rewards

        self.duration_time += self.envs[0].ep_time + 1
        for i, env in enumerate(self.envs):
            env.goal = self.end_point[i]

        return infos, rewards

    def single_agent_start_end_point(self):
        start_quadrant = np.random.randint(low=0, high=4)

        xy_noise = np.random.uniform(low=-0.2, high=0.2, size=2)
        start_x, start_y = self.free_x[start_quadrant], self.free_y[start_quadrant]
        start_z = np.random.uniform(low=1.5, high=3.5)
        start_point = np.array((start_x + xy_noise[0], start_y + xy_noise[1], start_z))

        end_quadrant = 3 - start_quadrant
        xy_noise = np.random.uniform(low=-0.2, high=0.2, size=2)
        end_x, end_y = self.free_x[end_quadrant], self.free_y[end_quadrant]
        end_z = np.random.uniform(low=1.5, high=3.5)
        end_point = np.array((end_x + xy_noise[0], end_y + xy_noise[1], end_z))

        return start_point, end_point, start_quadrant

    def reset(self, obst_map=None, cell_centers=None):
        self.start_point = []
        self.end_point = []

        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        if obst_map is None or cell_centers is None:
            raise NotImplementedError

        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))

        start_point, end_point, start_quadrant = self.single_agent_start_end_point()
        self.start_point.append(start_point)
        self.end_point.append(end_point)

        self.start_point = np.array(self.start_point)
        self.end_point = np.array(self.end_point)

        self.duration_time = np.random.uniform(low=4.0, high=6.0)
        self.update_formation_and_relate_param()

        formation_z = np.random.uniform(low=1.5, high=3.5)
        self.formation_center = np.array((self.free_x[start_quadrant], self.free_y[start_quadrant], formation_z))

        self.goals = copy.deepcopy(self.start_point)
