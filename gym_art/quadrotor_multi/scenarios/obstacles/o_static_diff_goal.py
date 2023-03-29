import numpy as np

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base


class Scenario_o_static_diff_goal(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                 quads_formation_size):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                         quads_formation_size)
        # teleport every [4.0, 6.0] secs
        duration_time = 6.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

    def step(self, infos, rewards):
        return infos, rewards

    def reset(self, obst_map=None, cell_centers=None):
        # Update duration time
        duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        if obst_map is None or cell_centers is None:
            raise NotImplementedError

        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))

        # Reset formation, and parameters related to the formation; formation center; goals
        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        self.formation_center = self.generate_pos_obst_map()

        # Regenerate goals, we don't have to assign goals to the envs,
        # the reset function in quadrotor_multi.py would do that
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                         layer_dist=self.layer_dist)
        np.random.shuffle(self.goals)

    # def check_surroundings(self, row, col):
    #     length, width = self.obstacle_map.shape[0], self.obstacle_map.shape[1]
    #     obstacle_map = self.obstacle_map
    #     # Check if the given position is out of bounds
    #     if row < 0 or row >= width or col < 0 or col >= length:
    #         raise ValueError("Invalid position")
    #
    #     # Check if the surrounding cells are all 0s
    #     # Check cell above
    #     if row > 0 and obstacle_map[row - 1][col] != 0:
    #         return False
    #     # Check cell below
    #     if row < width - 1 and obstacle_map[row + 1][col] != 0:
    #         return False
    #     # Check cell to the left
    #     if col > 0 and obstacle_map[row][col - 1] != 0:
    #         return False
    #     # Check cell to the right
    #     if col < length - 1 and obstacle_map[row][col + 1] != 0:
    #         return False
    #
    #     return True

