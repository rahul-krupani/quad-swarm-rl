import numpy as np

from gym_art.quadrotor_multi.utils.quadrotor_multi_neighbor_utils import calculate_nei_dist_matrix


class NeighborObs:
    def __init__(self, num_agents=8, room_dims=np.array([10, 10, 10]), resolution=0.1):
        self.num_agents = num_agents
        self.prev_locations = []
        self.room_dims = room_dims

    def reset(self, obs=None, quads_pos=None):
        neighbor_obs = []

        for index in range(len(quads_pos)):
            neighbor_obs.append(calculate_nei_dist_matrix(quads_pos, index))

        obs = np.concatenate((obs, neighbor_obs), axis=1)

        return obs

    def step(self, obs=None, quads_pos=None):
        neighbor_obs = []

        for index in range(len(quads_pos)):
            neighbor_obs.append(calculate_nei_dist_matrix(quads_pos, index))

        obs = np.concatenate((obs, neighbor_obs), axis=1)

        return obs
