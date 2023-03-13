import numpy as np
from numba import njit

from scipy import spatial

from gym_art.quadrotor_multi.octomap_creation import OctTree
from gym_art.quadrotor_multi.quad_utils import EPS


class MultiObstacles:
    def __init__(self, num_obstacles=0, room_dims=np.array([10, 10, 10]), resolution=0.05,
                 obstacle_size=1.0, obst_shape="cube", obst_obs_type='octomap', obst_local_num=6):
        self.num_obstacles = num_obstacles
        self.obstacles = []
        self.room_dims = room_dims
        self.obst_shape = obst_shape
        self.obstacle_size = obstacle_size
        self.obst_obs_type = obst_obs_type
        self.obst_local_num = obst_local_num
        self.pos_arr = None
        self.octree = OctTree(obstacle_size=self.obstacle_size, room_dims=room_dims,
                              resolution=resolution, obst_shape=self.obst_shape)

    def reset(self, obs=None, quads_pos=None, pos_arr=None):
        self.octree.reset()
        self.octree.set_obst(pos_arr)
        self.pos_arr = np.array(pos_arr)

        obst_obs = []

        if self.obst_obs_type == 'octomap':
            for quad in quads_pos:
                obst_obs.append(self.octree.get_surround(quad))
        elif self.obst_obs_type == 'closest_pos':
            for quad in quads_pos:
                obst_rel_pos = n_nearest_obstacles(quad, self.pos_arr, self.obst_local_num)
                while obst_rel_pos.shape[0] < self.obst_local_num:
                    np.append(obst_rel_pos, np.zeros(3))
                obst_obs.append(np.array(closest_points(obst_rel_pos, self.obstacle_size)).flatten())




        obs = np.concatenate((obs, obst_obs), axis=1)

        return obs

    def step(self, obs=None, quads_pos=None):
        obst_obs = []

        if self.obst_obs_type == 'octomap':
            for quad in quads_pos:
                obst_obs.append(self.octree.get_surround(quad))
        elif self.obst_obs_type == 'closest_pos':
            for quad in quads_pos:
                obst_rel_pos = n_nearest_obstacles(quad, self.pos_arr, self.obst_local_num)
                while obst_rel_pos.shape[0] < self.obst_local_num:
                    np.append(obst_rel_pos, np.zeros(3))
                obst_obs.append(np.array(closest_points(obst_rel_pos, self.obstacle_size)).flatten())

        obs = np.concatenate((obs, obst_obs), axis=1)

        return obs

    def collision_detection(self, pos_quads=None):
        drone_collision = []

        for i, quad in enumerate(pos_quads):
            curr = self.octree.sdf_dist(quad)
            if curr < 0.1 + 1e-5:
                drone_collision.append(i)

        return drone_collision


    def closest_obstacle(self, pos):
        rel_dist = np.linalg.norm(self.octree.pos_arr[:, :2] - pos[:2], axis=1)
        closest_index = np.argmin(rel_dist)
        closest = self.octree.pos_arr[closest_index]
        return closest

@njit
def n_nearest_obstacles(quad, pos, n):
    rel_pos = np.subtract(pos, quad)
    dists = []

    for i in rel_pos:
        dists.append(np.linalg.norm(i))

    idx = np.argsort(np.array(dists))
    return rel_pos[idx[:n]]
@njit
def closest_points(pos, size):
    posxy = []
    for i in range(len(pos)):
        mag = np.linalg.norm(pos[i]) + EPS
        posxy.append((pos[i][:2]/mag)*(mag-size/2))

    return posxy