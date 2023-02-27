import numpy as np
import math
import octomap
import random

from gym_art.quadrotor_multi.utils.quad_utils import EPS


class OctTreeBase:
    def __init__(self, obstacle_size=1.0, room_dims=np.array([10, 10, 10]), resolution=0.05, obst_shape='cube'):
        self.start_points = None
        self.resolution = resolution
        self.octree = octomap.OcTree(self.resolution)
        self.room_dims = np.array(room_dims)
        self.half_room_length = self.room_dims[0] / 2
        self.half_room_width = self.room_dims[1] / 2
        self.grid_size = obstacle_size
        self.size = obstacle_size
        self.obst_shape = obst_shape
        self.start_range = np.zeros((2, 2))
        self.end_range = np.zeros((2, 2))
        self.init_box = np.array([[-0.5, -0.5, -0.5 * 2.0], [0.5, 0.5, 1.5 * 2.0]])
        self.pos_arr = None

    def reset(self):
        del self.octree
        self.octree = octomap.OcTree(self.resolution)
        # Allows add_node and remove_node to be deterministic
        self.octree.setProbMiss(0.0)
        self.octree.setProbHit(1.0)
        return
    def add_node(self, pos):
        self.octree.updateNode(pos, True)

    def remove_node(self, pos):
        self.octree.updateNode(pos, False)

    def update_sdf(self):
        self.octree.dynamicEDT_update(True)

    def generate_sdf(self):
        # max_dist: clamps distances at maxdist
        max_dist = 100
        bottom_left = np.array([-1.0 * self.room_dims[0], -1.0 * self.room_dims[1], 0]) - 2.0 * self.resolution
        upper_right = np.array([1.0 * self.room_dims[0], 1.0 * self.room_dims[1], self.room_dims[2]]) + 2.0 * self.resolution

        self.octree.dynamicEDT_generate(maxdist=max_dist,
                                        bbx_min=bottom_left,
                                        bbx_max=upper_right,
                                        treatUnknownAsOccupied=False)
        self.octree.dynamicEDT_update(True)

    def sdf_dist(self, p):
        return self.octree.dynamicEDT_getDistance(p)

    def get_surround(self, p):
        # Get SDF in xy plane
        state = []
        for x in np.arange(p[0] - self.resolution, p[0] + self.resolution + EPS, self.resolution):
            for y in np.arange(p[1] - self.resolution, p[1] + self.resolution + EPS, self.resolution):
                state.append(self.sdf_dist(np.array([x, y, p[2]])))

        state = np.array(state)
        return state

    def get_surround_z(self, p):
        state = []
        for x in np.arange(p[0] - self.resolution, p[0] + self.resolution + EPS, self.resolution):
            for y in np.arange(p[1] - self.resolution, p[1] + self.resolution + EPS, self.resolution):
                for z in np.arange(p[2] - self.resolution, p[2] + self.resolution + EPS, self.resolution):
                    state.append(self.sdf_dist(np.array([x, y, z])))

        state = np.array(state)
        return state
