import numpy as np
import math
import octomap
import random

from gym_art.quadrotor_multi.quad_utils import EPS


class OctTree:
    def __init__(self, obstacle_size=1.0, room_dims=np.array([10, 10, 10]), resolution=0.05, obst_shape='cube'):
        self.start_points = None
        self.resolution = resolution
        self.octree = octomap.OcTree(self.resolution)
        self.room_dims = np.array(room_dims)

    def reset(self):
        del self.octree
        self.octree = octomap.OcTree(self.resolution)
        return

    def add_node(self, pos):
        self.octree.updateNode(pos, True)

    def remove_node(self, pos):
        self.octree.updateNode(pos, False)

    def update_sdf(self):
        self.octree.dynamicEDT_update(True)

    def generate_sdf(self):
        # max_dist: clamps distances at maxdist
        max_dist = 1.0
        bottom_left = np.array([-1.0 * self.room_dims[0], -1.0 * self.room_dims[1], 0])
        upper_right = np.array([1.0 * self.room_dims[0], 1.0 * self.room_dims[1], self.room_dims[2]])

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