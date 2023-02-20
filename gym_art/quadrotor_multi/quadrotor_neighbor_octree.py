import numpy as np

from gym_art.quadrotor_multi.octomap_creation import OctTree


class NeighborOctree:
    def __init__(self, num_agents=8, room_dims=np.array([10, 10, 10]), resolution=0.1):
        self.num_agents = num_agents
        self.locations = []
        self.room_dims = room_dims
        self.octree = OctTree(obstacle_size=1.0, room_dims=room_dims, resolution=resolution)

    def reset(self, obs=None, quads_pos=None):
        self.octree.reset()
        for pos in quads_pos:
            self.octree.add_node(pos)
            self.locations.append(pos)

        self.octree.generate_sdf()

        neighbor_obs = []

        for pos in quads_pos:
            neighbor_obs.append(self.get_state(pos))

        obs = np.concatenate((obs, neighbor_obs), axis=1)

        return obs

    def step(self, obs=None, quads_pos=None):
        while len(self.locations) > 0:
            self.octree.remove_node(self.locations.pop(0))

        for pos in quads_pos:
            self.octree.add_node(pos)
            self.locations.append(pos)

        self.octree.update_sdf()

        neighbor_obs = []

        for pos in quads_pos:
            neighbor_obs.append(self.get_state(pos))

        obs = np.concatenate((obs, neighbor_obs), axis=1)

        return obs

    def get_state(self, pos):

        self.octree.remove_node(pos)
        obs = self.octree.get_surround_z(pos)
        self.octree.add_node(pos)

        return obs

    def collision_detection(self, pos_quads=None):
        drone_collision = []

        for i, quad in enumerate(pos_quads):
            curr = self.octree.sdf_dist(quad)
            if curr < 0.1 + 1e-5:
                drone_collision.append(i)

        return drone_collision
