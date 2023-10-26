import copy
import numpy as np

from gym_art.quadrotor_multi.obstacles.utils import get_surround_sdfs, collision_detection, get_surround_multi_ranger


class MultiObstacles:
    def __init__(self, obstacle_size=1.0, quad_radius=0.046, room_dims=[10., 10., 10.]):
        self.size = obstacle_size
        self.obstacle_radius = obstacle_size / 2.0
        self.quad_radius = quad_radius
        self.pos_arr = []
        self.resolution = 0.1
        self.obstacle_height = 10.
        self.room_dims = np.array(room_dims)

    def reset(self, obs, quads_pos, pos_arr, quads_rot):
        self.pos_arr = copy.deepcopy(np.array(pos_arr))
        self.obstacle_heights = np.array([self.obstacle_height for i in range(len(pos_arr))])

        # quads_sdf_obs = 100 * np.ones((len(quads_pos), 5))
        # quads_sdf_obs = get_surround_sdfs(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius, resolution=self.resolution)

        quads_sdf_obs = get_surround_multi_ranger(quad_poses=quads_pos, obst_poses=self.pos_arr, obst_radius=self.obstacle_radius,
                                  obst_heights=self.obstacle_heights, room_dims=self.room_dims,
                                  scan_max_dist=4.0, quad_rotations=quads_rot)

        obs = np.concatenate((obs, quads_sdf_obs), axis=1)

        return obs

    def step(self, obs, quads_pos, quads_rot):
        # quads_sdf_obs = 100 * np.ones((len(quads_pos), 5))
        # quads_sdf_obs = get_surround_sdfs(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius, resolution=self.resolution)

        quads_sdf_obs = get_surround_multi_ranger(quad_poses=quads_pos, obst_poses=self.pos_arr, obst_radius=self.obstacle_radius,
                                 obst_heights=self.obstacle_heights, room_dims=self.room_dims,
                                 scan_max_dist=4.0, quad_rotations=quads_rot)

        obs = np.concatenate((obs, quads_sdf_obs), axis=1)

        return obs

    def collision_detection(self, pos_quads):
        quad_collisions = collision_detection(quad_poses=pos_quads[:, :2], obst_poses=self.pos_arr[:, :2],
                                              obst_radius=self.obstacle_radius, quad_radius=self.quad_radius)

        collided_quads_id = np.where(quad_collisions > -1)[0]
        collided_obstacles_id = quad_collisions[collided_quads_id]
        quad_obst_pair = {}
        for i, key in enumerate(collided_quads_id):
            quad_obst_pair[key] = int(collided_obstacles_id[i])

        return collided_quads_id, quad_obst_pair
