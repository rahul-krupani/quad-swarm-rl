import copy
import numpy as np

from gym_art.quadrotor_multi.obstacles.utils import get_surround_sdfs, collision_detection, get_ToFs_depthmap


class MultiObstacles:
    def __init__(self, obstacle_size=1.0, quad_radius=0.046, obs_type='octomap', obst_noise=0.0):
        self.size = obstacle_size
        self.obstacle_radius = obstacle_size / 2.0
        self.quad_radius = quad_radius
        self.pos_arr = []
        self.resolution = 0.1
        self.obst_noise = obst_noise
        self.obs_type = obs_type
        self.fov_angle = 45 * np.pi / 180
        self.scan_angle_arr = np.array([0., np.pi/2, np.pi, -np.pi/2])
        self.num_rays = 4

    def reset(self, obs, quads_pos, pos_arr, quads_rots=None):
        if len(pos_arr) == 0:
            pos_arr = np.array([[10., 10., 5.]])

        self.pos_arr = copy.deepcopy(np.array(pos_arr))

        #quads_sdf_obs = 2.0 * np.ones((len(quads_pos), len(self.scan_angle_arr) * self.num_rays))

        if self.obs_type == 'octomap':
            quads_sdf_obs = 100 * np.ones((len(quads_pos), 9))
            quads_sdf_obs = get_surround_sdfs(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],
                                              quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius,
                                              resolution=self.resolution)
        else:
            quads_sdf_obs = get_ToFs_depthmap(quad_poses=quads_pos, obst_poses=self.pos_arr,
                                              obst_radius=self.obstacle_radius, scan_max_dist=2.0,
                                              quad_rotations=quads_rots, scan_angle_arr=self.scan_angle_arr,
                                              fov_angle=self.fov_angle, num_rays=self.num_rays, obst_noise=self.obst_noise)

        obs = np.concatenate((obs, quads_sdf_obs), axis=1)

        return obs

    def step(self, obs, quads_pos, quads_rots=None):

        # quads_sdf_obs = 2.0 * np.ones((len(quads_pos), len(self.scan_angle_arr) * self.num_rays))

        if self.obs_type == 'octomap':
            quads_sdf_obs = 100 * np.ones((len(quads_pos), 9))
            quads_sdf_obs = get_surround_sdfs(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],
                                              quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius,
                                              resolution=self.resolution)
        else:
            quads_sdf_obs = get_ToFs_depthmap(quad_poses=quads_pos, obst_poses=self.pos_arr,
                                              obst_radius=self.obstacle_radius, scan_max_dist=2.0,
                                              quad_rotations=quads_rots, scan_angle_arr=self.scan_angle_arr,
                                              fov_angle=self.fov_angle, num_rays=self.num_rays, obst_noise=self.obst_noise)

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
