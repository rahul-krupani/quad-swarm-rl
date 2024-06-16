import copy
import numpy as np

from gym_art.quadrotor_multi.obstacles.utils import get_surround_sdfs, collision_detection, get_ToFs_depthmap, \
    get_ToFs_depthmap_subset
from scipy.spatial import KDTree


class MultiObstacles:
    def __init__(self, obstacle_size=1.0, quad_radius=0.046, obs_type='octomap', obst_noise=0.0, obst_tof_resolution=4, grid_size=1.):
        self.size = obstacle_size
        self.obstacle_radius = obstacle_size / 2.0
        self.quad_radius = quad_radius
        self.pos_arr = []
        self.resolution = 0.1
        self.obs_type = obs_type
        self.obst_noise = obst_noise
        self.fov_angle = 45 * np.pi / 180
        self.scan_angle_arr = np.array([0., np.pi/2, np.pi, -np.pi/2])
        self.num_rays = obst_tof_resolution
        self.grid_size = grid_size
        self.obst_subset, self.obst = [], []
        self.n = int(2.0 / self.grid_size)
        self.if_speedup = True

    def reset(self, obs, quads_pos, pos_arr, quads_rots=None, cell_centers=None, obst_map=None, obst_area_length=8):
        self.pos_arr = copy.deepcopy(np.array(pos_arr))

        if self.obs_type == 'octomap':
            quads_sdf_obs = 100 * np.ones((len(quads_pos), 9))
            quads_sdf_obs = get_surround_sdfs(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],
                                              quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius,
                                              resolution=self.resolution)
        else:
            if self.if_speedup:
                self.cell_centers = cell_centers
                self.tree = KDTree(self.cell_centers)
                obst_subset = self.reduce_obst_list(quads_pos, obst_map, obst_area_length)

                quads_sdf_obs = get_ToFs_depthmap_subset(quad_poses=quads_pos, obst_poses=obst_subset,
                                                          obst_radius=self.obstacle_radius, scan_max_dist=2.0,
                                                          quad_rotations=quads_rots, scan_angle_arr=self.scan_angle_arr,
                                                          fov_angle=self.fov_angle, num_rays=self.num_rays,
                                                          obst_noise=self.obst_noise)

            else:
                quads_sdf_obs = get_ToFs_depthmap(quad_poses=quads_pos, obst_poses=self.pos_arr,
                                              obst_radius=self.obstacle_radius, scan_max_dist=2.0,
                                              quad_rotations=quads_rots, scan_angle_arr=self.scan_angle_arr,
                                              fov_angle=self.fov_angle, num_rays=self.num_rays, obst_noise=self.obst_noise)


        obs = np.concatenate((obs, quads_sdf_obs), axis=1)

        return obs

    def step(self, obs, quads_pos, quads_rots=None, obst_map=None, obst_area_length=8):
        if self.obs_type == 'octomap':
            quads_sdf_obs = 100 * np.ones((len(quads_pos), 9))
            quads_sdf_obs = get_surround_sdfs(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],
                                              quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius,
                                              resolution=self.resolution)
        else:
            if self.if_speedup:
                obst_subset = self.reduce_obst_list(quads_pos, obst_map, obst_area_length)

                quads_sdf_obs = get_ToFs_depthmap_subset(quad_poses=quads_pos, obst_poses=obst_subset,
                                                  obst_radius=self.obstacle_radius, scan_max_dist=2.0,
                                                  quad_rotations=quads_rots, scan_angle_arr=self.scan_angle_arr,
                                                  fov_angle=self.fov_angle, num_rays=self.num_rays, obst_noise=self.obst_noise)

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

    def reduce_obst_list(self, quads_pos, obst_map, obst_area_length):
        obst_subset, obst = [], []
        distance, index = self.tree.query(quads_pos[:, :2])

        for ind in index:
            obst_grid_length_num = int(obst_area_length // self.grid_size)
            rid = ind % obst_grid_length_num
            cid = (ind - ind % obst_grid_length_num) // obst_grid_length_num

            for i in range(-self.n, self.n + 1):
                for j in range(-self.n, self.n + 1):
                    if rid + i >= 0 and cid + j >= 0 and rid + i < obst_grid_length_num and cid + j < obst_grid_length_num:
                        if obst_map[rid + i, cid + j] == 1:
                            obst.append(list(self.cell_centers[(rid + i) + obst_grid_length_num * (cid + j)]) + [
                                self.pos_arr[0][2]])

            obst_subset.append(np.array(obst))

        return np.array(obst_subset)