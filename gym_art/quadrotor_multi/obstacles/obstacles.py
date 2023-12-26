import copy
import numpy as np

from gym_art.quadrotor_multi.obstacles.utils import get_surround_sdfs, collision_detection, get_surround_multi_ranger, \
    get_surround_multi_ranger_depth, get_surround_multi_ranger_4x4_depth
import matplotlib.pyplot as plt
import matplotlib
#from gym_art.quadrotor_multi.obstacles.mapping import Mapping


class MultiObstacles:
    def __init__(self, obstacle_size=1.0, quad_radius=0.046, room_dims=[10., 10., 10.]):
        self.size = obstacle_size
        self.obstacle_radius = obstacle_size / 2.0
        self.quad_radius = quad_radius
        self.pos_arr = []
        self.resolution = 0.1
        self.obstacle_height = 10.
        self.room_dims = np.array(room_dims)
        self.count = 0
        self.hist = []
        self.scan_angle_arr = np.array([0., np.pi / 2, np.pi, -np.pi / 2])
        # self.SDF = Mapping(size=10, resolution=10)
        # self.SDF.create_empty_map()

    def reset(self, obs, quads_pos, pos_arr, quads_rot):
        self.pos_arr = copy.deepcopy(np.array(pos_arr))
        # quads_sdf_obs = 100 * np.ones((len(quads_pos), 9))
        # quads_sdf_obs = get_surround_sdfs(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],
        #                                   quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius,
        #                                   resolution=self.resolution)

        # self.obstacle_heights = np.array([self.obstacle_height for i in range(len(pos_arr))])
        # q_yaws = []
        # for q_id in range(len(quads_pos)):
        #     q_yaws.append(np.array([np.arctan2(quads_rot[q_id][1, 0], quads_rot[q_id][0, 0])]))
        # q_yaws = np.array(q_yaws)
        #
        # pos_xy_yaw = np.concatenate((quads_pos[:, :2], q_yaws), axis=1)


        quads_sdf_obs = get_surround_multi_ranger_depth(quad_poses=quads_pos, obst_poses=self.pos_arr,
                                                        obst_radius=self.obstacle_radius,
                                                        scan_max_dist=4.0,
                                                        quad_rotations=quads_rot)

        # quads_sdf_obs = get_surround_sdfs(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],
        #                                   quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius,
        #                                   resolution=self.resolution)

        # dist = get_surround_multi_ranger(quad_poses=quads_pos, obst_poses=self.pos_arr, obst_radius=self.obstacle_radius,
        #                               obst_heights=self.obstacle_heights, room_dims=self.room_dims, scan_max_dist=4.0,
        #                               quad_rotations=quads_rot)
        #
        # for agent in range(pos_xy_yaw.shape[0]):
        #     self.SDF.update_grid_map(np.expand_dims(dist[agent, :4], axis=0).T, np.expand_dims(self.scan_angle_arr, axis=0).T, np.expand_dims(pos_xy_yaw[agent, :], axis=0).T)
        #quads_sdf_obs = np.concatenate((quads_sdf_obs, quads_pos, q_yaws), axis=1)

        obs = np.concatenate((obs, quads_sdf_obs), axis=1)
        self.count += 1

        return obs

    def step(self, obs, quads_pos, quads_rot):
        # quads_sdf_obs = 100 * np.ones((len(quads_pos), 9))
        # quads_sdf_obs = get_surround_sdfs(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius, resolution=self.resolution)
        # [1, 3, 4, 5, 7]

        # q_yaws = []
        # for q_id in range(len(quads_pos)):
        #     q_yaws.append(np.array([np.arctan2(quads_rot[q_id][1, 0], quads_rot[q_id][0, 0])]))
        # q_yaws = np.array(q_yaws)
        #
        # pos_xy_yaw = np.concatenate((quads_pos[:, :2], q_yaws), axis=1)
        #
        quads_sdf_obs = get_surround_multi_ranger_depth(quad_poses=quads_pos, obst_poses=self.pos_arr, obst_radius=self.obstacle_radius,
                                      scan_max_dist=4.0, quad_rotations=quads_rot)

        # dist = get_surround_multi_ranger(quad_poses=quads_pos, obst_poses=self.pos_arr, obst_radius=self.obstacle_radius,
        #                               obst_heights=self.obstacle_heights, room_dims=self.room_dims, scan_max_dist=4.0,
        #                               quad_rotations=quads_rot)
        #
        # for agent in range(pos_xy_yaw.shape[0]):
        #     self.SDF.update_grid_map(np.expand_dims(dist[agent, :4], axis=0).T,
        #                              np.expand_dims(self.scan_angle_arr, axis=0).T,
        #                              np.expand_dims(pos_xy_yaw[agent, :], axis=0).T)

        # self.SDF.update_grid_map(dist[:, :4].T, np.expand_dims(self.scan_angle_arr, axis=0).T, pos_xy_yaw.T)

        #quads_sdf_obs = np.concatenate((quads_sdf_obs, quads_pos, q_yaws), axis=1)

        obs = np.concatenate((obs, quads_sdf_obs), axis=1)
        #self.count += 1

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
