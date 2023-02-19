import math

import numpy as np

from gym_art.quadrotor_multi.octomap_creation import OctTree
from gym_art.quadrotor_multi.utils.quad_utils import EPS


class MultiObstacles:
    def __init__(self, num_obstacles=0, room_dims=np.array([10, 10, 10]), resolution=0.05, obstacle_size=1.0, obst_shape='cube'):
        self.num_obstacles = num_obstacles
        self.room_dims = np.array(room_dims)
        self.resolution = resolution
        self.obst_shape = obst_shape
        self.octree = OctTree(obstacle_size=1.0, room_dims=room_dims, resolution=resolution)
        self.grid_size = obstacle_size
        self.size = obstacle_size
        self.half_room_length = self.room_dims[0] / 2
        self.half_room_width = self.room_dims[1] / 2
        self.start_range = np.zeros((2, 2))
        self.end_range = np.zeros((2, 2))
        self.init_box = np.array([[-2.0, -2.0, -0.5 * 2.0], [2.0, 2.0, 1.5 * 2.0]])
        self.drone_pos = []
        self.cell_centers = [
            (i + (self.grid_size / 2) - self.half_room_length, -(j + (self.grid_size / 2) - self.half_room_width)) for i
            in
            np.arange(0, self.room_dims[0], self.grid_size) for j in np.arange(0, self.room_dims[1], self.grid_size)]
        self.pos_arr = None

    def reset(self, obs=None, quads_pos=None, start_point=np.array([0., 0., 2.]), end_point=np.array([0., 0., 2.])):
        self.octree.reset()
        self.generate_obstacles(num_obstacles=self.num_obstacles, start_point=start_point, end_point=end_point)

        obs = self.concate_obst_obs(quads_pos=quads_pos, obs=obs)
        return obs

    def step(self, obs=None, quads_pos=None):
        obs = self.concate_obst_obs(quads_pos=quads_pos, obs=obs)
        return obs

    def concate_obst_obs(self, quads_pos, obs):
        obst_obs = []

        for quad in quads_pos:
            surround_obs = self.octree.get_surround(quad)
            approx_part = np.random.uniform(low=-1.0 * self.resolution, high=0.0, size=surround_obs.shape)

            surround_obs += approx_part
            surround_obs = np.maximum(surround_obs, 0.0)
            obst_obs.append(surround_obs)

        obst_obs = np.array(obst_obs)

        # Extract closest obst
        self.extract_closest_obst_dist(obst_obs=obst_obs)

        # Add noise to obst_obs
        noise_part = np.random.normal(loc=0, scale=0.01, size=obst_obs.shape)
        obst_obs += noise_part

        obst_obs = np.maximum(obst_obs, 0.0)
        obs = np.concatenate((obs, obst_obs), axis=1)

        return obs

    def extract_closest_obst_dist(self, obst_obs):
        self.closest_obst_dist = []
        for item in obst_obs:
            tmp_item = item.flatten()
            center_idx = int(len(tmp_item) - 1 / 2)
            center_dist = tmp_item[center_idx]
            self.closest_obst_dist.append(center_dist)

        self.closest_obst_dist = np.array(self.closest_obst_dist)

    def collision_detection(self):
        drone_collision = np.where(self.closest_obst_dist < 0.06 + EPS)[0]
        return drone_collision

    def closest_obstacle(self, pos):
        rel_dist = np.linalg.norm(self.pos_arr[:, :2] - pos[:2], axis=1)
        closest_index = np.argmin(rel_dist)
        closest = self.pos_arr[closest_index]
        return closest

    def check_pos(self, pos_xy, goal_range):
        min_pos = goal_range[0] - np.array([0.5 * self.size, 0.5 * self.size])
        max_pos = goal_range[1] + np.array([0.5 * self.size, 0.5 * self.size])
        closest_point = np.maximum(min_pos, np.minimum(pos_xy, max_pos))
        closest_dist = np.linalg.norm(pos_xy - closest_point)
        if closest_dist <= 0.25:
            # obstacle collide with the spawn range of drones
            return True
        else:
            return False

    def gaussian_pos(self, goal_start_point=np.array([-3.0, -2.0, 2.0]), goal_end_point=np.array([3.0, 2.0, 2.0]),
                     y_gaussian_scale=None):
        middle_point = (goal_start_point + goal_end_point) / 2

        goal_vector = goal_end_point - goal_start_point
        goal_distance = np.linalg.norm(goal_vector)

        alpha = math.atan2(goal_vector[1], goal_vector[0])

        pos_x = np.random.normal(loc=middle_point[0], scale=goal_distance / 4.0)
        if y_gaussian_scale is None:
            y_gaussian_scale = np.random.uniform(low=0.2, high=0.5)

        pos_y = np.random.normal(loc=middle_point[1], scale=y_gaussian_scale)

        rot_pos_x = middle_point[0] + math.cos(alpha) * (pos_x - middle_point[0]) - math.sin(alpha) * (
                pos_y - middle_point[1])
        rot_pos_y = middle_point[1] + math.sin(alpha) * (pos_x - middle_point[0]) + math.cos(alpha) * (
                pos_y - middle_point[1])

        rot_pos_x = np.clip(rot_pos_x, a_min=-self.half_room_length + self.grid_size,
                            a_max=self.half_room_length - self.grid_size)
        rot_pos_y = np.clip(rot_pos_y, a_min=-self.half_room_width + self.grid_size,
                            a_max=self.half_room_width - self.grid_size)

        if self.resolution >= 0.1:
            pos_xy = np.around([rot_pos_x, rot_pos_y], decimals=1)
        else:
            raise NotImplementedError(f'Current obstacle resolution: {self.resolution} is not supported!')

        collide_start = self.check_pos(pos_xy, self.start_range)
        collide_end = self.check_pos(pos_xy, self.end_range)
        collide_flag = collide_start or collide_end

        return pos_xy, collide_flag

    @staticmethod
    def y_gaussian_generation(regen_id=0):
        if regen_id < 3:
            return None

        y_low = 0.13 * regen_id - 0.1
        y_high = y_low * np.random.uniform(low=1.5, high=2.5)
        y_gaussian_scale = np.random.uniform(low=y_low, high=y_high)
        return y_gaussian_scale

    def get_pos_no_overlap(self, pos_item, pos_arr, min_gap=0.2):
        # In this function, we assume the shape of all obstacles is cube
        # But even if we have this assumption, we can still roughly use it for shapes like cylinder
        if len(pos_arr) == 0:
            return False

        overlap_flag = False
        for j in range(len(pos_arr)):
            # TODO: This function only supports for cylinder
            if np.linalg.norm(pos_item[:2] - pos_arr[j][:2]) < self.size + min_gap:
                overlap_flag = True
                break
        return overlap_flag

    def generate_obstacles(self, num_obstacles=0, start_point=np.array([-3.0, -2.0, 2.0]),
                           end_point=np.array([3.0, 2.0, 2.0])):
        self.pos_arr = []
        self.start_range = np.array([start_point[:2] + self.init_box[0][:2], start_point[:2] + self.init_box[1][:2]])
        self.end_range = np.array([end_point[:2] + self.init_box[0][:2], end_point[:2] + self.init_box[1][:2]])
        pos_z = 0.5 * self.room_dims[2]
        for i in range(num_obstacles):
            for regen_id in range(20):
                y_gaussian_scale = self.y_gaussian_generation(regen_id=regen_id)
                pos_xy, collide_flag = self.gaussian_pos(y_gaussian_scale=y_gaussian_scale,
                                                         goal_start_point=start_point, goal_end_point=end_point)
                pos_item = np.array([pos_xy[0], pos_xy[1], pos_z])
                overlap_flag = self.get_pos_no_overlap(pos_item=pos_item, pos_arr=self.pos_arr)
                if collide_flag is False and overlap_flag is False:
                    self.pos_arr.append(pos_item)
                    break

        self.pos_arr = np.around(self.pos_arr, decimals=1)
        self.mark_octree()
        self.octree.generate_sdf()

        return self.pos_arr

    def mark_octree(self):
        self.mark_obstacles()
        self.mark_walls()

    def mark_obstacles(self):
        range_shape = 0.5 * self.size
        # Mark obstacles
        for item in self.pos_arr:
            # Add self.resolution: when drones hit the wall, they can still get proper surrounding value
            xy_min = np.maximum(item[:2] - range_shape, -0.5 * self.room_dims[:2] - self.resolution)
            xy_max = np.minimum(item[:2] + range_shape, 0.5 * self.room_dims[:2] + self.resolution)

            range_x = np.arange(xy_min[0], xy_max[0], self.resolution)
            range_x = np.around(range_x, decimals=1)

            range_y = np.arange(xy_min[1], xy_max[1], self.resolution)
            range_y = np.around(range_y, decimals=1)

            range_z = np.arange(0, self.room_dims[2] + self.resolution, self.resolution)
            range_z = np.around(range_z, decimals=1)

            if self.obst_shape == 'cube':
                for x in range_x:
                    for y in range_y:
                        for z in range_z:
                            self.octree.add_node([x, y, z])
            elif self.obst_shape == 'cylinder':
                for x in range_x:
                    for y in range_y:
                        if np.linalg.norm(np.array([x, y]) - item[:2]) <= self.size / 2:
                            for z in range_z:
                                self.octree.add_node([x, y, z])
            else:
                raise NotImplementedError(f'{self.obst_shape} is not supported!')

    def mark_walls(self):
        bottom_left = np.array([-0.5 * self.room_dims[0] - self.resolution, -0.5 * self.room_dims[1] - self.resolution, 0.0])
        upper_right = np.array([0.5 * self.room_dims[0], 0.5 * self.room_dims[1], self.room_dims[2]])

        range_x = np.arange(bottom_left[0] + self.resolution, upper_right[0], self.resolution)
        range_x = np.around(range_x, decimals=1)

        range_y = np.arange(bottom_left[1], upper_right[1] + self.resolution, self.resolution)
        range_y = np.around(range_y, decimals=1)

        range_z = np.arange(0, self.room_dims[2] + self.resolution, self.resolution)
        range_z = np.around(range_z, decimals=1)

        for x in [bottom_left[0], upper_right[0]]:
            for y in range_y:
                for z in range_z:
                    self.octree.add_node([x, y, z])

        for y in [bottom_left[1], upper_right[1]]:
            for x in range_x:
                for z in range_z:
                    self.octree.add_node([x, y, z])
