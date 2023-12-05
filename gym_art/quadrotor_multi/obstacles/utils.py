import numpy as np
from numba import njit


@njit
def get_surround_sdfs(quad_poses, obst_poses, quads_sdf_obs, obst_radius, resolution=0.1):
    # Shape of quads_sdf_obs: (quad_num, 9)

    sdf_map = np.array([-1., -1., -1., 0., 0., 0., 1., 1., 1.])
    sdf_map *= resolution

    for i, q_pos in enumerate(quad_poses):
        q_pos_x, q_pos_y = q_pos[0], q_pos[1]

        for g_i, g_x in enumerate([q_pos_x - resolution, q_pos_x, q_pos_x + resolution]):
            for g_j, g_y in enumerate([q_pos_y - resolution, q_pos_y, q_pos_y + resolution]):
                grid_pos = np.array([g_x, g_y])

                min_dist = 100.0
                for o_pos in obst_poses:
                    dist = np.linalg.norm(grid_pos - o_pos)
                    if dist < min_dist:
                        min_dist = dist

                g_id = g_i * 3 + g_j
                quads_sdf_obs[i, g_id] = min_dist - obst_radius

    return quads_sdf_obs


@njit
def is_surface_in_cylinder_view(vector, q_pos, o_pos, o_radius, fov_angle):
    # Calculate the direction vector from the origin to the cylinder center
    direction_vector = o_pos - q_pos
    if np.linalg.norm(direction_vector) <= o_radius:
        return 0, None

    # Calculate the unit vector in the direction of the given view vector
    view_vector = vector / np.linalg.norm(vector)

    # Calculate the angle between the direction vector and the view vector
    angle = np.arccos(np.dot(direction_vector, view_vector) / (np.linalg.norm(direction_vector) * np.linalg.norm(view_vector)))

    if np.dot(direction_vector, view_vector) > 0:
        if angle <= fov_angle/2:
            return np.linalg.norm(direction_vector) - o_radius, 2*o_radius

        # Calculate the angle between the direction vector and the normal to the cylinder surface
        angle_to_surface = np.arcsin(o_radius / np.linalg.norm(direction_vector))

        edge_vector_1 = np.dot(np.array([[np.cos(angle_to_surface), -np.sin(angle_to_surface)],[np.sin(angle_to_surface), np.cos(angle_to_surface)]]), direction_vector)
        edge_vector_2 = np.dot(np.array([[np.cos(angle_to_surface), np.sin(angle_to_surface)],[-np.sin(angle_to_surface), np.cos(angle_to_surface)]]), direction_vector)

        edge_angle_1 = np.arccos(np.dot(edge_vector_1, view_vector) / (np.linalg.norm(edge_vector_1) * np.linalg.norm(view_vector)))
        edge_angle_2 = np.arccos(np.dot(edge_vector_2, view_vector) / (np.linalg.norm(edge_vector_2) * np.linalg.norm(view_vector)))

        # Case where edge is in FOV
        if edge_angle_1 <= fov_angle / 2 or edge_angle_2 <= fov_angle / 2:
            # Create a triangle with direction vector
            closest_dist = np.linalg.norm(direction_vector) * np.sin(angle - (fov_angle / 2))
            full_dist = np.linalg.norm(direction_vector) * np.cos(angle - (fov_angle / 2))
            if closest_dist <= o_radius:
                len_in_obst = (o_radius ** 2 - closest_dist ** 2) ** 0.5
                return full_dist - len_in_obst, 2 * len_in_obst
            else:
                return (None, None)
        # Case where full FOV is between center and edge
        if (np.dot(edge_vector_1, view_vector) > 0 and np.isclose(angle_to_surface, angle + edge_angle_1,
                                                                    atol=1e-5)) or (
                np.dot(edge_vector_1, view_vector) > 0 and np.isclose(angle_to_surface, angle + edge_angle_2,
                                                                        atol=1e-5)):
            # Create a triangle with direction vector
            closest_dist = np.linalg.norm(direction_vector) * np.sin(angle - (fov_angle / 2))
            full_dist = np.linalg.norm(direction_vector) * np.cos(angle - (fov_angle / 2))
            len_in_obst = (o_radius ** 2 - closest_dist ** 2) ** 0.5
            return full_dist - len_in_obst, 2 * len_in_obst
    return (None, None)

@njit
def get_surround_multi_ranger_depth(quad_poses, obst_poses, obst_radius, scan_max_dist,
                              quad_rotations):
        """
            quad_poses:     quadrotor positions, only with xy pos
            obst_poses:     obstacle positions, only with xy pos
            quad_vels:      quadrotor velocities, only with xy vel
            obst_radius:    obstacle raidus
        """
        quads_obs = scan_max_dist * np.ones((len(quad_poses), 4*4))
        scan_angle_arr = np.array([0., np.pi / 2, np.pi, -np.pi / 2])
        fov_angle = np.pi / 180 * 45
        sensor_offset = 0.01625
        modifications = np.array([-3 * (fov_angle / 8), -1 * (fov_angle / 8), (fov_angle / 8), 3 * (fov_angle / 8)])

        for q_id in range(len(quad_poses)):
            q_pos_xy = quad_poses[q_id][:2]
            q_yaw = np.arctan2(quad_rotations[q_id][1, 0], quad_rotations[q_id][0, 0])
            base_rad = q_yaw
            walls = np.array([[5, q_pos_xy[1]], [-5, q_pos_xy[1]], [q_pos_xy[0], 5], [q_pos_xy[1], -5]])
            for ray_id, rad_shift in enumerate(scan_angle_arr):
                for sec_id, sec in enumerate(modifications):
                    cur_rad = base_rad + rad_shift + sec
                    cur_dir = np.array([np.cos(cur_rad), np.sin(cur_rad)])
                    for w_id in range(len(walls)):
                        wall_dir = walls[w_id] - q_pos_xy
                        if np.dot(wall_dir, cur_dir) > 0:
                            angle = np.arccos(
                                np.dot(wall_dir, cur_dir) / (np.linalg.norm(wall_dir) * np.linalg.norm(cur_dir)))
                            if angle <= fov_angle / 8:
                                quads_obs[q_id][ray_id*4+sec_id] = min(quads_obs[q_id][ray_id*4+sec_id], (np.linalg.norm(wall_dir))-sensor_offset)
                            else:
                                quads_obs[q_id][ray_id*4+sec_id] = min(quads_obs[q_id][ray_id*4+sec_id],
                                                                       (np.linalg.norm(wall_dir) / np.cos(
                                                                  angle - (fov_angle / 8)))-sensor_offset)
                    for o_id in range(len(obst_poses)):
                        o_pos_xy = obst_poses[o_id][:2]

                        # Returns distance and length of the path inside the circle along the shortest distance vector
                        distance, circle_len = is_surface_in_cylinder_view(cur_dir, q_pos_xy, o_pos_xy, obst_radius,
                                                                           fov_angle / 4)
                        if distance is not None:
                            quads_obs[q_id][ray_id*4+sec_id] = min(quads_obs[q_id][ray_id*4+sec_id], distance-sensor_offset)

            # quads_obs[q_id][len(scan_angle_arr)] = min(quads_obs[q_id][len(scan_angle_arr)],room_dims[2] - q_z)

        quads_obs = np.clip(quads_obs, a_min=0.0, a_max=scan_max_dist)
        return quads_obs

@njit
def get_surround_multi_ranger_4x4_depth(quad_poses, obst_poses, obst_radius, scan_max_dist,
                                    quad_rotations):
    """
        quad_poses:     quadrotor positions, only with xy pos
        obst_poses:     obstacle positions, only with xy pos
        quad_vels:      quadrotor velocities, only with xy vel
        obst_radius:    obstacle raidus
    """
    quads_obs = scan_max_dist * np.ones((len(quad_poses), 4 * 4 * 4))
    scan_angle_arr = np.array([0., np.pi / 2, np.pi, -np.pi / 2])
    fov_angle = np.pi / 180 * 27
    sensor_offset = 0#0.01625
    modifications = np.array([-3 * (fov_angle / 8), -1 * (fov_angle / 8), (fov_angle / 8), 3 * (fov_angle / 8)])

    for q_id in range(len(quad_poses)):
        q_pos_xy = quad_poses[q_id][:2]
        q_yaw = np.arctan2(quad_rotations[q_id][1, 0], quad_rotations[q_id][0, 0])
        q_pitch = np.arctan2(-quad_rotations[q_id][2, 0],
                             np.sqrt(quad_rotations[q_id][2, 1] ** 2 + quad_rotations[q_id][2, 2] ** 2))
        q_roll = np.arctan2(quad_rotations[q_id][2, 1], quad_rotations[q_id][2, 2])
        deflect_angles = np.array([q_pitch, q_roll, -q_pitch, -q_roll])
        base_rad = q_yaw
        walls = np.array([[5, q_pos_xy[1]], [-5, q_pos_xy[1]], [q_pos_xy[0], 5], [q_pos_xy[1], -5]])
        for ray_id, rad_shift in enumerate(scan_angle_arr):
            # pitch +ve is up, roll +ve is down to the right
            for sec_id, sec in enumerate(modifications):
                cur_rad = base_rad + rad_shift + sec
                cur_dir = np.array([np.cos(cur_rad), np.sin(cur_rad)])
                for w_id in range(len(walls)):
                    wall_dir = walls[w_id] - q_pos_xy
                    if np.dot(wall_dir, cur_dir) > 0:
                        angle = np.arccos(
                            np.dot(wall_dir, cur_dir) / (np.linalg.norm(wall_dir) * np.linalg.norm(cur_dir)))
                        if angle <= fov_angle / 8:
                            for v_sec_id, v_sec in enumerate(modifications):
                                if deflect_angles[ray_id] + v_sec > 0:
                                    project_to = deflect_angles[ray_id] + v_sec - (fov_angle / 8)
                                    quads_obs[q_id][ray_id * 16 + sec_id * 4 + v_sec_id] = min(
                                        quads_obs[q_id][ray_id * 16 + sec_id * 4 + v_sec_id],
                                        (np.linalg.norm(wall_dir) / np.cos(project_to))-sensor_offset)
                                elif deflect_angles[ray_id] + v_sec < 0:
                                    project_to = deflect_angles[ray_id] + v_sec + (fov_angle / 8)
                                    quads_obs[q_id][ray_id * 16 + sec_id * 4 + v_sec_id] = min(
                                        quads_obs[q_id][ray_id * 16 + sec_id * 4 + v_sec_id],
                                        (np.linalg.norm(wall_dir) / np.cos(project_to))-sensor_offset)
                                else:
                                    quads_obs[q_id][ray_id * 16 + sec_id * 4 + v_sec_id] = min(
                                        quads_obs[q_id][ray_id * 16 + sec_id * 4 + v_sec_id], np.linalg.norm(wall_dir)-sensor_offset)
                        else:
                            for v_sec_id, v_sec in enumerate(modifications):
                                distance = np.linalg.norm(wall_dir) / np.cos(angle - (fov_angle / 8))
                                if deflect_angles[ray_id] + v_sec > 0:
                                    project_to = deflect_angles[ray_id] + v_sec - (fov_angle / 8)
                                    quads_obs[q_id][ray_id * 16 + sec_id * 4 + v_sec_id] = min(
                                        quads_obs[q_id][ray_id * 16 + sec_id * 4 + v_sec_id],
                                        (distance / np.cos(project_to))-sensor_offset)
                                elif deflect_angles[ray_id] + v_sec < 0:
                                    project_to = deflect_angles[ray_id] + v_sec + (fov_angle / 8)
                                    quads_obs[q_id][ray_id * 16 + sec_id * 4 + v_sec_id] = min(
                                        quads_obs[q_id][ray_id * 16 + sec_id * 4 + v_sec_id],
                                        (distance / np.cos(project_to))-sensor_offset)
                                else:
                                    quads_obs[q_id][ray_id * 16 + sec_id * 4 + v_sec_id] = min(
                                        quads_obs[q_id][ray_id * 16 + sec_id * 4 + v_sec_id], distance-sensor_offset)
                for o_id in range(len(obst_poses)):
                    o_pos_xy = obst_poses[o_id][:2]

                    # Returns distance and length of the path inside the circle along the shortest distance vector
                    distance, circle_len = is_surface_in_cylinder_view(cur_dir, q_pos_xy, o_pos_xy, obst_radius,
                                                                       fov_angle / 4)
                    if distance is not None:
                        for v_sec_id, v_sec in enumerate(modifications):
                            if deflect_angles[ray_id] + v_sec > 0:
                                project_to = deflect_angles[ray_id] + v_sec - (fov_angle / 8)
                                quads_obs[q_id][ray_id * 16 + sec_id * 4 + v_sec_id] = min(
                                    quads_obs[q_id][ray_id * 16 + sec_id * 4 + v_sec_id], (distance / np.cos(project_to))-sensor_offset)
                            elif deflect_angles[ray_id] + v_sec < 0:
                                project_to = deflect_angles[ray_id] + v_sec + (fov_angle / 8)
                                quads_obs[q_id][ray_id * 16 + sec_id * 4 + v_sec_id] = min(
                                    quads_obs[q_id][ray_id * 16 + sec_id * 4 + v_sec_id], (distance / np.cos(project_to))-sensor_offset)
                            else:
                                quads_obs[q_id][ray_id * 16 + sec_id * 4 + v_sec_id] = min(
                                    quads_obs[q_id][ray_id * 16 + sec_id * 4 + v_sec_id], distance-sensor_offset)

        # quads_obs[q_id][len(scan_angle_arr)] = min(quads_obs[q_id][len(scan_angle_arr)],room_dims[2] - q_z)

    quads_obs = np.clip(quads_obs, a_min=0.0, a_max=scan_max_dist)
    return quads_obs

#@njit
def get_surround_multi_ranger(quad_poses, obst_poses, obst_radius, obst_heights, room_dims, scan_max_dist,
                              quad_rotations):
    """
        quad_poses:     quadrotor positions, only with xy pos
        obst_poses:     obstacle positions, only with xy pos
        quad_vels:      quadrotor velocities, only with xy vel
        obst_radius:    obstacle raidus
    """
    quads_obs = scan_max_dist * np.ones((len(quad_poses), 5))
    scan_angle_arr = np.array([0., np.pi / 2, np.pi, -np.pi / 2])
    fov_angle = (np.pi / 180) * 27
    arm_length = 0.046
    quad_height = 0.04

    for q_id in range(len(quad_poses)):
        q_pos_xy = quad_poses[q_id][:2]
        q_z = quad_poses[q_id][2]
        q_yaw = np.arctan2(quad_rotations[q_id][1, 0], quad_rotations[q_id][0, 0])
        q_pitch = np.arctan2(-quad_rotations[q_id][2, 0],
                             np.sqrt(quad_rotations[q_id][2, 1] ** 2 + quad_rotations[q_id][2, 2] ** 2))
        q_roll = np.arctan2(quad_rotations[q_id][2, 1], quad_rotations[q_id][2, 2])
        z_angle_arr = np.array([q_pitch, -q_roll, -q_pitch, q_roll])
        base_rad = q_yaw
        walls = np.array([[5, q_pos_xy[1]], [-5, q_pos_xy[1]], [q_pos_xy[0], 5], [q_pos_xy[1], -5]])
        for ray_id, rad_shift in enumerate(scan_angle_arr):
            cur_rad = base_rad + rad_shift
            cur_dir = np.array([np.cos(cur_rad), np.sin(cur_rad)])
            z_angle = z_angle_arr[ray_id]
            '''for w_id in range(len(walls)):
                wall_dir = walls[w_id]-q_pos_xy
                if np.dot(wall_dir, cur_dir) > 0:
                    angle = np.arccos(np.dot(wall_dir, cur_dir)/(np.linalg.norm(wall_dir)*np.linalg.norm(cur_dir)))
                    if angle <= fov_angle/2:
                        quads_obs[q_id][ray_id] = min(quads_obs[q_id][ray_id], np.linalg.norm(wall_dir))
                    else:
                        quads_obs[q_id][ray_id] = min(quads_obs[q_id][ray_id], np.linalg.norm(wall_dir)/np.cos(angle-(fov_angle/2)))'''
            for o_id in range(len(obst_poses)):
                o_pos_xy = obst_poses[o_id][:2]
                height = obst_heights[o_id]
                z_diff = np.abs(q_z - obst_poses[o_id][2])
                # Returns distance and length of the path inside the circle along the shortest distance vector
                distance, circle_len = is_surface_in_cylinder_view(cur_dir, q_pos_xy, o_pos_xy, obst_radius, fov_angle)

                if distance is not None:
                    if distance < 0:
                        quads_obs[q_id][len(scan_angle_arr)] = min(quads_obs[q_id][len(scan_angle_arr)],
                                                                   z_diff - height / 2)
                    elif height / 2 >= z_diff:
                        quads_obs[q_id][ray_id] = min(quads_obs[q_id][ray_id], distance)
                    else:
                        # Use distance and height difference to find the angle
                        vertical_angle = np.arctan2(z_diff - height / 2, distance)

                        # Is obstacle above
                        if q_z > obst_poses[o_id][2] + height / 2:
                            if z_angle_arr[ray_id] <= fov_angle / 2:
                                curr_angle = np.abs(z_angle_arr[ray_id] - fov_angle / 2)
                            else:
                                curr_angle = None
                        else:
                            if z_angle_arr[ray_id] > -fov_angle / 2:
                                curr_angle = np.abs(z_angle_arr[ray_id] + fov_angle / 2)
                            else:
                                curr_angle = None

                        if curr_angle is not None:
                            if np.abs(vertical_angle) <= curr_angle:
                                quads_obs[q_id][ray_id] = min(quads_obs[q_id][ray_id],
                                                              (distance ** 2 + (z_diff - height / 2) ** 2) ** 0.5)
                            else:
                                # Checks if FOV angle connects with the base
                                # TODO: error with tan?
                                if (z_diff - height / 2) * (1 / np.tan(curr_angle)) - distance <= circle_len:
                                    quads_obs[q_id][ray_id] = min(quads_obs[q_id][ray_id], (z_diff - height / 2) * (
                                            1 / np.sin(curr_angle)))

            '''for o_quad in range(len(quad_poses)):
                if o_quad == q_id:
                    continue
                opp_quad_xy = quad_poses[o_quad][:2]
                z_diff = np.abs(q_z - quad_poses[o_quad][2])
                distance, circle_len = is_surface_in_cylinder_view(cur_dir, q_pos_xy, opp_quad_xy, arm_length,
                                                                   fov_angle)

                if distance is not None:
                    if distance < 0:
                        quads_obs[q_id][len(scan_angle_arr)] = min(quads_obs[q_id][len(scan_angle_arr)],
                                                                   z_diff - quad_height / 2)
                    else:
                        # Use distance and height difference to find the angle
                        vertical_angle = np.arctan2(z_diff, distance)

                        # Is obstacle above
                        if q_z > quad_poses[o_quad][2] + quad_height / 2:
                            if z_angle_arr[ray_id] <= fov_angle / 2:
                                curr_angle = np.abs(z_angle_arr[ray_id] - fov_angle / 2)
                            else:
                                curr_angle = None
                        else:
                            if z_angle_arr[ray_id] > -fov_angle / 2:
                                curr_angle = np.abs(z_angle_arr[ray_id] + fov_angle / 2)
                            else:
                                curr_angle = None

                        if curr_angle is not None:
                            if np.abs(vertical_angle) <= curr_angle:
                                quads_obs[q_id][ray_id] = min(quads_obs[q_id][ray_id],
                                                              (distance ** 2 + (z_diff - quad_height / 2) ** 2) ** 0.5)
                            else:
                                # Checks if FOV angle connects with the base
                                if (z_diff - quad_height / 2) * (1 / np.tan(curr_angle)) - distance <= circle_len:
                                    quads_obs[q_id][ray_id] = min(quads_obs[q_id][ray_id],
                                                                  (z_diff - quad_height / 2) * (
                                                                          1 / np.sin(curr_angle)))'''

            quads_obs[q_id][len(scan_angle_arr)] = min(quads_obs[q_id][len(scan_angle_arr)], room_dims[2] - q_z)

    quads_obs = np.clip(quads_obs, a_min=0.0, a_max=scan_max_dist)
    return quads_obs


@njit
def get_surround_sdf_multi_ranger(quad_poses, obst_poses, obst_radius, obst_heights, room_dims, scan_max_dist, quad_rotations, resolution):
    scan_angle_arr = np.array([0., np.pi / 2, np.pi, -np.pi / 2])
    quads_sdf_obs = []
    res = np.array([-resolution, 0, resolution])

    distances = get_surround_multi_ranger(quad_poses, obst_poses, obst_radius, obst_heights, room_dims, scan_max_dist, quad_rotations)

    for q_id in range(len(quad_poses)):
        q_yaw = np.arctan2(quad_rotations[q_id][1, 0], quad_rotations[q_id][0, 0])
        base_rad = q_yaw

        directions = np.array([])
        sdf_obs = []
        for ray_id, rad_shift in enumerate(scan_angle_arr):
            cur_rad = base_rad + rad_shift
            cur_dir = np.array([np.cos(cur_rad), np.sin(cur_rad)])
            cur_dir = (cur_dir/np.linalg.norm(cur_dir))*distances[q_id, ray_id]
            directions = np.append(directions, cur_dir)


        for g_x in res:
            for g_y in res:
                min_dist = 4.0
                for direc in directions:
                    min_dist = min(min_dist, np.linalg.norm(direc-np.array([g_x, g_y])))

                sdf_obs.append(min_dist)

        quads_sdf_obs.append(sdf_obs)

    return np.array(quads_sdf_obs)

def get_surround_sdf_multi_ranger_v2(quad_poses, resolution, directions, distances):
    quads_sdf_obs = []
    res = np.array([-resolution, 0, resolution])

    for q_id in range(len(quad_poses)):
        sdf_obs = []

        for g_x in res:
            for g_y in res:
                min_dist = 4.0
                for direc in directions[q_id]:
                    min_dist = min(min_dist, np.linalg.norm(direc - np.array([g_x, g_y])))

                sdf_obs.append(min_dist)

        quads_sdf_obs.append(sdf_obs)

    return np.array(quads_sdf_obs)


def sdf_index(x, y):
    return int(np.floor((x+5) * 10)), int(np.floor((y+5) * 10))

def propogate_sdf(pos, distances, directions, SDF, counts):
    height, width = SDF.shape

    # Unpack the agent position
    agent_x, agent_y = pos
    # 26 to add some buffer
    cone_angle = np.radians(26)
    step_size = 0.1

    for direc, distance in zip(directions, distances):
        x, y = agent_x, agent_y
        start_angle = np.arctan2(direc[1], direc[0]) - np.radians(13)
        cone_radius = min(distance, 4.0)
        if cone_radius == 4.0:
            continue
        cone_radius = min(1.0, distance)
        mag = np.linalg.norm(direc)

        for _ in np.arange(0, cone_radius, 0.1):

            for angle in range(int(np.degrees(start_angle)), int(np.degrees(start_angle+cone_angle))):
                angle_radians = np.radians(angle)

                for r in range(1, int(cone_radius / step_size) + 1):
                    x = r * step_size * np.cos(angle_radians)
                    y = r * step_size * np.sin(angle_radians)
                    ind_x, ind_y = sdf_index(x, y)

                    if 0 <= ind_x < height and 0 <= ind_x < width:
                        if np.linalg.norm(direc - np.array([x, y])) < 4:
                            SDF[ind_x, ind_y] = (counts[ind_x, ind_y]*SDF[ind_x, ind_y] + np.linalg.norm(direc - np.array([x, y])))/(counts[ind_x, ind_y]+1)
                            counts[ind_x, ind_y] += 1
                        if SDF[ind_x, ind_y] < 0:
                            SDF[ind_x, ind_y] = 0

            x += (0.1*direc/mag)[0]
            y += (0.1*direc/mag)[1]

    return SDF, counts

def build_sdf_multi_ranger(quad_poses, obst_poses, obst_radius, obst_heights, room_dims, scan_max_dist, quad_rotations,
                           resolution, SDF, counts):
    scan_angle_arr = np.array([0., np.pi / 2, np.pi, -np.pi / 2])
    distances = get_surround_multi_ranger(quad_poses, obst_poses, obst_radius, obst_heights, room_dims, scan_max_dist,
                                          quad_rotations)
    quads_sdf_obs = np.ones((len(quad_poses), 9))

    directions = []
    for i, q_pos in enumerate(quad_poses):
        q_yaw = np.arctan2(quad_rotations[i][1, 0], quad_rotations[i][0, 0])
        base_rad = q_yaw

        direc = []
        for ray_id, rad_shift in enumerate(scan_angle_arr):
            cur_rad = base_rad + rad_shift
            cur_dir = np.array([np.cos(cur_rad), np.sin(cur_rad)])
            cur_dir = (cur_dir / np.linalg.norm(cur_dir)) * distances[i, ray_id]
            direc.append(cur_dir)
        directions.append(direc)

    sdf_from_rays = get_surround_sdf_multi_ranger_v2(quad_poses, resolution, directions, distances)

    for i, q_pos in enumerate(quad_poses):
        q_pos_x, q_pos_y = q_pos[0], q_pos[1]
        SDF, counts = propogate_sdf(q_pos[:2], distances[i], directions[i], SDF, counts)
        for g_i, g_x in enumerate([q_pos_x - resolution, q_pos_x, q_pos_x + resolution]):
            for g_j, g_y in enumerate([q_pos_y - resolution, q_pos_y, q_pos_y + resolution]):
                ind_x, ind_y = sdf_index(g_x, g_y)
                if SDF[ind_x, ind_y] == 4.0:
                    quads_sdf_obs[i][g_i * 3 + g_j] = sdf_from_rays[i][g_i * 3 + g_j]
                else:
                    quads_sdf_obs[i][g_i * 3 + g_j] = SDF[ind_x, ind_y]

    return SDF, counts, quads_sdf_obs


@njit
def collision_detection(quad_poses, obst_poses, obst_radius, quad_radius):
    quad_num = len(quad_poses)
    collide_threshold = quad_radius + obst_radius
    # Get distance matrix b/w quad and obst
    quad_collisions = -1 * np.ones(quad_num)
    for i, q_pos in enumerate(quad_poses):
        for j, o_pos in enumerate(obst_poses):
            dist = np.linalg.norm(q_pos - o_pos)
            if dist <= collide_threshold:
                #print("Collide")
                quad_collisions[i] = j
                break

    return quad_collisions


@njit
def get_cell_centers(obst_area_length, obst_area_width, grid_size=1.):
    count = 0
    i_len = obst_area_length / grid_size
    j_len = obst_area_width / grid_size
    cell_centers = np.zeros((int(i_len * j_len), 2))
    for i in np.arange(0, obst_area_length, grid_size):
        for j in np.arange(obst_area_width - grid_size, -grid_size, -grid_size):
            cell_centers[count][0] = i + (grid_size / 2) - obst_area_length // 2
            cell_centers[count][1] = j + (grid_size / 2) - obst_area_width // 2
            count += 1

    return cell_centers


if __name__ == "__main__":
    from gym_art.quadrotor_multi.obstacles.test.unit_test import unit_test
    from gym_art.quadrotor_multi.obstacles.test.speed_test import speed_test

    # Unit Test
    unit_test()
    speed_test()
