"""Mapping module

This module implements an algorithm that create and update an occupancy grid
map using ultrasound sensors range inputs.
"""


import numpy as np
from skimage.draw import line as bresenham
from math import floor

from gym_art.quadrotor_multi.obstacles.utils import esdf, isValid


class Mapping:
    def __init__(self, size, resolution, origin=None):
        """Initialize the parameters dictionary given a map size and resolution

        If origin is None, sets it to be in the middle of the map

        Args:
            size: Size of the square map in meters
            resolution: Number of cells to subdivide 1 meter into
            origin: Cell that represents the origin

        Returns:
            Parameters dictionary:
                "resolution": Resolution
                "size": Size
                "origin": Initial index coordinates (middle of the map if not def)
        """
        self.grid = None
        self.params = {
            "resolution": resolution,
            "size": size,
            "origin": origin,
        }
        if self.params["origin"] is None:  # if origin is not set
            self.params["origin"] = (
                floor(self.params["resolution"]*self.params["size"] / 2),
                floor(self.params["resolution"]*self.params["size"] / 2),
            )


    def create_empty_map(self):
        """Return an empty map of size params.size

        Map is a square matrix of n = params.size * params.resolution
        The x-axis is pointing downward and the y-axis towards the right

        Args:
            params: Dict of parameters

        Returns:
            Square numpy array
        """
        self.grid = np.zeros((
            self.params["size"]*self.params["resolution"],
            self.params["size"]*self.params["resolution"],
        ))


    def discretize(self, position):
        """Discretize the vehicule position

        Given a (x, y) tuple of GLOBAL coordinates, compute the corresponding
        indexes on the grid map.
        The (0, 0) coordinates are put in the middle of the map.


        Args:
            position: Vector ((x, y), n) or tuple (x, y)  of GLOBAL coordinates
            params: Dict of parameters

        Returns:
            2D vector ((x, y), n) of INDEX coordinates

        """
        assert position.shape[0] == 2, \
            "Error: Position vector shape should be (2, n)"
        idx = np.stack((
            ((position[0]) * self.params["resolution"]).astype("int")
            + self.params["origin"][0],
            ((position[1]) * self.params["resolution"]).astype("int")
            + self.params["origin"][1],
        )).astype(np.int16)
        return np.clip(idx, a_max=self.params["resolution"]*self.params["size"]-1, a_min=0)


    def target_cell(self, states, sensor_range, sensor_bearing):
        """Find the (x, y) GLOBAL coordinates of the observed point(s)

        NOTE: target point could be out of range (not in the map)

        Args:
            states (3 x n_particles): One or multiple states of the vehicle
                in the GLOBAL frame
            sensor_range: Observed ranges
            sensor_bearing: Sensor headings

        Returns:
            2D or 3D vector (2 x n_cells x n_particles) of GLOBAL coordinates
        """
        # If is sensor input is a vector, then use vectorization
        if states.ndim == 2:
            n_particles = states.shape[1]
        elif states.ndim == 1:
            n_particles = 1

        i = 0
        while i < n_particles:
            j = 0
            while j < len(sensor_range):
                if sensor_range[j][i] == 2.0:
                    sensor_range = np.delete(sensor_range, j, axis=0)
                    sensor_bearing = np.delete(sensor_bearing, j, axis=0)
                j += 1
            i += 1

        if len(sensor_range) == 0:
            return None

        n_target_cells = len(sensor_bearing)*6
        sensor_range = sensor_range.reshape((-1, 1))
        sensor_bearing = sensor_bearing.reshape((-1, 1))
        states = states.reshape((3, -1))
        x = (sensor_range * np.cos(np.radians(-3) + states[2, :] + sensor_bearing)) + states[0, :]
        y = (-sensor_range * np.sin(np.radians(-3) + states[2, :] + sensor_bearing)) + states[1, :]

        for i in range(-2, 3):
            x = np.append(x, (sensor_range * np.cos(np.radians(i) + states[2, :] + sensor_bearing)) + states[0, :], axis=0)
            y = np.append(y, (-sensor_range * np.sin(np.radians(i) + states[2, :] + sensor_bearing)) + states[1, :], axis=0)

        x = x.reshape((1, n_target_cells, n_particles))
        y = y.reshape((1, n_target_cells, n_particles))
        res = np.concatenate((
            x,
            y,
        ), axis=0)
        return res.squeeze()


    def bresenham_line(self, start, end):
        """Find the cells that should be selected to form a straight line

        Use scikit-image implementation of the Bresenham line algorithm

        Args:
            start: (x, y) INDEX coordinates of the starting point
            end: ((x, y), n) INDEX coordinates of the ending points

        Returns:
            List of (x, y) INDEX coordinates that form the straight lines
        """
        path = list()
        for target in end.T:  # TODO: delete for loop
            tmp = bresenham(start[0], start[1], target[0], target[1])
            path += (list(zip(tmp[0], tmp[1]))[1:-1])  # start + end points removed
        return path


    def update_grid_map(self, ranges, angles, state):
        """Update the grid map given a new set on sensor data

        Args:
            grid: Grid map to be updated
            ranges: Set of range inputs from the sensor
            angles: Angles at which the range points are captured
            state: State estimate (x, y, yaw)
            params: Parameters dictionary

        Returns:
            Updated occupancy grid map
        """
        LOG_ODD_MAX = 100
        LOG_ODD_MIN = -50
        LOG_ODD_OCCU = 1
        LOG_ODD_FREE = 0.3

        # compute the measured position
        targets = self.target_cell(state, ranges, angles)
        if targets is None:
            return
        targets = self.discretize(targets)
        targets = np.unique(targets.T, axis=0).T

        # find the affected cells
        position = self.discretize(state[:2])
        cells = self.bresenham_line(position.reshape(2), targets)

        # update log odds
        self.grid[position[0], position[1]] -= LOG_ODD_FREE
        self.grid[tuple(np.array(cells).T)] -= LOG_ODD_FREE
        self.grid[targets[0], targets[1]] += LOG_ODD_OCCU

        self.grid = np.clip(self.grid, a_max=LOG_ODD_MAX, a_min=LOG_ODD_MIN)



    def build_esdf(self, obstacle_x, obstacle_y):
        self.sdf = esdf(obstacle_x, obstacle_y)

    def get_surround(self, quads_poses):
        quads_sdf = 2*np.ones((len(quads_poses), 9))
        dir = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]])
        for i, pos in enumerate(quads_poses):
            disc_pos = self.discretize(pos)
            for j, d in enumerate(dir):
                if isValid(disc_pos[0]+d[0], disc_pos[1]+d[1]):
                    quads_sdf[i, j] = self.sdf[disc_pos[0]+d[0], disc_pos[1]+d[1]]
        return quads_sdf