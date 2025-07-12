import numpy as np
import airsim
from airsim import Vector3r
import os
import pandas as pd
from typing import Tuple
from airsimcontroller.utils import length_2d, angle, distance_2d

""" 
An object of the WayPoint class loads all waypoints from the path.csv file and stores them in a list.
The WayPoint class provides several functions to interact with the waypoints.
"""


class WayPoints:

    def __init__(self):
        # load waypoints from path.csv file
        self._waypoints = np.array([])
        self._load_waypoints()

    # *** public methods ***

    def get_speed(self, waypoint_index: int):
        return self._waypoints[waypoint_index, 2]

    def get_array(self):
        return self._waypoints

    def route_length(self, start_waypoint: int, target_waypoint: int) -> float:
        """ Calculates the length of the route from start_waypoint to target_waypoint.

        Args:
            start_waypoint: The waypoint index of the start waypoint.
            target_waypoint: The waypoint index of the target waypoint.

        Returns:
            The length of the route from the start waypoint to the target waypoint.
        """
        length = 0

        if target_waypoint < start_waypoint:
            return np.Infinity

        for i in range(target_waypoint - start_waypoint):
            start = self._waypoints[start_waypoint + i]
            target = self._waypoints[start_waypoint + i + 1]
            length += length_2d([start[0] - target[0], start[1] - target[1]])

        return length

    def distance_to_route(self, near_waypoint: int, car_position: Vector3r) -> Tuple[float, float, float]:
        """ Calculates distance from car to route and the direction.

        Args:
            near_waypoint: The index of the nearest waypoint to the car.
            car_position: The x, y and z position of the car in the world coordinate frame.

        Returns:
            The distance to route, the direction to route in rad, the direction of the route in rad.
        """
        if near_waypoint + 1 < len(self._waypoints):
            point = np.array([self._waypoints[near_waypoint][0], self._waypoints[near_waypoint][1]])
            direction = np.array([self._waypoints[near_waypoint+1][0] - point[0],
                                  self._waypoints[near_waypoint+1][1] - point[1]])
        else:
            point = np.array([self._waypoints[near_waypoint-1][0], self._waypoints[near_waypoint-1][1]])
            direction = np.array([self._waypoints[near_waypoint][0] - point[0],
                                  self._waypoints[near_waypoint][1] - point[1]])

        x_diff = point[0] - car_position.x_val
        y_diff = point[1] - car_position.y_val
        factor = -(x_diff * direction[0] + y_diff * direction[1]) / (direction[0] ** 2 + direction[1] ** 2)
        route_point = np.array([point[0] + direction[0] * factor, point[1] + direction[1] * factor])
        car_to_route = np.array([route_point[0] - car_position.x_val, route_point[1] - car_position.y_val])

        return length_2d(car_to_route), angle(car_to_route), angle(direction)

    def nearest_waypoint(self, car_position: Vector3r) -> int:
        """Calculates the nearest waypoint from the car.

        Args:
            car_position: The x, y and z position of the car in the world coordinate frame.

        Returns:
            The index of the nearest waypoint to the car.
        """
        min_idx = 0
        min_dist = float("inf")

        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                self._waypoints[i][0] - car_position.x_val,
                self._waypoints[i][1] - car_position.y_val]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        return min_idx

    def car_at_goal(self, car_position: Vector3r) -> bool:
        """ Check if the car is at the end of the route.

        Args:
            car_position: The x, y and z position of the car in the world coordinate frame.

        Returns:
            True if the car is at the end of the route,
            False if not.
        """
        target = Vector3r(x_val=self._waypoints[len(self._waypoints) - 2][0],
                          y_val=self._waypoints[len(self._waypoints) - 2][1])
        return distance_2d(target, car_position) < 2

    def show_route(self, client, for_seconds: int):
        """ Shows the route with a red line.

        Args:
            for_seconds: The time how long the route is shown.
        """
        route = []
        for point in self._waypoints:
            route.append(airsim.Vector3r(x_val=point[0], y_val=point[1], z_val=-1))

        client.simPlotLineStrip(points=route, duration=for_seconds)

    # *** protected methods ***

    def _load_waypoints(self):
        """ Loads the waypoints of the route from path.csv

            A waypoint saves the x and y position and the preferred velocity.
            waypoints = [[x0, y0, v0],
                         [x1, y1, v1],
                          ...
                         [xn, yn, vn]]
        """
        filename = os.path.join("airsimcar", "path.csv")
        if os.path.exists(filename):
            data = pd.read_csv(filename)
            self._waypoints = data.to_numpy()


