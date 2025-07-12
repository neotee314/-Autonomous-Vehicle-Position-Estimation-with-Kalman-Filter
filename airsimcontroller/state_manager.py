import numpy as np
from airsimcar.car import Car
from airsim import Vector3r
from typing import Optional
from airsimcontroller.waypoints import WayPoints
from airsimcontroller.utils import unit_vector, angle, braking_distance
from enum import Enum
""" An object of the StateManger checks the state of the car and decides how the car should behave.

    The car can have the DriveState Drive and Stop.
    In the Drive state the car drives along the given path.
    In the Stop state, the car brakes until it stops.
    The StateManager receives the position and velocity of the car and decides in which state the car is.
    The StateManager also checks whether the car has reached the end of the route or can no longer continue.

"""


class DriveState(Enum):
    Drive = 1
    Decelerate_To_Stop = 2
    Stop = 3


class StateManager:
    """ Manages the State of the car.

        The StageManger checks if the car is in the State Drive or Stop.
        If the car is in the State Drive the method _car.drive is executed.
        If the car is in the State Stop the method _car.stop is executed.

        The car enters the Stop state when it encounters a stop sign.
        When the car is in the Stop state and its velocity is zero, it enters the state Drive.
        The car starts in the state Drive.
    """
    def __init__(self, car: Car, waypoints: WayPoints, delta_time: float, stuck_after: float = 5.0):
        """ Initialize StateManager

        Args:
            car: The car that is controlled by the StateManager
            waypoints: The waypoints of the driven route.
                       A waypoint saves the x and y position and the preferred velocity.
                       waypoints = [[x0, y0, v0],
                                    [x1, y1, v1],
                                    ...
                                    [xn, yn, vn]]
            delta_time: The time between every update step.
            stuck_after: The time after the car is counted as stuck.
        """
        self._car = car
        self._state = DriveState.Drive
        self._waypoints = waypoints
        # index of waypoint in the last call of the method update()
        self._last_waypoint = 0

        self._delta_time = delta_time
        self._stop_time = int(2 / delta_time)
        self._stop_counter = 0
        self._time_for_route = int(120 / delta_time)

        # positions of stop signs along our track. In the future we can add a neural network that detects stop signs
        # and then we can add this net in this state manager to automatically detect whether there is a stop sign in
        # front of the car
        self._stop_signs = np.array([60, 248])
        self._last_stop = None

        self._stuck_after = stuck_after / delta_time
        self._stuck_position = Vector3r(0.0, 0.0, 0.0)
        self._stuck_counter = 0

        self._collided = False
        self._route_driven = False
        self._collisions = 0

    # *** public methods ***

    def update(self, car_position: Vector3r, velocity: Vector3r) -> bool:
        """ Updates the car's behavior after checking the state of the car.

        Args:
            car_position: The x, y and z position of the car in the world coordinate frame.
            velocity: The velocity in the x, y and z direction in the world coordinate frame.

        Returns:
            False if the car is stuck or at the end of the route,
            True if not.
        """
        norm_direction = unit_vector(velocity.to_numpy_array())
        yaw = angle(norm_direction)
        car_position = Vector3r(x_val=car_position.x_val + 1.7 * norm_direction[0],
                                y_val=car_position.y_val + 1.7 * norm_direction[1])
        waypoint_idx = self._waypoints.nearest_waypoint(car_position)
        self._last_waypoint = waypoint_idx
        dist, dist_direction, route_direction = self._waypoints.distance_to_route(waypoint_idx, car_position)

        self._update_state(waypoint_idx)
        self._check_collision()

        if not self._car_can_drive(car_position):
            return False

        if self._state is DriveState.Drive:
            target_speed = self._waypoints.get_speed(waypoint_idx)
            self._car.drive(dist, dist_direction, route_direction, yaw, velocity, target_speed, self._delta_time)
        elif self._state is DriveState.Decelerate_To_Stop:
            self._car.drive(dist, dist_direction, route_direction, yaw, velocity, 0.0, self._delta_time)
        elif self._state is DriveState.Stop:
            self._car.accelerate(route_direction, yaw, velocity, 0.0, self._delta_time)
            self._stop_counter += 1

        self._time_for_route -= 1
        return True

    def get_target_speed(self) -> float:
        return self._waypoints.get_speed(self._last_waypoint)

    def get_collisions(self) -> int:
        return self._collisions

    def get_route_driven(self):
        return self._route_driven

    def get_collided(self):
        return self._collided

    def get_driving_time(self) -> float:
        """ Calculates the driving time.

        Returns:
            the driving time in seconds.
        """
        return 120 - self._time_for_route * self._delta_time

    # *** protected methods ***

    def _update_state(self, car_waypoint_idx: int):
        """ Updates the state of the car.

            The car enters the Stop state when it encounters a stop sign.
            When the car is in the Stop state and its velocity is zero, it enters the state Drive.

        Args:
            car_waypoint_idx: The waypoint index of the closest waypoint to the car.
        """
        if self._state is DriveState.Decelerate_To_Stop:
            if self._car.get_speed() < 0.05:
                self._state = DriveState.Stop
        elif self._state is DriveState.Stop:
            if self._stop_time == self._stop_counter:
                self._stop_counter = 0
                self._state = DriveState.Drive
        elif self._state is DriveState.Drive:
            stop_idx = self._car_at_stop_sign(car_waypoint_idx,
                                              braking_distance(self._waypoints.get_speed(car_waypoint_idx)))
            if stop_idx is not None and self._last_stop != stop_idx:
                self._last_stop = stop_idx
                self._state = DriveState.Decelerate_To_Stop

    def _nearest_sign(self, car_waypoint_idx: int) -> int:
        """ Calculates the nearest sign.

        Args:
            car_waypoint_idx: The waypoint index of the closest waypoint to the car.

        Returns:
            The waypoint index of the nearest sign.
        """
        next_sign = None

        if len(self._stop_signs) > 0:
            diff = self._stop_signs - car_waypoint_idx
            next_sign = self._stop_signs[np.where(diff > 0, diff, np.inf).argmin()]

        return next_sign

    def _car_at_stop_sign(self, car_waypoint_idx: int, max_distance: float) -> Optional[int]:
        """ Checks if the car encounters a stop sign in max_distance.

        Args:
            car_waypoint_idx: The waypoint index of the closest waypoint to the car.
            max_distance: The maximum distance where a stop sign is searched.

        Returns:
            Stop sign index if the car do encounter a stop sign in max_distance,
            None if the car doesn't encounter a stop sign in max_distance.
        """
        next_sign = self._nearest_sign(car_waypoint_idx)

        if car_waypoint_idx < next_sign and self._waypoints.route_length(car_waypoint_idx, next_sign) < max_distance:
            return next_sign
        else:
            return None

    def _car_can_drive(self, car_position: Vector3r) -> bool:
        """ Checks if the car is stuck or at the end of the route.

        Args:
            car_position: The x, y and z position of the car in the world coordinate frame.

        Returns:
            False if the car reaches the end of the route or the car doesn't move after _stuck_after updates,
            True if not.
        """
        if self._state is DriveState.Drive and self._is_stuck():
            return False

        if self._waypoints.car_at_goal(car_position) and self.get_driving_time() > 30 or self._time_for_route <= 0:
            self._route_driven = True
            return False
        else:
            return True

    def _is_stuck(self) -> bool:
        """ Checks if the car is stuck.

            The method counts the updates where the car was in the same position.

        Returns:
            True if the car is the last _stuck_after updates at the same position.
            False if not.
        """
        car_x = self._car.get_kinematics().position.x_val
        car_y = self._car.get_kinematics().position.y_val
        distance = ((car_x - self._stuck_position.x_val) ** 2 + (car_y - self._stuck_position.y_val) ** 2) ** 0.5

        if distance > 0.2:
            self._stuck_position = Vector3r(car_x, car_y, 0)
            self._stuck_counter = 0
        else:
            self._stuck_counter += 1

        return self._stuck_counter >= self._stuck_after

    def _check_collision(self):
        if self._car.has_collided():
            if not self._collided:
                self._collided = True
                self._collisions += 1
        else:
            self._collided = False
