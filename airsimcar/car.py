import time
import airsim
import numpy as np

import airsimcar.gps as gps
import airsimcar.imu as imu
import airsimcar.control as control
from airsimcontroller.utils import quaternion_to_mat
from airsim import Vector3r

""" An object of the Car class connects to the airsim CarClient and serves as the interface to the car in airsim.
    The attributes of the car are defined in settings.json.
    The car object manages the car's sensors that the car owns and the controls of the car.
    You have access through the car object to the sensor data and the controls of the car.
    You can control the car with the method drive and stop.
"""


class Car:
    def __init__(self, car_name: str = "TestCar"):
        """ Initialize Car.

        Args:
            car_name: The name of the car defined in settings.json.
        """
        self.airsim_car_client = self._init_airsim(car_name)
        self.car_name = car_name

        self.my_gps = gps.GPS(self.airsim_car_client, car_name)

        self.my_imu = imu.IMU(self.airsim_car_client, car_name)

        self.car_controls = airsim.CarControls()

        self.my_control = control.Control(self.airsim_car_client, self.car_controls, car_name)

    # *** PUBLIC GET methods ***

    def get_gps_velocity(self, delta_time: float) -> Vector3r:
        """ Get velocity of car as 3d vector estimated from recent GPS position measurements

        Args:
            delta_time: The time between every measurement in seconds.

        Returns:
            The estimated velocity in m/s in the x, y and z direction.
        """
        velocity = self.my_gps.get_calculated_velocity(delta_time)

        return velocity

    def get_imu_lin_acc(self) -> Vector3r:
        lin_acc = self.my_imu.get_lin_acc()

        return lin_acc

    def get_imu_ang_velo(self) -> Vector3r:
        ang_vel = self.my_imu.get_ang_velo()

        return ang_vel

    def get_gps_position(self) -> Vector3r:
        return self.my_gps.get_measured_position()

    def get_gps_y_stddev(self) -> float:
        return self.my_gps.get_y_stddev()

    def get_gps_x_stddev(self) -> float:
        return self.my_gps.get_x_stddev()

    def get_control_input(self) -> np.ndarray:
        """ Calculates the control input with the throttle and brake.

        Returns:
            the control input as a 2x1 vector [x, y].T.
        """
        state = self._get_car_state()
        if self.my_control.get_car_controls().brake == 0:
            acceleration = self.my_control.get_car_controls().throttle
        else:
            acceleration = -self.my_control.get_car_controls().brake

        vector = np.array([[acceleration, 0.0, 0.0]]).T
        quaternion = state.kinematics_estimated.orientation
        rotation_mat = quaternion_to_mat(quaternion)
        control_input = rotation_mat @ vector

        return control_input[:2]

    # *** PUBLIC methods ***

    def drive(self,
              distance: float,
              distance_direction: float,
              route_direction: float,
              car_direction: float,
              velocity: Vector3r,
              target_speed: float,
              delta_time: float):
        """Drives the car along the route.

        Args:
            distance: The distance between car and route in meter.
            distance_direction: The direction from car to route in rad.
            route_direction: The direction of travel of the route in rad.
            car_direction: The drive direction of the vehicle in rad.
            velocity: The velocity of the car in the x, y and z direction in m/s.
            target_speed: The target speed in m/s.
            delta_time: The time to the next update in seconds.
        """
        self.my_control.calc_drive_signal(distance, distance_direction, route_direction, car_direction,
                                          velocity, target_speed, delta_time)

    def accelerate(self,
                   route_direction: float,
                   car_direction: float,
                   velocity: Vector3r,
                   target_speed: float,
                   delta_time: float):
        """ Accelerates the car to target speed.

        Args:
            route_direction: The direction of travel of the route in rad.
            car_direction: The drive direction of the vehicle in rad.
            velocity: The velocity of the car in the x, y and z direction in m/s.
            target_speed: The target speed in m/s.
            delta_time: The time to the next update in seconds.
        """
        self.my_control.calc_longitudinal_signal(route_direction, car_direction, velocity, target_speed, delta_time)

    def has_collided(self) -> bool:
        collision = self.airsim_car_client.simGetCollisionInfo(self.car_name)

        return collision.has_collided

    def reset_airsim(self):
        """ Resets the simulation and car position. """
        time.sleep(0.1)
        while self.airsim_car_client.simIsPause():
            self.airsim_car_client.simPause(False)
        self.airsim_car_client.reset()
        self.airsim_car_client.enableApiControl(False)

    # *** PROTECTED methods ***

    @staticmethod
    def _init_airsim(car_name: str = "TestCar"):
        client = airsim.CarClient()
        client.confirmConnection()
        client.enableApiControl(True, car_name)

        return client

    def _get_car_state(self):
        car_state = self.airsim_car_client.getCarState(self.car_name)

        return car_state

    def get_kinematics(self) -> float:
        car_state = self._get_car_state()

        kinematics = car_state.kinematics_estimated

        return kinematics

    def get_speed(self) -> float:
        car_state = self._get_car_state()

        return car_state.speed
