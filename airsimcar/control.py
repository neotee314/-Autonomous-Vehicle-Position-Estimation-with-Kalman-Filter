import numpy as np
import airsim
from airsim import Vector3r
from airsimcar.pid_controller import PIDController
from airsimcontroller.utils import rotation_angle
""" An object of the Control class calculates the lateral and longitudinal control of the airsim car.

    The lateral control of the car can be calculated with the PID controller or with the Stanley controller.
    Both controllers calculate the steering control of the car.
    The longitudinal control is calculated with the PID controller.
    The longitudinal PID controller calculates the throttle or brake control of the car.
    With a Control object you can calculate a stop signal or a drive signal.
    When you want to calculate a drive signal the lateral controller tries to follow the path and 
    the longitudinal controller to reach the specified speed.
    When you want to calculate a stop signal the lateral controller tries to follow the path,
    but the longitudinal controller wants to stop the car as fast as possible.
"""


class Control:
    def __init__(self, airsim_car_client: airsim.CarClient, car_controls: airsim.CarControls, car_name: str = "TestCar"):
        """ Initialize Control.

        Args:
            airsim_car_client: The airsim CarClient to get access to the control of the car.
            car_controls: The last CarControls of the airsim car.
            car_name: The name of the airsim car defined in settings.json.
        """
        self._airsim_car_client = airsim_car_client
        self._car_name = car_name
        self._max_steer_angle = 60
        self._car_controls = car_controls
        self._throttle_pid = None           # PID control to control the throttle and break
        # PID control to control steering angle (not used at the moment as steering angle is controlled by Stanley controller)
        self._steer_pid = None

    # *** PUBLIC GET methods ***

    def get_car_controls(self) -> airsim.CarControls:
        return self._car_controls

    # *** PUBLIC methods ***

    def calc_longitudinal_signal(self,
                                 route_direction: float,
                                 car_direction: float,
                                 velocity: Vector3r,
                                 target_speed: float,
                                 delta_time: float):
        """ Calculates only the longitudinal control for the car.

        Args:
            route_direction: The direction of the route in rad.
            car_direction: The drive direction of the car in rad.
            velocity: The velocity in the x, y and z direction of the car in m/s.
            target_speed: The target speed for this route.
            delta_time: The time to the next update of the controls.
        """
        self._pid_acceleration(velocity, target_speed, delta_time, route_direction, car_direction)
        self._airsim_car_client.setCarControls(self._car_controls, self._car_name)

    def calc_drive_signal(self,
                          distance: float,
                          distance_direction: float,
                          route_direction: float,
                          car_direction: float,
                          velocity: Vector3r,
                          target_speed: float,
                          delta_time: float):
        """ Calculates the lateral and longitudinal control to drive along the route.

            The Stanley controller calculates the steering to follow the route.
            The PID controller calculates the brake and throttle of the car.

        Args:
            distance: The distance between car and route in meter.
            distance_direction: The direction from car to route in rad.
            route_direction: The direction of the route in rad.
            car_direction: The drive direction of the car in rad.
            velocity: The velocity in the x, y and z direction of the car in m/s.
            target_speed: The target speed for this route.
            delta_time: The time to the next update of the controls.
        """
        self._pid_acceleration(velocity, target_speed, delta_time, route_direction, car_direction)
        self._stanley_steer(distance, distance_direction, route_direction, car_direction, velocity)
        self._airsim_car_client.setCarControls(self._car_controls, self._car_name)

    # *** PRIVATE methods ***

    def _stanley_steer(self,
                       distance: float,
                       distance_direction: float,
                       route_direction: float,
                       car_direction: float,
                       velocity: Vector3r):
        """ Steers car to route with the stanley lateral controller.

        Args:
            distance: The distance between car and route in meter.
            distance_direction: The direction from the car to the nearest point on the route.
            route_direction: The direction of the route in rad.
            car_direction: The drive direction of the car in rad.
            velocity: The velocity in the x, y and z direction of the car in m/s.
        """

        # calculates angle between drive direction of car and distance_direction
        if rotation_angle(car_direction, distance_direction) < 0:
            distance *= -1

        # calculates angle between drive direction of car and direction of route
        psi = rotation_angle(car_direction, route_direction)

        # Distance error weight
        k = 1.5

        steer = 1.6 * psi + np.arctan(k * distance / (velocity.get_length()+0.01))

        self._set_steer(steer)

    def _pid_steer(self, distance: float, car_direction: float, distance_direction: float, delta_time: float):
        """ Calculates the steering of the car to follow the route.

        Args:
            distance: The distance between car and route in meter.
            car_direction: The driving direction of the car in rad.
            distance_direction: The direction from car to route in rad.
            delta_time: The time to the next update of the controls.
        """
        if self._steer_pid is None:
            self._steer_pid = PIDController(p_gain=np.pi / 4, i_gain=np.pi / 400, d_gain=np.pi / 40)

        if rotation_angle(car_direction, distance_direction) < 0:
            distance *= -1

        self._set_steer(self._steer_pid.update(distance, delta_time))

    def _pid_acceleration(self, velocity: Vector3r, target_speed: float, delta_time: float, route_direction: float,
                          car_direction: float):
        """Accelerates car with PID to target_speed.

        Args:
            velocity: The velocity in the x, y and z direction of the car in m/s.
            target_speed: The target speed for this route.
            delta_time: The time to the next update of the controls.
        """
        start_throttle = 0.25
        v_error = target_speed - velocity.get_length()
        if self._throttle_pid is None:
            # critical p = 1.5
            # perioden länge = 7.7 sec / 6
            self._throttle_pid = PIDController(p_gain=0.675, i_gain=0.014, d_gain=0.0, i_lb=-1.35, i_ub=0.75)

        output = start_throttle + self._throttle_pid.update(v_error, delta_time)

        # calculates angle between drive direction of car and direction of route
        psi = rotation_angle(car_direction, route_direction)

        # TODO: under construction
        # if angle between drive direction and direction of route is too large and velocity is high enough,
        # then do not accelerate further, resp. brake a little bit to slow down
        # --> avoid skidding of car
        """
        if np.abs(psi) > 0.3 and velocity.get_length() > 4.5:
            # print('avoiding skidding', output, velocity.get_length(), psi)
            output = np.fmin(output, -velocity.get_length() / 10.0)
        """

        if output < -0.1:
            self._set_brake(abs(output))
        else:
            self._set_throttle(max(0.0, output))

    def _set_steer(self, steer_in_rad: float):
        """ Sets the steering of the car in the CarControls.

        Args:
            steer_in_rad: The calculated steer in rad.
        """
        input_steer = steer_in_rad * 180 / self._max_steer_angle / np.pi
        steer = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._car_controls.steering = steer

    def _set_throttle(self, input_throttle: float):
        """ Sets the throttle of the car in the CarControls.

        Args:
            input_throttle: The calculated throttle between -1.0 and 1.0.
        """
        throttle = np.clip(input_throttle, -1.0, 1.0)
        self._car_controls.brake = 0.0
        self._car_controls.set_throttle(throttle, throttle >= 0)

    def _set_brake(self, input_brake: float):
        """ Sets the brake of the car in the CarControls.

        Args:
            input_brake: The calculated brake between 0.0 and 1.0.
        """
        brake = np.clip(input_brake, 0.0, 1.0)
        self._car_controls.throttle = 0.0
        self._car_controls.brake = brake


