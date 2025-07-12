import numpy as np
import time
from airsim import Vector3r

import airsimcar.car as car
from airsimcontroller.state_manager import StateManager
from airsimcontroller.plot_manager import PlotManager
from airsimcontroller.waypoints import WayPoints
""" An object of the Manager manages the controls of the car and give access to the sensor data.

    To create a manager you can simply run:
        - manager = Manager()
        
    With a Manager you can get the measurements of the IMU (Inertial Measurement Unit) 
    and the GPS (Global Positioning System) of the car in the simulation.
    The IMU measures the acceleration of the car in the x, y, and z direction in m/s^2 and 
    the angular velocity around the x, y, z axis in rad/s.
    You can get the standard deviation and the measurements from the IMU with the following commands:
        - manager.get_imu_accelerator_stddev()
        - manager.get_imu_angular_velocity_stddev()
        - manager.get_imu_lin_acceleration()
        - manager.get_imu_angular_velocity()
        
    The GPS measures the x, y and z position of the car in meter.
    You can get the standard deviation and the measurements from the GPS with the following commands:
        - manager.get_gps_x_stddev()
        - manager.get_gps_y_stddev()
        - manager.get_gps_position()
        
    The GPS also calculates the velocity in the x and y direction from the last 5 position measurements.
    You can get a airsim.Vector3r with the calculated velocity with the following command:
        - velocity = manager.get_gps_velocity()
        - With the command "velocity.x_val" and "velocity.y_val" you get the velocity in the x and y direction.
    
    You can use the manager to pass the calculated variance from the Kalman filter to the PlotManager.
    The PlotManager then plots the variance in a graph at the end of the simulation.
    With the following commands you can pass the variance to the PlotManager:
        - manager.add_position_variance(px_variance, py_variance)
        - manager.add_velocity_variance(vx_variance, vy_variance)
        - manager.add_acceleration_variance(ax_variance, ay_variance)
        
    To run the simulation you need to loop the update method until the method returns false.
    
        For example:
        
        manager = Manager()
        p = manager.get_gps_position()
        v = manager.get_gps_velocity()
        
        while manager.update(p.x_val, p.y_val, v.x_val, v.y_val):
            p = manager.get_gps_position()
            v = manager.get_gps_velocity()

"""


class Manager:
    """ Manages the car and give access to its sensors.

        At the time the manager is created, the client is paused.
        After the position has been updated, the client will continue for delta_time seconds.
        Each time the update method is called, the car is steered along the route based on
        the estimated position and velocity.
        The StateManager is responsible for deciding how the car should behave.
        The PlotManager is responsible for visualizing the measured data.
    """

    def __init__(self, plot_at_end: bool = True, live_plot: bool = True):
        """ Initialize Manager

            The airsim clint will be paused by creating a manager.
            After each call of the method update the clint will continue for 0.1 seconds (_delta_time).
            In addition, when creating a manager, the route that the car should follow is shown as a red line.

        Args:
            plot_at_end: Decides whether the measured data should be plotted after the car has driven.
            live_plot: Decides whether the measured data should be plotted while the car is driving.
        """
        self._waypoints = WayPoints()

        self._car = car.Car()
        self._delta_time = 0.1

        self._car.airsim_car_client.simPause(True)
        self._state_manager = StateManager(self._car, self._waypoints, self._delta_time)
        self._plot_data = plot_at_end
        self._plot_manager = PlotManager(self._car, self._waypoints, self._delta_time, live_plot=live_plot)
        self._waypoints.show_route(self._car.airsim_car_client, 120)
        self._destroy_all_deer()

    # *** public GET methods ***

    def get_time_difference(self) -> float:
        return self._delta_time

    def get_control_input(self) -> np.ndarray:
        """ Get the control input from the brake or throttle command.

            The control input indicates the strength of the acceleration in the x and y direction.
            The strength of the acceleration can range from -1 to 1.
            1 means that the car accelerates in this direction with maximum acceleration.

        Returns:
            the control input as a 2 dimensional vector [x, y].T
        """
        return self._car.get_control_input()

    def get_imu_accelerator_stddev(self) -> float:
        return self._car.my_imu.get_stddev_acc()

    def get_imu_angular_velocity_stddev(self) -> float:
        return self._car.my_imu.get_stddev_ang_vel()

    def get_imu_lin_acceleration(self) -> Vector3r:
        return self._car.get_imu_lin_acc()

    def get_imu_angular_velocity(self) -> Vector3r:
        return self._car.get_imu_ang_velo()

    def get_gps_y_stddev(self) -> float:
        return self._car.get_gps_y_stddev()

    def get_gps_x_stddev(self) -> float:
        return self._car.get_gps_x_stddev()

    def get_gps_position(self) -> Vector3r:
        return self._car.get_gps_position()

    def get_gps_velocity(self) -> Vector3r:
        return self._car.get_gps_velocity(self._delta_time)

    # *** public methods ***

    def update(self, x: float, y: float, x_velocity: float, y_velocity: float) -> bool:
        """ Drives the car along the route with the new estimate.

            The car is controlled based on the position and velocity estimate.
            When you call the method update the airsim client runs for self._delta_time seconds.
            After self._delta_time seconds the client waits for new data.
            If the car is stuck or at the end of the route or the time has expired for the route
            the airsim client and car is reset.

        Args:
            x: The assumed x position of the car.
            y: The assumed y position of the car.
            x_velocity: The assumed linear x velocity of the car.
            y_velocity: The assumed linear y velocity of the car.

        Returns:
            False if the car is stuck, at the end of the route or the route has timed out.
            In the other cases True.
        """
        car_position = Vector3r(x, y, 0)
        car_velocity = Vector3r(x_velocity, y_velocity, 0)
        self._plot_manager.add_car_state(car_position, car_velocity, self._state_manager.get_target_speed())

        if self._state_manager.update(car_position, car_velocity):
            self._car.airsim_car_client.simContinueForTime(self._delta_time)
            while not self._car.airsim_car_client.simIsPause():
                time.sleep(self._delta_time/2)

            return True
        else:
            self._reset()
            self.plot_data()
            if self._state_manager.get_route_driven():
                self._plot_manager.print_score(self._state_manager.get_collisions(), self._state_manager.get_driving_time())
            return False

    def add_position_variance(self, px_variance, py_variance):
        """ Adds the variance of the x and y position from the Kalman Filter to the PlotManager.

            It is optional to add the variance to the PlotManager.
            When you add the variance to the PlotManager, the variance is plotted at the end of the simulation.

        Args:
            px_variance: Variance of the x position from the Kalman Filter.
            py_variance: Variance of the y position from the Kalman Filter.
        """
        self._plot_manager.add_p_variance(px_variance, py_variance)

    def add_velocity_variance(self, vx_variance, vy_variance):
        """ Adds the variance of the velocity in the x and y direction from the Kalman Filter to the PlotManager.

            It is optional to add the variance to the PlotManager.
            When you add the variance to the PlotManager, the variance is plotted at the end of the simulation.

        Args:
            vx_variance: Variance of the x velocity from the Kalman Filter.
            vy_variance: Variance of the y velocity from the Kalman Filter.
        """
        self._plot_manager.add_v_variance(vx_variance, vy_variance)

    def add_acceleration_variance(self, ax_variance, ay_variance):
        """ Adds the variance of the acceleration in the x and y direction from the Kalman Filter to the PlotManager.

            It is optional to add the variance to the PlotManager.
            When you add the variance to the PlotManager, the variance is plotted at the end of the simulation.

        Args:
            ax_variance: Variance of the x acceleration from the Kalman Filter.
            ay_variance: Variance of the y acceleration from the Kalman Filter.
        """
        self._plot_manager.add_a_variance(ax_variance, ay_variance)

    def plot_data(self):
        """ Plots the measured data compared to the real data and the driven route.

            Generates 5 graphs:
            The first graph shows the difference of the estimated x position and the real x position.
            The second graph shows the difference of the estimated y position and the real y position.
            The third graph shows the difference of the estimated x velocity and the real x velocity.
            The fourth graph shows the difference of the estimated y velocity and the real y velocity.
            The last graph shows the driven route with the real data and
            estimated data and the default route from the waypoints.

        """
        self._plot_manager.plot(self._plot_data)

    # *** protected methods ***

    def _reset(self):
        """ Resets the simulation and car position. """
        self._car.reset_airsim()

    def _destroy_all_deer(self):
        """ Destroys every deer object in the village.

            You must restart the simulation to have deer again.
        """
        self._car.airsim_car_client.simDestroyObject('DeerBothBP_12')
        self._car.airsim_car_client.simDestroyObject('DeerBothBP2_19')
        self._car.airsim_car_client.simDestroyObject('DeerBothBP3_43')
        self._car.airsim_car_client.simDestroyObject('DeerBothBP4_108')
        self._car.airsim_car_client.simDestroyObject('DeerBothBP5_223')
        self._car.airsim_car_client.simDestroyObject('RaccoonBP_50')
        self._car.airsim_car_client.simDestroyObject('RaccoonBP2_85')
        self._car.airsim_car_client.simDestroyObject('RaccoonBP3_154')
        self._car.airsim_car_client.simDestroyObject('RaccoonBP4_187')
