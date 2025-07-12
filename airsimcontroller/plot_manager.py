import os
import csv
import pandas as pd
import numpy as np
from airsim import Vector3r
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import multiprocessing as mp
from typing import Sequence
from airsimcar.car import Car
from airsimcontroller.waypoints import WayPoints
from airsimcontroller.utils import normal_distribution, distance_2d

""" An object of the class PlotManager saves data and visualizes the data in graphs

    With a PlotManager you can save every step your estimated postion and velocity of the car.
    At the end of the simulation you can plot all the saved data to compare the estimation with the real data.
    You can also plot an animation of the data during the simulation.
    
        For example:
        
        # Create a PlotManager and specifies with live_plot=True that the data should be shown during the simulation.
        plot_manager = PlotManager(car, waypoints, delta_t, live_plot=True)
        
        while(True):
            position_estimation = get_position_estimation()
            velocity_estimation = get_velocity_estimation()
            
             # Adds for every step the estimated position and velocity to the manager.
            plot_manager.add_data(position_estimation, velocity_estimation)
            
        # Plots at the end of the simulation the saved data.
        plot_manager.plot()
            
"""


class PlotManager:
    """ The PlotManager visualizes the saved data.

        First you can plot the saved data with the method plot.
        Second you can set live_plot = True in the __init__ constructor.
        Then the data is plotted during the simulation.

        With the function add_data the PlotManager saves the data in lists.
        If live_plot == True then the data will be also saved in a csv File.


    """

    def __init__(self, car: Car, waypoints: WayPoints, delta_t: float, live_plot: bool = False):
        """ Initialize PlotManager.

        Args:
            car: The car object with the real position of the car in the simulation.
            waypoints: The waypoints of the driven route.
                       A waypoint saves the x and y position and the preferred velocity.
                       waypoints = [[x0, y0, v0],
                                    [x1, y1, v1],
                                    ...
                                    [xn, yn, vn]]
            delta_t: The time between every update step.
            live_plot: If True then the data is plotted during the simulation.
        """
        self._car = car
        self.live_plot = live_plot

        self._time = []
        self._delta_t = delta_t
        self._counter = 0

        self._waypoints = waypoints

        # Arrays with the estimated position and velocity of the car
        self._px_estimates = []
        self._py_estimates = []
        self._vx_estimates = []
        self._vy_estimates = []

        # Arrays with the real position and velocity of the car
        self._x_positions = []
        self._y_positions = []
        self._x_velocities = []
        self._y_velocities = []

        # Arrays for the error variance from the Kalman filter
        self._px_variances = []
        self._py_variances = []
        self._vx_variances = []
        self._vy_variances = []
        self._ax_variances = []
        self._ay_variances = []

        if self.live_plot:
            self._csv_filename = "data.csv"
            self._fieldnames = ["x_time", "px_estimate", "py_estimate", "vx_estimate", "vy_estimate",
                                "x_position", "y_position", "x_velocity", "y_velocity", "target_speed"]
            self._create_data_file()
            self.prozess = mp.Process(target=_live_plot, args=(waypoints.get_array(), self._csv_filename))
            self.prozess.start()

    # *** PUBLIC methods ***

    def add_car_state(self, estimated_position: Vector3r, estimated_velocity: Vector3r, target_speed: float):
        """ Adds new estimated position and velocity with the target speed.

            The real position and velocity is queried directly from the _car.
            The time in seconds that the data was measured is fetched from the _counter.

        Args:
            estimated_position: The estimated postion of the car.
            estimated_velocity: The estimated velocity of the car.
            target_speed: The target speed at the car's position in m/s. Defined in waypoints.
        """

        if self.live_plot:
            self._write_data(estimated_position, estimated_velocity, target_speed)

        self._time.append(self._counter)
        self._counter += self._delta_t

        self._px_estimates.append(estimated_position.x_val)
        self._py_estimates.append(estimated_position.y_val)
        self._vx_estimates.append(estimated_velocity.x_val)
        self._vy_estimates.append(estimated_velocity.y_val)
        self._x_positions.append(self._car.get_kinematics().position.x_val)
        self._y_positions.append(self._car.get_kinematics().position.y_val)
        self._x_velocities.append(self._car.get_kinematics().linear_velocity.x_val)
        self._y_velocities.append(self._car.get_kinematics().linear_velocity.y_val)

    def add_p_variance(self, px_variance: float, py_variance: float):
        """ Adds the variance of the estimated x and y position.

        Args:
            px_variance: the variance of the estimated x position.
            py_variance: the variance of the estimated y position.
        """
        self._px_variances.append(px_variance)
        self._py_variances.append(py_variance)

    def add_v_variance(self, vx_variance: float, vy_variance: float):
        """ Adds the variance of the estimated velocity in the x and y direction.

        Args:
            vx_variance: the variance of the estimated velocity in the x direction.
            vy_variance: the variance of the estimated velocity in the y direction.
        """
        self._vx_variances.append(vx_variance)
        self._vy_variances.append(vy_variance)

    def add_a_variance(self, ax_variance: float, ay_variance: float):
        """ Adds the variance of the estimated acceleration in the x and y direction.

        Args:
            ax_variance: the variance of the estimated acceleration in the x direction.
            ay_variance: the variance of the estimated acceleration in the y direction.
        """
        self._ax_variances.append(ax_variance)
        self._ay_variances.append(ay_variance)

    def plot(self, show_plot: bool = True):
        """ Plots the measured data compared to the real data and the driven route.

            Generates 7 graphs:
            The first graph shows the difference of the estimated x position and the real x position.
            The second graph shows the difference of the estimated y position and the real y position.
            The third graph shows the difference of the estimated x velocity and the real x velocity.
            The fourth graph shows the difference of the estimated y velocity and the real y velocity.
            The fifth graph shows the variances of the position estimation from the kalman filter.
            The sixth graph shows the variances of the velocity estimation from the kalman filter.
            The last graph shows the driven route with the real data and the estimated data and
            the default route from the waypoints.

        """
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        fig.set_size_inches(6.4, 3 * 4)
        fig.subplots_adjust(left=0.133,
                            bottom=0.078,
                            right=0.977,
                            top=0.953,
                            wspace=0.2,
                            hspace=0.748)

        self._plot_p_variance()
        self._plot_v_variance()
        self._plot_a_variance()

        self._plot_position(ax1, ax2)
        self._plot_velocity(ax3, ax4)
        # Save figure
        fig.savefig(os.path.join("plots", "state.png"), dpi=300, format="png")

        self._plot_route()
        if show_plot:
            plt.show()

    def print_score(self, collisions: int, driving_time: float):
        """ Prints the driving time, the number of collisions and the score.

        Args:
            collisions: The number of collisions from the car.
            driving_time: The time the car took to drive the route.
        """
        estimate_score = self._calculate_estimate_score()
        drive_score = self._calculate_drive_score(collisions, driving_time)

        print("*********************************************")
        print("  Driving time:   ", round(driving_time, 2), " seconds")
        print("  Collisions:     ", collisions)
        print("  Estimate score: ", round(estimate_score * 100, 2), " / 100 Points")
        print("  Drive score:    ", round(drive_score * 100, 2), " / 100 Points")
        print("*********************************************")

    # *** PROTECTED methods ***

    def _calculate_estimate_score(self) -> float:
        """ Calculates the estimate score after the car has driven the route.

        Returns:
            The calculated estimate score.
        """
        distance = 0.0
        counter = 0

        for i in range(len(self._px_estimates)):
            a = Vector3r(x_val=self._px_estimates[i], y_val=self._py_estimates[i])
            b = Vector3r(x_val=self._x_positions[i], y_val=self._y_positions[i])
            distance += distance_2d(a, b)
            counter += 1

        a = 1 / (2 * np.pi * 0.4 ** 2) ** 0.5
        score = normal_distribution(0.4, 0, distance / counter) / a

        return score

    def _calculate_drive_score(self, collision: int, driving_time: float) -> float:
        """ Calculates the drive score after the car has driven the route.

        Args:
            collision: The number of collisions from the car.
            driving_time: The time the car took to drive the route.

        Returns:
            The calculated drive score.
        """
        distance_error = 0.0
        counter = 0
        collision_penalty = 0.6

        for i in range(40, len(self._x_positions)):
            position = Vector3r(x_val=self._x_positions[i], y_val=self._y_positions[i], z_val=0.0)
            waypoint = self._waypoints.nearest_waypoint(position)
            distance, _, _ = self._waypoints.distance_to_route(waypoint, position)
            tolerance = 0.04
            distance_error += max(0.0, distance - tolerance)
            counter += 1

        distance_error = distance_error / counter
        # 77 seconds is the expected time.
        time_score = (77 - driving_time) / 100

        a = 1 / (2 * np.pi * 0.2 ** 2) ** 0.5
        distance_score = normal_distribution(0.2, 0.0, distance_error) / a

        return max(min(distance_score + time_score, 1.0), 0.0) * collision_penalty ** collision

    def _plot_position(self, x_axis, y_axis):
        """ Plots the difference of the estimated position and the real position.

        Args:
            x_axis: The axis for the difference of the estimated x position and the real x position.
            y_axis: The axis for the difference of the estimated y position and the real y position.
        """
        x_axis.set_title('X Position Estimation Error')
        x_axis.set_xlabel('seconds')
        x_axis.set_ylabel('meter')
        x_axis.plot(self._time, np.subtract(self._x_positions, self._px_estimates), color='#7d1c0e')
        x_axis.grid(True)

        y_axis.set_title('Y Position Estimation Error')
        y_axis.set_xlabel('seconds')
        y_axis.set_ylabel('meter')
        y_axis.plot(self._time, np.subtract(self._y_positions, self._py_estimates), color='#7d1c0e')
        y_axis.grid(True)

    def _plot_velocity(self, x_axis, y_axis):
        """ Plots the difference of the estimated velocity and the real velocity.

        Args:
            x_axis: The axis for the difference of the estimated x velocity and the real x velocity.
            y_axis: The axis for the difference of the estimated y velocity and the real y velocity.
        """
        x_axis.set_title('Car X Velocity')
        x_axis.set_xlabel('seconds')
        x_axis.set_ylabel('m/s')
        x_axis.plot(self._time, self._vx_estimates, color='#dca81d', label='Estimated x velocity')
        x_axis.plot(self._time, self._x_velocities, color='#1032cb', label='Car x velocity')
        x_axis.plot(self._time, np.subtract(self._x_velocities, self._vx_estimates), color='#7d1c0e',
                    label='Difference')
        x_axis.grid(True)
        x_axis.legend()

        y_axis.set_title('Car Y Velocity')
        y_axis.set_xlabel('seconds')
        y_axis.set_ylabel('m/s')
        y_axis.plot(self._time, self._vy_estimates, color='#dca81d', label='Estimated y velocity')
        y_axis.plot(self._time, self._y_velocities, color='#1032cb', label='Car y velocity')
        y_axis.plot(self._time, np.subtract(self._y_velocities, self._vy_estimates), color='#7d1c0e',
                    label='Difference')
        y_axis.grid(True)
        y_axis.legend()

    def _plot_route(self):
        """ Plots the driven route with the real data and the estimated data and the default route from the waypoints.

        """
        route_fig, route_ax = plt.subplots(1)
        route_fig.set_size_inches(9, 9)
        route_fig.tight_layout()

        route_ax.set_title('Route')
        route_ax.set_xlabel('meter')
        route_ax.set_ylabel('meter')
        route_ax.plot(self._x_positions, np.multiply(self._y_positions, -1), color='#1032cb', label='Car route')
        route_ax.plot(self._px_estimates, np.multiply(self._py_estimates, -1), color='#dca81d', label='Estimated route')
        route_ax.plot(self._waypoints.get_array()[:, 0], self._waypoints.get_array()[:, 1] * -1, color='#7d1c0e', label='Route')
        route_ax.legend()
        # Save figure
        route_fig.savefig(os.path.join("plots", "route.png"), dpi=300, format="png")

    def _plot_p_variance(self):
        """ Plots the variance of the estimated x and y position from the Kalman Filter.

            The variance has to be added to the PlotManager with the method add_p_variance()

        """
        if len(self._px_variances) > 0:
            fig, ax = plt.subplots(1)
            fig.subplots_adjust(left=0.104,
                                bottom=0.121,
                                right=0.977,
                                top=0.926,
                                wspace=0.2,
                                hspace=0.511)

            if len(self._px_variances) == len(self._time):
                x = self._time
            else:
                x = list(range(len(self._px_variances)))

            ax.set_title('Variance of the Position Estimate')
            ax.set_xlabel('seconds')
            ax.set_ylabel('m^2')
            ax.plot(x, self._px_variances, color='#0066ff', label='x')
            ax.plot(x, self._py_variances, color='#ff9933', label='y')
            ax.grid(True)
            ax.legend()

            # Save figure
            fig.savefig(os.path.join("plots", "position_variance.png"), dpi=300, format="png")

    def _plot_v_variance(self):
        """ Plots the variance of the estimated velocity in the x and y direction from the Kalman Filter.

            The variance has to be added to the PlotManager with the method add_v_variance()

        """
        if len(self._vx_variances) > 0:
            fig, ax = plt.subplots(1)
            fig.subplots_adjust(left=0.104,
                                bottom=0.121,
                                right=0.977,
                                top=0.926,
                                wspace=0.2,
                                hspace=0.511)

            if len(self._vx_variances) == len(self._time):
                x = self._time
            else:
                x = list(range(len(self._vx_variances)))

            ax.set_title('Variance of the Velocity Estimate')
            ax.set_xlabel('seconds')
            ax.set_ylabel('m^2/s^2')
            ax.plot(x, self._vx_variances, color='#0066ff', label='x')
            ax.plot(x, self._vy_variances, color='#ff9933', label='y')
            ax.grid(True)
            ax.legend()

            # Save figure
            fig.savefig(os.path.join("plots", "velocity_variance.png"), dpi=300, format="png")

    def _plot_a_variance(self):
        """ Plots the variance of the estimated acceleration in the x and y direction from the Kalman Filter.

            The variance has to be added to the PlotManager with the method add_a_variance()

        """
        if len(self._ax_variances) > 0:
            fig, ax = plt.subplots(1)
            fig.subplots_adjust(left=0.104,
                                bottom=0.121,
                                right=0.977,
                                top=0.926,
                                wspace=0.2,
                                hspace=0.511)

            if len(self._ax_variances) == len(self._time):
                x = self._time
            else:
                x = list(range(len(self._ax_variances)))

            ax.set_title('Variance of the Acceleration Estimate')
            ax.set_xlabel('seconds')
            ax.set_ylabel('m^2/s^4')
            ax.plot(x, self._ax_variances, color='#0066ff', label='x')
            ax.plot(x, self._ay_variances, color='#ff9933', label='y')
            ax.grid(True)
            ax.legend()

            # Save figure
            fig.savefig(os.path.join("plots", "acceleration_variance.png"), dpi=300, format="png")

    # *** csv file ***

    def _create_data_file(self):
        """ Creates a csv file for the measured data during the simulation.

            x_time - The time in seconds, where the data was measured.
            px_estimate - The estimated x position of the car.
            py_estimate - The estimated y position of the car.
            vx_estimate - The estimated x velocity of the car.
            vy_estimate - The estimated y velocity of the car.
            x_position - The real x position of the car.
            y_position - The real y position of the car.
            x_velocity - The real x velocity of the car.
            y_velocity - The real y velocity of the car.
        """

        if os.path.exists(self._csv_filename):
            os.remove(self._csv_filename)

        with open(self._csv_filename, "w") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self._fieldnames)
            csv_writer.writeheader()

    def _write_data(self, estimated_position: Vector3r, estimated_velocity: Vector3r, target_speed: float):
        """ Writes the new measured data in the created csv file.

            The real position and velocity is queried directly from the _car.
            The time in seconds that the data was measured is fetched from the _counter.

        Args:
            estimated_position: Estimated position of the car as a 3 dimensional vector.
            estimated_velocity: Estimated velocity of the car as a 3 dimensional vector.
            target_speed: The target speed at the car's position in m/s. Defined in waypoints.
        """
        assert os.path.exists(self._csv_filename)

        with open(self._csv_filename, "a") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self._fieldnames)

            row = {
                self._fieldnames[0]: np.round(self._counter, 1),
                self._fieldnames[1]: np.round(estimated_position.x_val, 2),
                self._fieldnames[2]: np.round(estimated_position.y_val, 2),
                self._fieldnames[3]: np.round(estimated_velocity.x_val, 2),
                self._fieldnames[4]: np.round(estimated_velocity.y_val, 2),
                self._fieldnames[5]: np.round(self._car.get_kinematics().position.x_val, 2),
                self._fieldnames[6]: np.round(self._car.get_kinematics().position.y_val, 2),
                self._fieldnames[7]: np.round(self._car.get_kinematics().linear_velocity.x_val, 2),
                self._fieldnames[8]: np.round(self._car.get_kinematics().linear_velocity.y_val, 2),
                self._fieldnames[9]: np.round(target_speed, 2)
            }

            csv_writer.writerow(row)


def _live_plot(waypoints: np.ndarray, file_name: str):
    """ Starts the animation of the saved data in the csv File 'data.csv'.

    Args:
        waypoints: The waypoints of the driven route.
                   A waypoint saves the x and y position and the preferred velocity.
                   waypoints = [[x0, y0, v0],
                                [x1, y1, v1],
                                ...
                                [xn, yn, vn]]
        file_name: The name of the csv file with saved data.
    """
    fig1, (ax1, ax2) = plt.subplots(2)
    ani = FuncAnimation(plt.gcf(), _animation, fargs=(waypoints, file_name, ax1, ax2), interval=100)
    fig1: plt.Figure
    fig1.subplots_adjust(left=0.101,
                         bottom=0.121,
                         right=0.977,
                         top=0.926,
                         wspace=0.2,
                         hspace=0.511)
    plt.show()


def _animation(i: int, waypoints: np.ndarray, file_name: str, ax1: plt.Axes, ax2: plt.Axes):
    """ Updates the plotted graph with new data saved in the csv file file_name.

    Args:
        i: The number of the frame.
        waypoints: The waypoints of the driven route.
                   A waypoint saves the x and y position and the preferred velocity.
                   waypoints = [[x0, y0, v0],
                                [x1, y1, v1],
                                ...
                                [xn, yn, vn]]
        file_name: The name of the csv file with saved data.
    """
    if os.path.exists(file_name):
        data = pd.read_csv('data.csv')
        x_time = data["x_time"]
        px_estimates = data["px_estimate"]
        py_estimates = data["py_estimate"]
        vx_estimates = data["vx_estimate"]
        vy_estimates = data["vy_estimate"]
        x_positions = data["x_position"]
        y_positions = data["y_position"]
        x_velocities = data["x_velocity"]
        y_velocities = data["y_velocity"]
        target_speed = data["target_speed"]

        if x_time.shape == vy_estimates.shape == y_velocities.shape:
            ax1.cla()
            ax1.set_title('Route')
            ax1.set_xlabel('meter')
            ax1.set_ylabel('meter')
            # x_axis.plot(waypoints[:, 0], waypoints[:, 1] * -1, color='#7d1c0e', label='Route')
            ax1.plot(x_positions, np.multiply(y_positions, -1), color='#1032cb', label='Car route')
            ax1.plot(px_estimates, np.multiply(py_estimates, -1), color='#dca81d', label='Estimated route')
            ax1.legend()

            ax2.cla()
            ax2.set_title('Speed')
            ax2.set_xlabel('seconds')
            ax2.set_ylabel('m/s')
            ax2.plot(x_time, target_speed, color='#7d1c0e', label='Target speed')
            ax2.plot(x_time, np.sqrt(np.add(np.power(y_velocities, 2), np.power(x_velocities, 2))), color='#1032cb',
                     label='Car speed')
            ax2.plot(x_time, np.sqrt(np.add(np.power(vx_estimates, 2), np.power(vy_estimates, 2))), color='#dca81d',
                     label='Estimated speed')
            ax2.legend(loc=2)
