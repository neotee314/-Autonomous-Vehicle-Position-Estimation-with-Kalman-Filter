import airsim
import numpy as np
import pymap3d as pm
from airsim import Vector3r

""" An object of the class GPS represents a GPS measuring device installed in a vehicle. 
    It measures the position of the vehicle expressed as latitude and longitude. 
    The sampling rate of the GPS measuring device is defined in the settings.json file. 
    The measured position can also be returned in the NED coordinate system as x and y coordinates. 
    The measurement uncertainty is modelled by a mean-free Gaussian normal distribution 
    with the standard deviations stddev_long/stddev_lat. 
    The speed of the vehicle can also be estimated from the measured positions. 
"""


class GPS:
    def __init__(self,
                 airsim_car_client: airsim.CarClient,
                 car_name: str = "TestCar",
                 stddev_long: float = 0.000005,
                 stddev_lat: float = 0.000005):
        """

        Args:
            airsim_car_client: The airsim CarClient to get access to the control of the car.
            car_name: The name of the car defined in settings.json.
            stddev_long: standard deviation of the longitude measurements of the GPS
            stddev_lat: standard deviation of the latitude measurements of the GPS
        """
        self.airsim_car_client = airsim_car_client
        np.random.seed(20)
        self.car_name = car_name
        self.stddev_long = stddev_long
        self.stddev_lat = stddev_lat
        self.measured_positions = []

    # *** PUBLIC GET methods ***

    def get_x_stddev(self):

        x_stddev, _, _ = pm.geodetic2ned(self.stddev_lat, 0, 0, 0, 0, 0)

        return x_stddev

    def get_y_stddev(self):

        _, y_stddev, _ = pm.geodetic2ned(0, self.stddev_long, 0, 0, 0, 0)

        return y_stddev

    def get_longitude(self) -> float:
        gps_data = self._get_gps_data()
        longitude = gps_data.gnss.geo_point.longitude + np.random.normal(0, self.stddev_long)

        return longitude

    def get_latitude(self) -> float:
        gps_data = self._get_gps_data()
        latitude = gps_data.gnss.geo_point.latitude + np.random.normal(0, self.stddev_lat)

        return latitude

    def get_measured_position(self) -> Vector3r:
        """ Converts measured latitude and longitude to the NED coordinate frame.

            Initial default position of the car (see in https://microsoft.github.io/AirSim/settings/).
                "OriginGeopoint": {
                "Latitude": 47.641468,
                "Longitude": -122.140165,
                "Altitude": 122
                }

        Returns:
            The x and y position in the NED coordinate frame in meter.
        """
        lon = self.get_longitude()
        lat = self.get_latitude()
        x, y, z = pm.geodetic2ned(lat, lon, 0, 47.641468, -122.140165, 122)
        self.measured_positions.append([x, y])
        return Vector3r(x_val=x, y_val=y, z_val=z)

    def get_calculated_velocity(self, delta_time: float) -> Vector3r:
        """ Calculates the velocity through the last 5 measured positions

        Args:
            delta_time: The time between every measurement in seconds.

        Returns:
            The calculated velocity in m/s in the x, y and z direction.
        """
        last_positions = self.measured_positions[-5:]
        length = len(last_positions)
        if length > 1:
            v_x = 0.0
            v_y = 0.0
            for i in range(length - 1):
                v_x += last_positions[i+1][0] - last_positions[i][0]
                v_y += last_positions[i+1][1] - last_positions[i][1]

            v_x = v_x / ((length - 1) * delta_time)
            v_y = v_y / ((length - 1) * delta_time)

            return airsim.Vector3r(x_val=v_x, y_val=v_y, z_val=0.0)
        else:
            return airsim.Vector3r(x_val=0.0, y_val=0.0, z_val=0.0)

    # *** PUBLIC methods ***

    def print_data(self):
        gps_data = self._get_gps_data()
        print("gps_data: %s" % gps_data)

    # *** PRIVATE methods ***

    def _get_gps_data(self):
        gps_data = self.airsim_car_client.getGpsData("Gps", self.car_name)
        return gps_data


