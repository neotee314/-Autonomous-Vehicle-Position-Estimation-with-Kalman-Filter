import airsim
from airsimcontroller.utils import quaternion_to_mat
from airsim import Vector3r
""" An object of the class IMU represents an Inertial measurement unit installed in a vehicle. 
    It measures the linear acceleration of the vehicle in the x, y and z direction in the body frame in m/s^2.
    The function get_lin_acc transform the acceleration from the body frame to the world frame.
    It also measures the angular velocity of the car in rad/s.
    The sampling rate of the Inertial measurement unit is defined in the file "ImuSimpleParams.hpp" and is fixed to 1/1000 s, 
    see https://github.com/microsoft/AirSim/blob/main/AirLib/include/sensors/imu/ImuSimpleParams.hpp .
    
    # TODO: update documentation
    The measurement uncertainty is modelled by a mean-free Gaussian normal distribution 
    with the standard deviations _stddev_acc/_stddev_ang_vel. 
"""


class IMU:
    def __init__(self,
                 airsim_car_client: airsim.CarClient,
                 car_name: str = "TestCar"):

        self.airsim_car_client = airsim_car_client
        self.car_name = car_name

        # The VelocityRandomWalk and the AngularRandomWalk is defined in settings.json
        self._stddev_acc = 0.1  # VelocityRandomWalk * 9.80665f / 1000 / 0.0316 = 0.062
        self._stddev_ang_vel = 0.00276  # AngularRandomWalk / sqrt(3600.0f) * PI / 180 / 0.0316 = 0.00276

    # *** PUBLIC GET methods ***

    def get_lin_acc(self) -> Vector3r:
        """
        Transforms the 3D-acceleration measured by the accelerometer inside the IMU from the body frame to the world
        frame and returns it.

        Returns: measured acceleration as 3d vector with respect to the world coordinate frame

        """
        imu_data = self._get_imu_data()
        lin_acc = imu_data.linear_acceleration.to_numpy_array().T
        quaternion = imu_data.orientation

        rotation_mat = quaternion_to_mat(quaternion)
        acc = rotation_mat @ lin_acc

        return Vector3r(x_val=acc[0], y_val=acc[1], z_val=acc[2])

    def get_ang_velo(self) -> Vector3r:
        """ Returns: the measured angular velocity around the x, y and z axis in rad/s. """
        imu_data = self._get_imu_data()
        ang_vel_x = imu_data.angular_velocity.x_val

        return ang_vel_x

    def get_stddev_acc(self) -> float:
        return self._stddev_acc

    def get_stddev_ang_vel(self) -> float:
        return self._stddev_ang_vel

    # *** PRIVATE methods ***

    def _get_imu_data(self) -> airsim.ImuData:
        imu_data = self.airsim_car_client.getImuData(imu_name="IMU",
                                                     vehicle_name=self.car_name)

        return imu_data


