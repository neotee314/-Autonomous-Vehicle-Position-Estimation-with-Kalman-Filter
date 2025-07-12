import airsim
import numpy as np
import math

from airsim import Quaternionr
from typing import Sequence


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Calculates the unit vector from the vector.

    Args:
        vector: The given vector.

    Returns:
        The unit vector of the vector."""
    length = np.linalg.norm(vector)
    if length == 0:
        return vector
    else:
        return vector / length


def length_2d(vector: Sequence[float]) -> float:
    """Calculates the length of the 2d vector.

    Args:
        vector: The given vector.

    Returns:
         The length of a 2 dimensional vector.
    """
    return max(0.0, (vector[0] ** 2 + vector[1] ** 2) ** 0.5)


def length_3d(vector: Sequence[float]) -> float:
    """Calculates the length of the 3d vector.

    Args:
        vector: The given vector.

    Returns:
         The length of a 3 dimensional vector.
    """
    return max(0.0, (vector[0] ** 2 + vector[1] ** 2 + vector[2]) ** 0.5)


def distance_2d(point_a: airsim.Vector3r, point_b: airsim.Vector3r) -> float:
    """Calculates the distance between to 2d points.

    Args:
        point_a: The first point.
        point_b: The second point.

    Returns:
        The distance between point_a and point_b.
    """
    return max(0.0, ((point_a.x_val - point_b.x_val) ** 2 + (point_a.y_val - point_b.y_val) ** 2) ** 0.5)


def skew_symmetric(v: np.ndarray):
    """ Skew symmetric form of a 3x1 vector.

    Args:
        v: 3x1 vector [x, y, z].T

    Returns:
        The skew symmetric form for the vector v.
    """
    return np.array(
        [[0, -v[2], v[1]],
         [v[2], 0, -v[0]],
         [-v[1], v[0], 0]], dtype=np.float64)


def quaternion_to_mat(quaternion: Quaternionr) -> np.ndarray:
    """ Creates the rotation matrix from a quaternion.

    Args:
        quaternion: The quaternion from the airsim class Quaternionr.

    Returns:
        The rotation matrix.
    """
    v = np.array([quaternion.x_val, quaternion.y_val, quaternion.z_val]).reshape(3, 1)
    return (quaternion.w_val ** 2 - np.dot(v.T, v)) * np.eye(3) + 2 * np.dot(v, v.T) + 2 * quaternion.w_val * skew_symmetric(v)


def braking_distance(velocity: float):
    """Calculates the breaking distance.

    Args:
        velocity: The velocity in meters per second

    Returns:
        The breaking distance in meters.
    """
    return (velocity * 36 / 100) ** 2


def angle(vector: Sequence[float]) -> float:
    """Calculates the angle of the vector.

    The angle is in range of pi to -pi.

                  - 0.5 pi
                     ^ -y
            -pi      |
             -x <----|----> +x 0°
             pi      |
                     Y +y
                    0.5 pi

    Args:
        vector: The vector.

    Returns:
        the angle of the vector in rad.
    """
    return np.arctan2(vector[1], vector[0])


def normal_distribution(standard_deviation: float, median: float, measurement: float) -> float:
    """ Calculates the normal distribution.

    Args:
        standard_deviation: The standard deviation of the normal distribution.
        median: The median of the normal distribution.
        measurement: The measured value.

    Returns:
        The probability of the measurement.
    """
    factor = 1 / (standard_deviation * (2 * math.pi) ** 0.5)
    exponent = -(measurement - median) ** 2 / (2 * standard_deviation ** 2)
    return factor * math.e ** exponent


def rotation_angle(from_angle: float, to_angle: float) -> float:
    """Calculates the angle from from_angle to to_angle.

    Args:
        from_angle: The start angle.
        to_angle: The target angle.

    Returns:
        The rotation in rad.
    """
    rotation = to_angle - from_angle
    if rotation > np.pi:
        return -2 * np.pi + rotation
    elif rotation < -np.pi:
        return 2 * np.pi + rotation
    else:
        return rotation


def calc_direction(vector: np.ndarray) -> float:
    """Calculates the direction of the vector in the airsim world.

                     90°
                     ^ -y
                     |
        180° -x <----|----> +x 0°
                     |
                     Y +y
                    270°

    Args:
        vector: 2 dimensional vector.

    Returns:
        The direction in degree.

    """
    x = vector[0]
    y = vector[1]
    if x > 0 >= y:
        return math.degrees(math.atan2(abs(y), abs(x)))
    elif x <= 0 > y:
        return math.degrees(math.atan2(abs(x), abs(y))) + 90.0
    elif x < 0 <= y:
        return math.degrees(math.atan2(abs(y), abs(x))) + 180.0
    else:
        return math.degrees(math.atan2(abs(x), abs(y))) + 270.0
