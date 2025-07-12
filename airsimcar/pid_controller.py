import numpy as np
""" An object of the PIDController can calculate for example the throttle or steering of the car.
    PID controller means proportional–integral–derivative controller.

    For example:

        controller = PIDController(p_gain=1.0)

        delta_time = 0.1
        error = preferred_velocity - car_velocity

        throttle = controller.update(error, delta_time)
"""


class PIDController:
    def __init__(self, p_gain: float = 0.0, i_gain: float = 0.0, d_gain: float = 0.0, i_lb: float = float('-inf'), i_ub: float = float('inf')):
        """ Initialize PIDController

        Args:
            p_gain: The proportional gain of the controller.
            i_gain: The integral gain of the controller.
            d_gain: The derivative gain of the controller.
            i_lb: lower limit (boundary) of integral to avoid integral windup (default: -inf, so no boundary)
            i_ub: upper limit (boundary) of integral to avoid integral windup (default: inf, so no boundary)
        """
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.last_error = 0
        self.sum_error = 0
        self.i_lb = i_lb        # lower limit (boundary) of integral to avoid integral windup
        self.i_ub = i_ub        # upper limit (boundary) of integral to avoid integral windup

    def update(self, error: float, delta_t: float) -> float:
        """ Updates the controller.

        Args:
            error: The difference between the desired value (setpoint) and real value (process variable): 
                   control error = setpoint - process variable.
            delta_t: sampling rate of the control: The time between now and the last update call.

        Returns:
            The correction value (control variable).
        """
        p = self.p_gain * error
        i = self.i_gain * self.sum_error * delta_t
        d = self.d_gain * (self.last_error - error) / delta_t
        self.last_error = error
        self.sum_error += error
        # clip to avoid integral windup
        self.sum_error = np.clip(self.sum_error, self.i_lb, self.i_ub)
        return p + i + d
