import airsim
from manager import Manager
import numpy as np
from numpy import dot
from numpy.linalg import inv


class KalmanFilter:
    """Locates a car in a 2 dimensional world.

        Attributes:
            delta_seconds: The sample time in seconds.
            x: The estimated state vector.
            C: The output matrix.
            A: The system matrix.
            R: The measurement noise covariance matrix.
            Q: The process noise matrix. (not mentioned in lecture, in your initial solution you can ignore this matrix)
            P: The error covariance matrix.
            K: The kalman gain matrix.
    """

    def __init__(self, x: np.ndarray, delta_seconds: float, gps_stddev_x: float, gps_stddev_y: float, imu_stddev: float, measure):
        """ Initialize KalmanFilter

        Args:
            x (np.ndarray): The initial state estimate
            delta_seconds: The time between every update
            gps_stddev_x: standard deviation of the x-coordinate of the GPS sensor
            gps_stddev_y: standard deviation of the y-coordinate of the GPS sensor
            imu_stddev: standard deviation of the acceleration sensor, that is inside the IMU
        """


        # ****************************************************************
        # 1. Define the state vector of the car
        # ****************************************************************
        # You can define the initial state vector in the main function below.
        # TODO: x = np.array([state1, state2, ...]).T
        # x = (x,y,vx,vy,ax,ay).T
        self.x = x
        

        # ****************************************************************
        # 2. Define the output matrix C
        # ****************************************************************
        # TODO: Determine which states you measure with the sensors.
        # y = C xk + rk
        ### i)Position measuring
        self.C_p = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]]) #C 2*6
        ### ii) Just acceleartion Measuring
        self.C_a = np.array([[0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
        ### iii) Position+ accelleartion
        self.C_p_a = np.array([[1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1]])
       
    
        # ****************************************************************
        # 3. Define the system matrix
        # ****************************************************************
        # TODO: Define, with the motion model of your car, the system matrix.
        self.A = np.array([[1, 0, delta_seconds, 0, 0.5*delta_seconds**2, 0],
                           [0, 1, 0, delta_seconds, 0,  0.5*delta_seconds**2],
                           [0, 0, 1, 0, delta_seconds, 0],
                           [0, 0, 0, 1, 0, delta_seconds],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]]) #A6*6

        # ****************************************************************
        # 4. Define the measurement noise covariance matrix.
        # ****************************************************************
        # The R matrix indicates the inaccuracy of our measurement vector y.
        # TODO: Add the variance for the measurement noise of each sensor.
        self.R_p = np.diag([gps_stddev_x**2, gps_stddev_y**2]) # just for positiclson measuring R2*2
        self.R_a = np.diag([imu_stddev**2, imu_stddev**2]) #just acceleartion R 2*2 
        self.R_p_a = np.diag([gps_stddev_x**2, gps_stddev_y**2, imu_stddev**2, imu_stddev**2])#acceleartion + position  R4*4
    
        #By default we are going to measure just the position
        if measure=='p':
            self.C = self.C_p
            self.R = self.R_p
        elif measure=='a':
            self.R = self.R_a
            self.C = self.C_a
        elif measure =='p_a':
            self.R = self.R_p_a
            self.C = self.C_p_a
        else:
            self.R = self.R_p
            self.C = self.C_p
        # ****************************************************************
        # 5. The process noise matrix
        # ****************************************************************
        # The Q matrix indicates the inaccuracy of our motion model. (this matrix was not mentioned in lecture)
        # for your first implementation of your Kalman filter, you do not have to use this matrix. Later on, you should
        # use it. You can find information about this matrix in the script about the Kalman filter in ilias or anywhere in
        # the Internet.
        # The Q matrix has the same dimension as the P and A matrix.
        # TODO: Test with different values. What influence does the Q-Matrix have on the estimation of the Kalman Filter?
        # TODO: Q = np.diag([variance for state1, variance for state2, ...])
        #self.Q = np.diag([delta_seconds**4/4, delta_seconds**4/4, delta_seconds**2, delta_seconds**2, 1, 1])*imu_stddev**2 #Q6*6
        self.Q = np.diag([delta_seconds**2/2, delta_seconds**2/2, delta_seconds, delta_seconds, 1, 1])*imu_stddev**2 #Q6*6
        #self.Q = np.diag([5,5,1,1,0.6,0.6])*0.1
        
        
        # ****************************************************************
        # 6. The initial error covariance matrix P
        # ****************************************************************
        # TODO: Determine the error of the initial state estimate.
        # TODO: dx = np.array([standard deviation from the first state, ...])
        dx = np.array([[500.0, 500.0,500.0, 500.0, 500.0, 500.0]])

        self.P =np.dot(dx.T, dx) # P6*6= A6*6 std_velx = 0 = std_vely i dont know
        # *****************************************************************
        # 7. The control input matrix (OPTIONAL, as not mentioned in lecture)
        # *****************************************************************
        # TODO: Determine how much the control_input changes each state vector component.
        # You can implement the Kalman filter at the beginning without the control input and the B matrix.
        self.B = np.array([[delta_seconds**2/2 , 0],
                           [0, delta_seconds**2/2],
                           [delta_seconds, 0],
                           [0, delta_seconds],
                           [1, 0],
                           [0, 1]
                           ]) #

        # Kalman matrix = P C.t . ( C P C.t + R)^-1
        self.K = self.P @ self.C.T @ inv(self.C @ self.P @ self.C.T + self.R)
    
    def predict(self, control_input):
        #x_k = A x_(k-1)                                         + B u_(k-1) new predicted state
        x_predicted = (self.A @ self.x).reshape(6,)  #+ (self.B @ control_input).reshape(6,) # 6*6 * 6*1 + 6*2 * 2*1=> 6*1
        #P = A P A.T + Q error covariance
        P_predicted = (self.A @ self.P @ self.A.T ).reshape(6,6) + self.Q.reshape(6,6)  # 4*4 * 4*4 *4*4 + 4*4 => 4*4 
        return x_predicted,P_predicted

    def update(self, y: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        """ Updates the state of the Kalman Filter with new sensor data

        Args:
            y: The measurement vector with the measurements from our sensors [gps_x, gps_y, ...].T
            control_input: The current throttle or brake input from our car in the x and y direction.
                           The control input is a 2 dimensional vector [x, y].T.
                           The values can be between 1 and -1. control_input[0] = 1 means that the car drives with
                           maximum acceleration in the positive direction on the x-axis.
                           Can also be replaced with the last acceleration measurement.
        Returns:
            The estimated state of the car.
        """
   
        # Prediction step
        # TODO: Implement the prediction step. Update x with the motion model and calculate P.
        x_p, P_p = self.predict(control_input) #

        # Correction step
        # TODO: Implement the Correction step. Correct the prediction with the measurements.
       # K = P * C.T* inv(C*P*C.T+R)
        self.K = (P_p @ self.C.T @ inv((self.C @ P_p @ self.C.T) + self.R)) #6*6 * 6*2 * 2*2 => 6*2
        
        #x_k+1 = x_k + K (y - Cx)
        x_updated = x_p + self.K @ (self.C @ y - (self.C @ x_p)) # 6*1 + 6*2*2*1 => 6*1
        #P_k+1 = P_k - K C P A.t
        P_updated = P_p -self.K @ self.C @ P_p# @ self.A.T#  # 6*6 - 6*2 * 2*6*6*6 *6*6=> 6*6
        

        self.x = x_updated
        self.P = P_updated
       
    

        return self.x


if __name__ == '__main__':

    manager = Manager()
    activ = True

   
    kf = KalmanFilter(x = np.array([0, 0, 0, 0, 0, 0]), 
                      delta_seconds = manager.get_time_difference(), 
                      gps_stddev_x=1.0, 
                      gps_stddev_y=1.0,
                      imu_stddev=1.0,
                      measure = 'p_a')# measure = p for position, a for acceleration, p_a for both
  

    while activ:

        # ****************************************************************
        # The localization done only with the GPS sensor
        # Should be replaced with the Kalman filter
        # ****************************************************************
        # TODO: Implement your Kalman Filter here and update the manager with the Kalman Filter estimates of postition and velocity

        # Get GPS data
        gps_position = manager.get_gps_position()
        gps_velocity = manager.get_gps_velocity()
        acc = manager.get_imu_lin_acceleration()
        u = manager.get_control_input()
        u = np.array(u)
    
        # 4a,b,c
        mea_vector = np.array([gps_position.x_val, gps_position.y_val, gps_velocity.x_val, gps_velocity.y_val, acc.x_val, acc.y_val])
        
        p = kf.update(y = mea_vector , control_input = u)

        # Update car state
        activ = manager.update(x = p[0], y = p[1], x_velocity =  p[2], y_velocity =  p[3])
