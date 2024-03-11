import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy, copy

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

class EKF:

    def __init__(self, system, init):
        # EKF Construct an instance of this class
        # Inputs:
        #   system: system and noise models
        #   init:   initial state mean and covariance
        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.Gfun = init.Gfun  # Jocabian of motion model
        self.Vfun = init.Vfun  # Jocabian of motion model
        self.Hfun = init.Hfun  # Jocabian of measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance

        self.state_ = RobotState()

        # init state
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)


    ## Do prediction and set state in RobotState()
    def prediction(self, u, X, P, step):
        if step == 0:
            X = self.state_.getState()
            P = self.state_.getCovariance()
        else:
            X = X
            P = P
        ###############################################################################
        # TODO: Implement the prediction step for EKF                                 #
        # Hint: save your predicted state and cov as X_pred and P_pred                #
        ###############################################################################
        
        # Predict the state using the motion model
        X_pred = self.gfun(X, u)
        
        # Calculate the Jacobian of the motion model
        G = self.Gfun(X, u)
        V = self.Vfun(X, u)
        
        # Predict the state covariance
        M_noise = self.M(u)
        P_pred = G @ P @ G.T + V @ M_noise @ V.T

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)
        return np.copy(X_pred), np.copy(P_pred)


    def correction(self, z, landmarks, X, P):
        # EKF correction step
        #
        # Inputs:
        #   z:  measurement
        X_predict = X
        P_predict = P
        
        # landmark1 = landmarks.getLandmark(z[2].astype(int))
        # landmark2 = landmarks.getLandmark(z[5].astype(int))

        ###############################################################################
        # TODO: Implement the correction step for EKF                                 #
        # Hint: save your corrected state and cov as X and P                          #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################
        
        for i in range(0, len(z), 3):
            landmark = landmarks.getLandmark(z[i+2].astype(int))
            landmark_pos = landmark.getPosition()
            
            # Expected measurement for the current state estimate
            z_hat = self.hfun(landmark_pos[0], landmark_pos[1], X_predict)
            
            # Compute the Jacobian of the measurement model
            H = self.Hfun(landmark_pos[0], landmark_pos[1], X_predict, z_hat)
            
            # Kalman gain
            S = H @ P_predict @ H.T + self.Q
            K = P_predict @ H.T @ np.linalg.inv(S)
            
            # Measurement residual
            z_residual = z[i:i+2] - z_hat
            z_residual[0] = wrap2Pi(z_residual[0])  # Ensure bearing angle is within -pi to pi
            
            # Update state estimate and covariance
            X = X_predict + K @ z_residual
            P = (np.eye(len(P)) - K @ H) @ P_predict

            X_predict = X
            P_predict = P

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setState(X)
        self.state_.setCovariance(P)
        return np.copy(X), np.copy(P)


    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state