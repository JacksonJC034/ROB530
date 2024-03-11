
from scipy.linalg import block_diag
from copy import deepcopy, copy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi


class UKF:
    # UKF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):

        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance
        
        self.kappa_g = init.kappa_g
        
        self.state_ = RobotState()
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)
        self.n = 6


    def prediction(self, u, X, P , step):
        # prior belief
        if step == 0:
            mean = self.state_.getState()
            sigma = self.state_.getCovariance()
        else:
            mean = X
            sigma = P
        ###############################################################################
        # TODO: Implement the prediction step for UKF                                 #
        # Hint: save your predicted state and cov as X_pred and P_pred                #
        ###############################################################################
        # Generate sigma points for the current state
        self.sigma_point(mean, sigma, self.kappa_g)
        X_sigma = self.X
        w = self.w

        # Propagate each sigma point through the motion model
        X_sigma_pred = np.zeros((self.n, 2 * self.n + 1))
        for i in range(2 * self.n + 1):
            X_sigma_pred[:, i] = self.gfun(X_sigma[:, i], u).flatten()

        # Calculate predicted state mean
        X_pred = np.zeros(self.n)
        for i in range(2 * self.n + 1):
            X_pred += w[i] * X_sigma_pred[:, i]

        # Calculate predicted state covariance
        P_pred = np.zeros((self.n, self.n))
        for i in range(2 * self.n + 1):
            y = X_sigma_pred[:, i] - X_pred
            y = wrap2Pi(y)  # Assuming y is a vector and this function is vectorized
            P_pred += w[i] * np.outer(y, y)
        P_pred += self.M(u)  # Add motion noise covariance
        

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)
        return np.copy(self.Y), np.copy(self.w), np.copy(X_pred), np.copy(P_pred)

    def correction(self, z, landmarks, Y, w, X, P):

        X_predict = X
        P_predict = P
        self.Y = Y
        self.w = w        
        
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))

        ###############################################################################
        # TODO: Implement the correction step for EKF                                 #
        # Hint: save your corrected state and cov as X and P                          #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################
        # Generate sigma points for the predicted state
        self.sigma_point(X_predict, P_predict, self.kappa_g)
        X_sigma = self.X
        w = self.w

        # Propagate each sigma point through the measurement model
        Z_sigma = np.zeros((2, 2 * self.n + 1))  # Assuming measurement vector length is 2
        for i in range(2 * self.n + 1):
            X_temp = X_sigma[:, i]
            Z_sigma[:, i] = self.hfun(landmark1.getPosition()[0], landmark1.getPosition()[1], X_temp[:3]).flatten()

        # Calculate expected measurement
        z_pred = np.zeros(2)
        for i in range(2 * self.n + 1):
            z_pred += w[i] * Z_sigma[:, i]

        # Calculate innovation covariance
        S = np.zeros((2, 2))
        for i in range(2 * self.n + 1):
            z_diff = Z_sigma[:, i] - z_pred
            S += w[i] * np.outer(z_diff, z_diff)
        S += self.Q  # Add measurement noise covariance

        # Calculate cross covariance
        T = np.zeros((self.n, 2))
        for i in range(2 * self.n + 1):
            x_diff = X_sigma[:, i] - X_pred
            z_diff = Z_sigma[:, i] - z_pred
            T += w[i] * np.outer(x_diff, z_diff)

        # Kalman gain
        K = T.dot(np.linalg.inv(S))

        # Update state estimate and covariance
        z_diff = z[:2] - z_pred  # Assuming z contains actual measurements for the 2 dimensions
        X = X_pred + K.dot(z_diff)
        P = P_pred - K.dot(S).dot(K.T)

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setState(X)
        self.state_.setCovariance(P)
        return np.copy(X), np.copy(P)

    def sigma_point(self, mean, cov, kappa):
        self.n = len(mean) # dim of state
        L = np.sqrt(self.n + kappa) * np.linalg.cholesky(cov)
        Y = mean.repeat(len(mean), axis=1)
        self.X = np.hstack((mean, Y+L, Y-L))
        self.w = np.zeros([2 * self.n + 1, 1])
        self.w[0] = kappa / (self.n + kappa)
        self.w[1:] = 1 / (2 * (self.n + kappa))
        self.w = self.w.reshape(-1)

    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state