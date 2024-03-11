
from mimetypes import init
from os import stat
from statistics import mean
from scipy.linalg import block_diag
from copy import deepcopy, copy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

# import InEKF lib
from scipy.linalg import logm, expm


class InEKF:
    # InEKF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):

        self.gfun = system.gfun  # motion model
        # self.hfun = system.hfun  # measurement model
        # self.Gfun = init.Gfun  # Jocabian of motion model
        # self.Vfun = init.Vfun  
        # self.Hfun = init.Hfun  # Jocabian of measurement model
        self.W = system.W # motion noise covariance
        self.V = system.V # measurement noise covariance
        
        self.mu = init.mu
        self.Sigma = init.Sigma

        self.state_ = RobotState()
        X = np.array([self.mu[0,2], self.mu[1,2], np.arctan2(self.mu[1,0], self.mu[0,0])])
        self.state_.setState(X)
        self.state_.setCovariance(init.Sigma)

    
    def prediction(self, u, Sigma, mu, step):
        if step != 0 :
            self.Sigma = Sigma
            self.mu = mu
        state_vector = np.zeros(3)
        state_vector[0] = self.mu[0,2]
        state_vector[1] = self.mu[1,2]
        state_vector[2] = np.arctan2(self.mu[1,0], self.mu[0,0])
        H_prev = self.pose_mat(state_vector)
        state_pred = self.gfun(state_vector, u)
        H_pred = self.pose_mat(state_pred)

        u_se2 = logm(np.linalg.inv(H_prev) @ H_pred)

        ###############################################################################
        # TODO: Propagate mean and covairance (You need to compute adjoint AdjX)      #
        ###############################################################################
        # Propagate mean and covariance
        AdjX = expm(-u_se2)  # Adjoint of X, calculated using the matrix exponential of -u_se2
        self.mu_pred = H_pred
        self.sigma_pred = AdjX @ self.Sigma @ AdjX.T + self.W

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        return np.copy(self.mu_pred), np.copy(self.sigma_pred)

    def propagation(self, u, adjX, mu, Sigma , W):
        self.mu = mu
        self.Sigma = Sigma
        self.W = W
        ###############################################################################
        # TODO: Complete propagation function                                         #
        # Hint: you can save predicted state and cov as self.X_pred and self.P_pred   #
        #       and use them in the correction function                               #
        ###############################################################################
        # Convert input velocities to a twist in Lie algebra se(2)
        xi = np.array([u[0], u[1], u[2]])  # Assuming u = [v, omega, gamma]
        xi_hat = np.array([[0, -xi[2], xi[0]],
                           [xi[2], 0, xi[1]],
                           [0, 0, 0]])  # Twist matrix for se(2)
        
        # Exponential map to get the transformation matrix in SE(2)
        exp_xi_hat = expm(xi_hat)
        
        # Update mean using the exponential map (motion model)
        self.mu_pred = self.mu @ exp_xi_hat
        
        # Adjoint of the predicted mean for covariance update
        # Note: Adjust the adjoint calculation based on your system's specifics
        adjX_pred = np.array([[self.mu_pred[0, 0], self.mu_pred[0, 1], 0],
                              [self.mu_pred[1, 0], self.mu_pred[1, 1], 0],
                              [0, 0, 1]])
        
        # Update covariance with process noise
        self.sigma_pred = adjX_pred @ self.Sigma @ adjX_pred.T + W

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################
        return np.copy(self.mu_pred), np.copy(self.sigma_pred)
        
    def correction(self, Y1, Y2, z, landmarks, mu_pred, sigma_pred):
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))
        self.mu_pred = mu_pred
        self.sigma_pred = sigma_pred
        ###############################################################################
        # TODO: Implement the correction step for InEKF                               #
        # Hint: save your corrected state and cov as X and self.Sigma                 #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################
        # Extract landmark positions
        landmark_x1, landmark_y1 = landmark1.getPosition()
        landmark_x2, landmark_y2 = landmark2.getPosition()

        # Measurement model: Transform landmark positions from global frame to robot's frame
        H_inv = np.linalg.inv(self.mu_pred)
        Y1_pred = H_inv @ np.array([landmark_x1, landmark_y1, 1]).reshape(-1, 1)
        Y2_pred = H_inv @ np.array([landmark_x2, landmark_y2, 1]).reshape(-1, 1)

        # Innovation vectors
        Y1_innov = z[:2].reshape(-1, 1) - Y1_pred[:2]
        Y2_innov = z[3:5].reshape(-1, 1) - Y2_pred[:2]

        # Observation noise covariance matrix for each landmark
        V1 = self.V
        V2 = self.V  # Assuming V is the covariance matrix of observation noise

        # Innovation covariance
        S1 = self.Hfun(Y1_pred) @ self.sigma_pred @ self.Hfun(Y1_pred).T + V1
        S2 = self.Hfun(Y2_pred) @ self.sigma_pred @ self.Hfun(Y2_pred).T + V2

        # Kalman gain
        K1 = self.sigma_pred @ self.Hfun(Y1_pred).T @ np.linalg.inv(S1)
        K2 = self.sigma_pred @ self.Hfun(Y2_pred).T @ np.linalg.inv(S2)

        # Update state estimate
        self.mu = self.mu_pred + K1 @ Y1_innov + K2 @ Y2_innov

        # Update covariance estimate
        I = np.eye(self.sigma_pred.shape[0])
        self.Sigma = (I - K1 @ self.Hfun(Y1_pred) - K2 @ self.Hfun(Y2_pred)) @ self.sigma_pred

        # Extract state vector from mu for compatibility with RobotState
        X = np.array([self.mu[0,2], self.mu[1,2], np.arctan2(self.mu[1,0], self.mu[0,0])])
        

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################
        self.state_.setState(X)
        self.state_.setCovariance(self.Sigma)
        return np.copy(X), np.copy(self.Sigma), np.copy(self.mu)

    def Hfun(self, Y_pred):
        # Assuming Y_pred is the predicted measurement in the robot frame
        # Calculate the Jacobian of the measurement model at Y_pred
        # This is a placeholder example; you need to adjust it based on your actual measurement model
        J = np.array([[1, 0, -Y_pred[1]],
                    [0, 1, Y_pred[0]]])
        return J
    
    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state

    def pose_mat(self, X):
        x = X[0]
        y = X[1]
        h = X[2]
        H = np.array([[np.cos(h),-np.sin(h),x],\
                      [np.sin(h),np.cos(h),y],\
                      [0,0,1]])
        return H
