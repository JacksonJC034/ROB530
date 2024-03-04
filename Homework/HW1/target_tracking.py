import numpy as np


def target_tracking():
    # known parameters
    mu0 = 20
    sigma0_square = 9
    F = 1
    Q = 4
    H = 1
    R = 1
    z1 = 22
    z2 = 23

    mu2 = 0
    sigma2_square = 0

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # first iteration
    # predicted mean
    mu1_hat = F * mu0
    # predicted covariance
    sigma1_square_hat = F**2 * sigma0_square + Q

    # correction step
    # innovation covariance
    S1 = H**2 * sigma1_square_hat + R
    # filter gain
    K1 = sigma1_square_hat * H / S1
    # corrected mean
    mu1 = mu1_hat + K1 * (z1 - H * mu1_hat)
    # corrected covariance
    sigma1_square = (1 - K1 * H) * sigma1_square_hat

    # second iteration
    # predicted mean
    mu2_hat = F * mu1
    # predicted covariance
    sigma2_square_hat = F**2 * sigma1_square + Q

    # correction step
    # innovation covariance
    S2 = H**2 * sigma2_square_hat + R
    # filter gain
    K2 = sigma2_square_hat * H / S2
    # corrected mean
    mu2 = mu2_hat + K2 * (z2 - H * mu2_hat)
    # corrected covariance
    sigma2_square = (1 - K2 * H) * sigma2_square_hat

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    return (mu2, sigma2_square)


if __name__ == '__main__':
    # Test your funtions here

    print('Answer for Problem 3:\n', target_tracking())
