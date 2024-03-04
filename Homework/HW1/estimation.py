import numpy as np

# data from sensor I
z1 = [10.6715, 8.7925, 10.7172, 11.6302, 10.4889, 11.0347, 10.7269, 9.6966, 10.2939, 9.2127];

# data from sensor II
z2 = [10.7107, 9.0823, 9.1449, 9.3524, 10.2602]

# noise variance of sensor I
variance_z1 = 1

# noise variance of sensor II
variance_z2 = 0.64

def Inference(mean, variance, variance_z, z):
    """
    MAP Bayesian inference using Gaussian prior and likelihood
    """

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # sigma_k+1^2 = 1 / (1/sigma_k^2 + 1/sigma_z_k^2)
    if variance == 0:
        # Avoid division by zero in the prior variance
        variance = 1e-10
    updated_variance = 1 / (1 / variance + 1 / variance_z)
    # mu_k+1 = (u_k/sigma_k+1^2 + z_k/sigma_z_k+1^2) * sigma_k+1^2
    updated_mean = (mean / variance + z / variance_z) * updated_variance

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    return updated_mean, updated_variance

def estimation_MAP(Inference_func):
    """
    recursive inference with data from sensor I and sensor II
    """
    # non-informative prior
    mean_1 = 0
    variance_1 = 1000
    mean_2 = 0
    variance_2 = 1000
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################
    
    # run inferece using z1 (use Inference_func instead of Inference)
    for z in z1:
        mean_1, variance_1 = Inference_func(mean_1, variance_1, variance_z1, z)

    # run inferece using z2 (use Inference_func instead of Inference)
    for z in z2:
        mean_2, variance_2 = Inference_func(mean_2, variance_2, variance_z2, z)
    
    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    return mean_1, variance_1, mean_2, variance_2

def KF_update(mean, variance, variance_R, z):
    """
    Kalman Filter Measurement Update
    """
    mean_c = 0
    variance_c = 0
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # compute innovation statistics
    v = z - mean
    S = variance + variance_R
    
    # Kalman gain
    K = variance / S
    
    # corrected mean and variance
    mean_c = K * v + mean
    variance_c = variance * (1 - K)

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    return mean_c, variance_c


def estimation_KF(KF_update_func):
    """
    recursive inference with data from sensor I and sensor II
    """
    # non-informative prior
    mean_1 = 0
    variance_1 = 1000
    mean_2 = 0
    variance_2 = 1000
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################
    
    # run inferece using z1 (use KF_update_func instead of KF_update)
    for z in z1:
        mean_1, variance_1 = KF_update_func(mean_1, variance_1, variance_z1, z)
    
    # run inferece using z2 (use KF_update_func instead of KF_update)
    for z in z2:
        mean_2, variance_2 = KF_update_func(mean_2, variance_2, variance_z2, z)
    
    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    return mean_1, variance_1, mean_2, variance_2