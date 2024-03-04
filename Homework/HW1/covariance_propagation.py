import numpy as np
import matplotlib.pyplot as plt

def calculateEllipseXY(mu, Sigma, k2, N=20):
    """
    input:
    mu     is the [2x1] [x;y] mean vector
    Sigma  is the [2x2] covariance matrix
    k2     is the Chi-squared 2 DOF variable
    N      is the number of points to use (optional, default 20)
    
    output:
    x and y coordinates to draw the ellipse
    """
    # set up angles for drawing the ellipse
    angles = np.linspace(0, 2*np.pi, num=N)
    _circle = np.array([np.cos(angles), np.sin(angles)])

    # make sure it is a numpy array
    mu = np.array(mu)
    Sigma = np.array(Sigma)
        
    # cholesky decomposition
    L = np.linalg.cholesky(Sigma) # Cholesky factor of covariance
    
    # apply the transformation and scale of the covariance
    ellipse = np.sqrt(k2) * L @ _circle

    # shift origin to the mean
    x = mu[0] + ellipse[0, :].T
    y = mu[1] + ellipse[1, :].T

    return x, y

def draw_ellipse(mu, Sigma, k2, colorin='red'):
    """   
    input:
    mu       is the [2x1] [x;y] mean vector
    Sigma    is the [2x2] covariance matrix
    k2       is the Chi-squared 2 DOF variable
    Npoints  number of points to draw the ellipse (default 20)
    colorin  color for plotting ellipses, red for analytical contours, blue for sample contours
    
    --- h = draw_ellipse(mu, Sigma, k2)
    Draws an ellipse centered at mu with covariance Sigma and confidence region k2, i.e.,
    K2 = 1; # 1-sigma
    K2 = 4; # 2-sigma
    K2 = 9; # 3-sigma
    K2 = chi2inv(.50, 2); # 50% probability contour
    """
    Npoints = 20
    
    x, y = calculateEllipseXY(mu, Sigma, k2, Npoints)
    
    if k2 == 9:
        if colorin == 'red':
            plt.plot(x, y, linewidth=1.25, color=colorin, label='analytical contours')
        elif colorin == 'blue':
            plt.plot(x, y, linewidth=1.25, color=colorin, label='sample contours')
    else:
        plt.plot(x, y, linewidth=1.25, color=colorin)

def covariance_propagation():
    # parameter setting
    N = 10000
    mu_sensor = [10, 0]
    sigma_sensor = [0.5, 0.25]

    # output
    result = {'N': N, 'mu_sensor': mu_sensor, 'sigma_sensor': sigma_sensor}

    #############################################################################
    #                                 Problem 4a                                #
    #############################################################################

    # generate point clouds
    r, theta = np.zeros(N), np.zeros(N)
    x, y = np.zeros(N), np.zeros(N)
    random_values1 = np.random.randn(N)
    random_values2 = np.random.randn(N)
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################
    # i) Sensor (r, theta) frame
    # note: use random_values1 and random_values2 here due to auto-grader requirement
    
    r = random_values1 * sigma_sensor[0] + mu_sensor[0]
    theta = random_values2 * sigma_sensor[1] + mu_sensor[1]

    # ii) Cartesian (x,y) coordinate frame
    # note: convert r and theta into Cartesian coordinates

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    result['r'] = r
    result['theta'] = theta
    result['x'] = x
    result['y'] = y

    #############################################################################
    #                                 Problem 4b                                #
    #############################################################################
    Jacobians = np.zeros((2,2))
    cov_cartesian = np.zeros((2,2))
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################
    
    # Implement the Jacobians
    Jacobians = np.array([[1, 0], [0, 10.0]])
    
    # Implement the linearized covariance in cartesian corridinates
    cov_sensor = np.array([[0.25, 0], [0, 0.0625]])
    # cov_new = J*cov_old*J.T
    cov_cartesian = np.matmul(np.matmul(Jacobians, cov_sensor), Jacobians.T)
    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    result['Jacobian'] = Jacobians
    result['cov_cartesian'] = cov_cartesian


    #############################################################################
    #                                 Problem 4c                                #
    #############################################################################
    
    # Sensor frame
    # compute the mean and covariance of samples in the sensor frame 
    # mu_sensor is given
    cov_sensor = np.zeros((2,2))
    mu_sensor_sample = np.zeros((2,))
    cov_sensor_sample = np.zeros((2,2))
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    cov_sensor = np.diag(np.square(sigma_sensor))
    mu_sensor_sample = np.mean(np.stack((result['r'], result['theta']), axis=0), axis=1)
    cov_sensor_sample = np.cov(result['r'], result['theta'])

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    result['mu_sensor'] = mu_sensor
    result['cov_sensor'] = cov_sensor
    result['mu_sensor_sample'] = mu_sensor_sample
    result['cov_sensor_sample'] = cov_sensor_sample

    # Cartesian frame
    # compute the mean and covariance of samples in the Cartesian frame
    mu_cartesian = np.zeros((2,))
    # cov_cartesian is calculated in Problem 4b
    mu_cartesian_sample = np.zeros((2,))
    cov_cartesian_sample = np.zeros((2,2))
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    mu_cartesian = [mu_sensor[0] * np.cos(mu_sensor[1]), mu_sensor[0] * np.sin(mu_sensor[1])]
    mu_cartesian_sample = np.mean(np.stack((result['x'], result['y']), axis=0), axis=1)
    cov_cartesian_sample = np.cov(result['x'], result['y'])

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    result['mu_cartesian'] = mu_cartesian
    # result['cov_cartesian'] = cov_cartesian
    result['mu_cartesian_sample'] = mu_cartesian_sample
    result['cov_cartesian_sample'] = cov_cartesian_sample

    #############################################################################
    #                                 Problem 4d                                #
    #############################################################################
    # counter for samples lie within the contour
    count_sensor = np.zeros((3,1)) # count results of samples in the sensor frame
    count_cartesian = np.zeros((3,1)) # count results of samples in the Cartesian frame

    # Compute the Mahalabobis distance of samples, and count how many samples lie in the contour
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################
    
    # sensor frame counts
    inv_cov_sensor = np.linalg.inv(cov_sensor)
    lenr = len(r)
    for i in range(lenr):
        distribution = np.array([r[i], theta[i]]) - mu_sensor
        distance = np.sqrt(np.matmul(np.matmul(distribution.T, inv_cov_sensor), distribution))
        for j in range(3):
            if distance <= j + 1:
                count_sensor[j] += 1
    
    # cartesian frame counts
    inv_cov_cartesian = np.linalg.inv(cov_cartesian)
    lenx = len(x)
    for i in range(lenx):
        distribution = np.array([x[i], y[i]]) - mu_cartesian
        distance = np.sqrt(np.matmul(np.matmul(distribution.T, inv_cov_cartesian), distribution))
        for j in range(3):
            if distance <= j + 1:
                count_cartesian[j] += 1
    
    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    result['count_sensor'] = count_sensor
    result['count_cartesian'] = count_cartesian


    #############################################################################
    #                                 Problem 4e                                #
    #############################################################################
    result['4e'] = {}
    for rho in [0.1, 0.5, 0.9]: # correlation coefficient, eg. 0.1, 0.5, 0.9
        result['4e'][rho] = {}
        
        # Part A. generate point clouds
        mu_sensor = np.zeros((2,))
        cov_sensor = np.zeros((2,2))
        r, theta = np.zeros(10), np.zeros(10)
        x, y = np.zeros(10), np.zeros(10)
        #############################################################################
        #                    TODO: Implement your code here                         #
        #############################################################################
        # i) Sensor (r, theta) frame 
        
        mu_sensor = [10, 0]
        # cov = [sigma_x^2 rho*sigma_x*sigma_y
        #        rho*sigma_x*sigma_y sigma_y^2]
        cov_sensor = np.array([[sigma_sensor[0]**2, rho * sigma_sensor[0] * sigma_sensor[1]],
                  [rho * sigma_sensor[0] * sigma_sensor[1], sigma_sensor[1]**2]])
        
        # note: do not change the next line due to autograder requirement
        r, theta = np.random.multivariate_normal(mu_sensor, cov_sensor, N).T

        # ii) Cartesian (x,y) coordinate frame
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        #############################################################################
        #                            END OF YOUR CODE                               #
        #############################################################################
        result['4e'][rho]['mu_sensor'] = mu_sensor
        result['4e'][rho]['cov_sensor'] = cov_sensor
        result['4e'][rho]['r'] = r
        result['4e'][rho]['theta'] = theta
        result['4e'][rho]['x'] = x
        result['4e'][rho]['y'] = y

        # Part C. Draw ellipse
        # compute the mean and covariance of samples in the sensor frame 
        mu_sensor_sample = np.zeros((2,))
        cov_sensor_sample = np.zeros((2,2))
        #############################################################################
        #                    TODO: Implement your code here                         #
        #############################################################################

        mu_sensor_sample = np.mean(np.stack((r, theta), axis=0), axis=1)
        cov_sensor_sample = np.cov(r, theta)

        #############################################################################
        #                            END OF YOUR CODE                               #
        #############################################################################
        result['4e'][rho]['mu_sensor_sample'] = mu_sensor_sample
        result['4e'][rho]['cov_sensor_sample'] = cov_sensor_sample

        # compute the mean and covariance of samples in the Cartesian frame
        mu_cartesian = np.zeros((2,))
        cov_cartesian = np.zeros((2,2))
        mu_cartesian_sample = np.zeros((2,))
        cov_cartesian_sample = np.zeros((2,2))
        #############################################################################
        #                    TODO: Implement your code here                         #
        #############################################################################

        mu_cartesian = [mu_sensor[0] * np.cos(mu_sensor[1]), mu_sensor[0] * np.sin(mu_sensor[1])]
        cov_cartesian = np.matmul(np.matmul(Jacobians, cov_sensor), Jacobians.T)
        mu_cartesian_sample = np.mean(np.stack((x, y), axis=1), axis=0)
        cov_cartesian_sample = np.cov(x, y)

        #############################################################################
        #                            END OF YOUR CODE                               #
        #############################################################################
        result['4e'][rho]['mu_cartesian'] = mu_cartesian
        result['4e'][rho]['cov_cartesian'] = cov_cartesian
        result['4e'][rho]['mu_cartesian_sample'] = mu_cartesian_sample
        result['4e'][rho]['cov_cartesian_sample'] = cov_cartesian_sample

    return result

if __name__ == '__main__':
    # Test your funtions here
    result = covariance_propagation()
    print('Answer for Problem 4d:\n', result)