import numpy as np


def probability_basics(mu_r, sigma_rr, mu_m, sigma_mm):
    ans_1a = np.zeros((2, 2))
    ans_1b = np.zeros((2, 2))
    ans_1c = np.zeros((2, 1))
    ans_1d = np.zeros((2, 2))
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # note: In this question, you may choose to input the numerical results.
    #       Make sure your answers are accurate to three decimal places.
    
    ans_1a = 4 * sigma_rr + 4 * mu_r * mu_r.T
    ans_1b = 6 * sigma_mm + 6 * mu_m * mu_m.T
    ans_1c = (mu_r * 4 + mu_m * 6) / 10
    ans_1d = (4 * sigma_rr + 6 * sigma_mm + 4 * np.outer(mu_r - ans_1c, mu_r - ans_1c) + 6 * np.outer(mu_m - ans_1c, mu_m - ans_1c))/10
    # or use mp.matmu or @ for matrix multiplication
    
    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    return ans_1a, ans_1b, ans_1c, ans_1d


if __name__ == '__main__':
    # Test your funtions here
    mu_r = np.array([[2], [3]])
    sigma_rr = np.array([[5, 3], [3, 7]])
    mu_m = np.array([[-2], [2]])
    sigma_mm = np.array([[8, 4], [4, 3]])
    ans_1a, ans_1b, ans_1c, ans_1d = probability_basics(mu_r, sigma_rr, mu_m, sigma_mm)
    print('Answer for problem 1a:\n', ans_1a)
    print('Answer for problem 1b:\n', ans_1b)
    print('Answer for problem 1c:\n', ans_1c)
    print('Answer for problem 1d:\n', ans_1d)