import numpy as np
import matplotlib.pyplot as plt

# colors
green = np.array([0.2980, 0.6, 0])
darkblue = np.array([0, 0.2, 0.4])
VermillionRed = np.array([156, 31, 46]) / 255


def plot_fuction(belief, prediction, posterior_belief):
    """
    plot prior belief, prediction after action, and posterior belief after measurement
    """
    fig = plt.figure()

    # plot prior belief
    ax1 = plt.subplot(311)
    plt.bar(np.arange(0, 10), belief.reshape(-1), color=darkblue)
    plt.title(r'Prior Belief')
    plt.ylim(0, 1)
    plt.ylabel(r'$bel(x_{t-1})$')

    # plot likelihood
    ax2 = plt.subplot(312)
    plt.bar(np.arange(0, 10), prediction.reshape(-1), color=green)
    plt.title(r'Prediction After Action')
    plt.ylim(0, 1)
    plt.ylabel(r'$\overline{bel(x_t})}$')

    # plot posterior belief
    ax3 = plt.subplot(313)
    plt.bar(np.arange(0, 10), posterior_belief.reshape(-1), color=VermillionRed)
    plt.title(r'Posterior Belief After Measurement')
    plt.ylim(0, 1)
    plt.ylabel(r'$bel(x_t})$')

    plt.show()


def bayes_filter_b():
    """
    Follow steps of Bayes filter.  
    You can use the plot_fuction() above to help you check the belief in each step.
    Please print out the final answer.
    """

    # Initialize belief uniformly
    belief = 0.1 * np.ones(10)

    posterior_belief = np.zeros(10)
    #############################################################################
    #                    TODO: Implement you code here                          #
    #############################################################################
    
    p_LM_withinLM = 0.8
    p_LM_withoutLM = 0.4
    landmark_places = [0, 3, 6]

    # Shift belief function
    def shift(bel, shift):
        return np.roll(bel, shift)

    # Update belief function
    def update_sensor_belief(bel, sees_landmark):
        sensor_model = np.array([p_LM_withinLM if place in landmark_places 
                                else p_LM_withoutLM for place in range(10)])
        if not sees_landmark:
            sensor_model = 1 - sensor_model
        return bel * sensor_model

    # First observation
    belief = update_sensor_belief(belief, True)
    belief /= np.sum(belief)  # Normalize

    # First movement (+3)
    belief = shift(belief, 3)

    # Second observation
    belief = update_sensor_belief(belief, True)
    belief /= np.sum(belief)  # Normalize

    # Second movement (+4)
    belief = shift(belief, 4)

    # Final observation
    belief = update_sensor_belief(belief, False)
    belief /= np.sum(belief)  # Normalize
    posterior_belief = belief
    
    print("belief:\n", belief)
    
    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    return posterior_belief


if __name__ == '__main__':
    # Test your funtions here
    belief = bayes_filter_b()
    print('Answer for Problem 2b:')
    for i in range(10):
        print("%6d %18.3f\n" % (i, belief[i]))
    plt.bar(np.arange(0, 10), belief)
