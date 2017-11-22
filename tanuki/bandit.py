import numpy as np


class MultiArmedBandit(object):
    """
    A multi-armed bandit aka たぬき
    """
    def __init__(self, k_arms):
        self.k_arms = k_arms
        self.actions_values = np.zeros(k_arms)
        self.optimal = 0

    def reset(self):
        self.actions = np.zeros(self.k_arms)
        self.optimal = 0

    def pull(self):
        return 0, True


class GaussianBandit(MultiArmedBandit):
    """
    Gaussian bandits model the reward of a given arm as normal distribution
    with provided mean and standard deviation.
    """
    def __init__(self, k_arms, mu=0, sigma=1):
        super(GaussianBandit, self).__init__(k_arms)
        self.mu = mu
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.action_values = np.random.normal(self.mu, self.sigma, self.k_arms)
        self.optimal = np.argmax(self.actions_values)

    def pull(self, action):
        return (np.random.normal(self.action_values[action]),
                action == self.optimal)
