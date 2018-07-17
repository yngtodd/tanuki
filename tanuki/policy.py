import numpy as np


class Policy(object):
    """
    A policy prescribes an action to be taken based on the memory of the agent.
    """
    def __str__(self):
        return 'generic policy'

    def choose(self, agent):
        return 0


class RandomPolicy(Policy):
    """
    The Random policy randomly selects from all available actions with no
    consideration to which is apparently best. This can be seen as a special
    case of EpsilonGreedy where epsilon = 1 i.e. always explore.
    """
    def __init__(self):
        super(RandomPolicy, self).__init__(1)

    def __str__(self):
        return 'Random policy.'

    def choose(self, agent):
        return np.random.choice(len(agent.value_estimates))


class EpsilonGreedyPolicy(Policy):
    """
    The Epsilon-Greedy Policy will choose a random action with probability
    epsilon and take the best apparent action with probability 1-epsilon. If
    multiple actions are tied for best choice, then a random action from that
    set will be taken.
    """
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __str__(self):
        return '\u03B5-greedy (\u03B5={})'.format(self.epsilon)

    def choose(self, agent):
        if np.random.random() < self.epsilon:
            # Choose randomly
            return np.random.choice(len(agent.value_estimates))
        else:
            action = np.argmax(agent.value_estimates)
            check = np.where(agent.value_estimates == action)[0]
            if len(check) == 0:
                return 0
            else:
                return np.random.choice(check)
