import numpy as np


class Environment(self):
    def __init__(self, bandit, agents, label='Multi-Armed Bandit'):
        self.bandit = bandit
        self.agents = agents
        self.label = label

    def reset(self):
        self.bandit.reset()
        for agent in self.agents:
            agent.reset()

    def run(self, trials=100, experiments=1):
        scores = np.zeros((trials, len(self.agents)))
        optimal = np.zeros_like(scores)

        for _ in range(experiments):
            self.reset()
            for t in range(trials):
                for i, agent in enumerate(self.agents):
                    action = agent.choose()
                    reward, is_optimal = self.bandit.pull(action)
                    agent.observe(reward)

                    scores[t, i] += reward
                    if is_optimal:
                        optimal[t, i] += 1

        return scores / experiments, optimal / experiments
