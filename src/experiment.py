import numpy as np
import matplotlib.pyplot as plt
from typing import List

from bandits.agent import Bandit


def experiment(probabilities: List[float], num_trials: int, epsilon: float):
    """Execute epsilon-greedy algorithm.

    :param probabilities: List of probabilities of winning (or win rates) associated to each bandit.
    That is, there are as many probabilities as there are bandits.
    :type probabilities: List[float]

    :param num_trials: Number of trials we loop over.
    :type num_trials: int

    :param epsilon: The fraction of randomness for exploration in our program.
    :param epsilon: float
    """
    bandits: list = [Bandit(p) for p in probabilities]

    rewards: np.ndarray = np.zeros(num_trials)
    num_times_explored: int = 0
    num_times_exploited: int = 0
    num_optimal: int = 0

    # The optimal j is the bandit with the highest probability of winning (should not be known in reality)
    optimal_j = np.argmax([b.p for b in bandits])
    print(f"optimal_j: {optimal_j}")

    for i in range(num_trials):
        if np.random.random() < epsilon:  # Explore selecting a random bandit
            num_times_explored += 1
            j = np.random.randint(len(bandits))
        else:  # Exploit pulling the arm of the bandit having the best winning probability estimate
            num_times_exploited += 1
            j = np.argmax([b.p_estimate for b in bandits])

        if j == optimal_j:
            num_optimal += 1

        # Pull the arm of the bandit with the largest sample
        x: int = bandits[j].pull()

        # Update rewards log
        rewards[i] = x

        # Update the the p_estimate attribute for the bandit whose arm we just pulled
        bandits[j].update(x)

    for b in bandits:
        print(f"Mean winning probability estimate: {b.p_estimate}")

    print(f"Total reward earned: {rewards.sum()}")
    print(f"Overall win rate: {rewards.sum() / num_trials}")
    print(f"Number of explorations: {num_times_explored}")
    print(f"Number of exploitations: {num_times_exploited}")
    print(f"Number of times we have selected the optimal bandit: {num_optimal}")

    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(num_trials) + 1)
    plt.plot(win_rates, label='Actual win rate', c='b')
    plt.plot(np.ones(num_trials) * np.max(probabilities), label='Optimal bandit win rate', c='red')
    plt.legend()
    plt.grid(c='k', ls=':')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    bandit_probabilities: List[float] = [0.2, 0.5, 0.75]
    epsilon: float = 0.1
    num_trials: int = 10_000

    experiment(bandit_probabilities, num_trials, epsilon)


