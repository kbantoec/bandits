import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union


class OneArmedBandit:
    """One-armed bandit (slot machine).

    :param p: Represents the true win rate for the bandit.
    :type p: float
    """
    def __init__(self, p: float):
        self.p: float = p
        self.p_estimate: float = 0.0
        self.N: int = 0

    def pull(self) -> int:
        """Draw a 1 with probability p."""
        return int(np.random.random() < self.p)

    def update(self, x: int) -> None:
        """Update the p_estimate attribute.

        :param x: sample value.
        :type x: int
        """
        self.N += 1
        self.p_estimate += 1 / self.N * (x - self.p_estimate)


class Experiment:
    def __init__(self, probabilities: List[float], num_trials: int, epsilon: float):
        self.probabilities: list = probabilities
        self.num_trials: int = num_trials
        self.epsilon: float = epsilon

        self.bandits: list = [OneArmedBandit(p) for p in probabilities]
        self.rewards: np.ndarray = np.zeros(num_trials)
        self.optimal_j = np.argmax([b.p for b in self.bandits])

        self.num_times_explored: int = 0
        self.num_times_exploited: int = 0
        self.num_optimal: int = 0

        self.__experimented: bool = False

    def epsilon_greedy(self) -> int:
        # Explore
        if np.random.random() < self.epsilon:
            self.num_times_explored += 1
            j = np.random.randint(len(self.bandits))
        # Exploit
        else:
            self.num_times_exploited += 1
            j = np.argmax([b.p_estimate for b in self.bandits])

        return int(j)

    def run(self) -> None:
        for i in range(self.num_trials):
            j = self.epsilon_greedy()

            # Is the j-th bandit the optimal bandit?
            if j == self.optimal_j:
                self.num_optimal += 1

            # Pull the j-th bandit
            x: int = self.bandits[j].pull()
            # Record the reward earned
            self.rewards[i] = x
            # Update the the p_estimate attribute of the j-th bandit
            self.bandits[j].update(x)

            self.__experimented = True

    def win_rates(self) -> np.ndarray:
        if not self.__experimented:
            raise Exception("You must run the experiment first!")
        cumulative_rewards = np.cumsum(self.rewards)
        return cumulative_rewards / (np.arange(self.num_trials) + 1)

    def plot_win_rates(self, img_name: Union[str, None] = None) -> None:
        if not self.__experimented:
            raise Exception("You must run the experiment first!")
        plt.plot(self.win_rates(), label='Actual win rate', c='b')
        plt.plot(np.ones(self.num_trials) * np.max(self.probabilities), label='Optimal bandit win rate', c='red')
        plt.xscale('log')
        plt.legend()
        plt.grid(c='k', ls=':')
        plt.tight_layout()
        if img_name is not None:
            plt.savefig(f'{img_name}.png', dpi=600, transparent=True)
        plt.show()

    def show_results(self) -> None:
        print(f"Optimal bandit is at index: {self.optimal_j}")

        for i, b in enumerate(self.bandits):
            print(f"Mean winning probability estimate for bandit at index {i}: {b.p_estimate:.3f}")

        print(f"Total reward earned: {self.rewards.sum()}")
        print(f"Overall win rate: {self.rewards.sum() / self.num_trials}")
        print(f"Number of explorations: {self.num_times_explored}")
        print(f"Number of exploitations: {self.num_times_exploited}")
        try:
            ratio = self.num_times_explored / self.num_times_exploited
            print(f"Exploration-Exploitation ratio: {ratio:.3f}")
        except ZeroDivisionError:
            print("Exploration-Exploitation ratio: N/A")
        print(f"Number of times we have selected the optimal bandit: {self.num_optimal}")