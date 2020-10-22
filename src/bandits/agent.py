import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    """
    One-armed bandit (slot machine).
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


if __name__ == '__main__':
    b = Bandit(0.05)

    for _ in range(200):
        x_ = b.pull()
        b.update(x_)
