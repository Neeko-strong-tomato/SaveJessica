import numpy as np
from collections import deque

class SlidingWindowTS:
    """
    Thompson sampling with sliding window of last k observations.
    Each arm maintains a deque of (success, failure) pairs.
    """
    def __init__(self, n_arms, window=200, prior_a=1, prior_b=1):
        self.n = n_arms
        self.window = window
        self.prior_a = prior_a
        self.prior_b = prior_b

        # For each arm, store last observations
        self.buffers = [deque(maxlen=self.window) for _ in range(self.n)]

    def select_arm(self):
        """Sample from Beta posterior of each arm."""
        samples = []
        for i in range(self.n):
            successes = sum([s for (s,f) in self.buffers[i]])
            failures  = sum([f for (s,f) in self.buffers[i]])
            alpha = self.prior_a + successes
            beta  = self.prior_b + failures
            samples.append(np.random.beta(alpha, beta))

        return np.argmax(samples)

    def update(self, arm, successes, trials):
        failures = trials - successes
        self.buffers[arm].append((successes, failures))
