import numpy as np

class LocalMortyEnv:
    """
    Environment: 3 planets with periodic survival probabilities.
    Periods: 10, 20, 200
    """
    def __init__(self):
        self.t = 0  # global time counter (one increment per trip)

        self.periods = [10, 20, 200]
        self.w = [2*np.pi/P for P in self.periods]

        # amplitude and offset (giving rates in [0.3, 0.7])
        self.A = [0.20, 0.25, 0.15]
        self.C = [0.50, 0.55, 0.45]

        # random phase for each planet
        self.phi = [np.random.uniform(0, 2*np.pi) for _ in range(3)]

    def true_rate(self, planet):
        """Return the true survival prob at current t."""
        return self.A[planet] * np.cos(self.w[planet]*self.t + self.phi[planet]) + self.C[planet]

    def send(self, planet, morty_count):
        """
        Send mortys. Allowed sizes = 1,2,3. Returns (successes, trials).
        """
        if morty_count not in [1,2,3]:
            raise ValueError("Morty count must be 1, 2 or 3")

        p = self.true_rate(planet)
        surv = np.random.binomial(morty_count, p)

        self.t += 1
        return surv, morty_count
