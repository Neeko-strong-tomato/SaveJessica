"""
local_env.py
Simulateur local pour tester les méthodes de prédiction.

Chaque planète a une probabilité de survie déterminée par :
    p(t) = clamp(C + A * cos(ω*t + φ), 0, 1)

t = numéro du voyage (global, pas par planète).
Envoyer 1,2 ou 3 Mortys renvoie {survived_count}.
"""

import numpy as np

class Planet:
    def __init__(self, name, period, amplitude=0.25, offset=0.5, phase=None):
        self.name = name
        self.period = period
        self.A = amplitude
        self.C = offset
        self.phase = phase if phase is not None else np.random.uniform(0, 2*np.pi)
        self.omega = 2*np.pi / period

    def survival_prob(self, t):
        p = self.C + self.A * np.cos(self.omega * t + self.phase)
        return np.clip(p, 0.0, 1.0)

    def send_morties(self, t, n):
        """
        t = voyage ID (global)
        n = 1,2,3
        return number of survivors
        """
        assert n in [1,2,3]
        p = self.survival_prob(t)
        return np.random.binomial(n, p)


class LocalEnvironment:
    def __init__(self):
        self.t = 0
        self.planets = {
            0: Planet("Potit Chat", 10),
            1: Planet("Potit Chien", 20),
            2: Planet("Potite Tortue", 200)
        }

    def send(self, planet_id, n):
        """
        Simule un voyage.
        Avance le temps global t.
        """
        self.t += 1
        return self.planets[planet_id].send_morties(self.t, n)

    def get_true_phase(self):
        """Pour débug : retourne la phase réelle de chaque planète."""
        return {pid: p.phase for pid, p in self.planets.items()}
