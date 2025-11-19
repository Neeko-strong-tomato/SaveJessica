# strategy_ekf_sinus.py
import numpy as np
from dataclasses import dataclass
from api_client import SphinxAPIClient


def clamp(x, a, b):
    return max(a, min(b, x))


@dataclass
class EKFState:
    x: np.ndarray      # [A, φ]
    P: np.ndarray      # covariance 2x2


class EKFSinusPlanet:
    """
    EKF pour un sinus p(t) = c0 + A*cos(wt + φ)
    Ici on estime A et φ, la fréquence w est connue.
    """

    def __init__(self, period, c0=0.5):
        self.w = 2 * np.pi / period
        self.c0 = c0
        
        # état initial
        self.state = EKFState(
            x=np.array([0.1, 0.0]),      
            P=np.eye(2) * 0.5         
        )

        # covariance transition
        self.Q = np.eye(2) * 1e-4

        # covariance observation
        self.R = np.array([[0.05]])


    def predict(self):
        self.state.P = self.state.P + self.Q


    def update(self, t, y):
        """EKF update avec observation binaire y."""

        A, phi = self.state.x
        w = self.w

        h = self.c0 + A * np.cos(w * t + phi)
        h = clamp(h, 0.001, 0.999)

        dh_dA = np.cos(w * t + phi)
        dh_dphi = -A * np.sin(w * t + phi)

        H = np.array([[dh_dA, dh_dphi]])   

        z = np.array([[y]]) - np.array([[h]])

        S = H @ self.state.P @ H.T + self.R
        K = self.state.P @ H.T @ np.linalg.inv(S)

        self.state.x = self.state.x + (K @ z).flatten()
        self.state.P = (np.eye(2) - K @ H) @ self.state.P


    def predict_prob(self, t):
        A, phi = self.state.x
        p = self.c0 + A * np.cos(self.w * t + phi)
        return clamp(p, 0, 1)
        

class EKFSinusStrategy:

    def __init__(self, client, explore_steps=80):
        self.client = client
        self.explore_steps = explore_steps
        self.t = {0:0, 1:0, 2:0}

        self.models = {
            0: EKFSinusPlanet(period=10,  c0=0.5),
            1: EKFSinusPlanet(period=20,  c0=0.5),
            2: EKFSinusPlanet(period=200, c0=0.5),
        }

        self.df = []


    def record(self, planet, result):
        self.df.append({
            "planet": planet,
            "t": self.t[planet],
            "survived": result["survived"]
        })


    def explore(self):
        for planet in [0,1,2]:
            print(f"\nExploration of planet {planet}")
            for _ in range(self.explore_steps):
                res = self.client.send_morties(planet, 1)
                y = float(res["survived"])

                self.models[planet].predict()
                self.models[planet].update(self.t[planet], y)

                self.record(planet, res)
                self.t[planet] += 1


    def choose_best(self):
        preds = {
            p: self.models[p].predict_prob(self.t[p])
            for p in [0,1,2]
        }
        return max(preds.items(), key=lambda kv: kv[1])[0]


    def exploit(self, batch=3):
        status = self.client.get_status()
        remaining = status["morties_in_citadel"]

        while remaining > 0:
            best = self.choose_best()
            send = min(batch, remaining)

            res = self.client.send_morties(best, send)
            remaining = res["morties_in_citadel"]

            y = (res["survived"] == send)
            y = float(y)

            self.models[best].predict()
            self.models[best].update(self.t[best], y)

            self.record(best, res)
            self.t[best] += 1

            print(f"Sent {send} to planet {best}, survived={y}")


