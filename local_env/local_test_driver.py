# local_test_driver.py
import numpy as np
import matplotlib.pyplot as plt
from local_env import LocalMortyEnv

def estimate_cos_coeffs(times, rates, omega):
    """
    Approximation des coefficients a et b
    r(t) = a cos(wt) + b sin(wt) + offset
    """
    X = np.column_stack([
        np.cos(omega * times),
        np.sin(omega * times),
        np.ones_like(times)
    ])
    sol, _, _, _ = np.linalg.lstsq(X, rates, rcond=None)
    a, b, offset = sol
    return a, b, offset


def run_experiment(n_exp=200, send_count=3):
    env = LocalMortyEnv()

    t_log = []
    observed_rate = []
    true_rate = []

    for t in range(n_exp):
        # envoie toujours sur la planète 2 (The Purge)
        r = env.send(planet_id=2, count=send_count)
        rate_est = r["survived"] / r["sent"]

        t_log.append(t)
        observed_rate.append(rate_est)
        true_rate.append(r["true_rate"])

    t_log = np.array(t_log)
    observed_rate = np.array(observed_rate)
    true_rate = np.array(true_rate)

    # estimation cos
    omega = 2*np.pi/200
    a, b, offset = estimate_cos_coeffs(t_log, observed_rate, omega)

    amplitude = np.sqrt(a**2 + b**2)
    phase_est = np.arctan2(-b, a)

    print("===== ESTIMATION =====")
    print(f"a = {a:.4f}")
    print(f"b = {b:.4f}")
    print(f"offset = {offset:.4f}")
    print(f"amplitude = {amplitude:.4f}")
    print(f"phase_est (rad) = {phase_est:.4f}")

    # Plot
    plt.plot(t_log, true_rate, label="True rate")
    plt.scatter(t_log, observed_rate, s=4, label="Observed")
    plt.plot(
        t_log,
        offset + amplitude * np.cos(omega*t_log + phase_est),
        label="Fitted cos"
    )
    plt.legend()
    plt.show()

    return a, b, offset, amplitude, phase_est


if __name__ == "__main__":
    for k in [20,40,60,100,200,400]:
        print("===", k, "samples ===")
        run_experiment(n_exp=k)

