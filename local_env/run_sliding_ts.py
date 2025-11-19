from local_env_ts import LocalMortyEnv
from sliding_ts import SlidingWindowTS
import numpy as np

def main():
    env = LocalMortyEnv()
    ts = SlidingWindowTS(n_arms=3, window=200)

    total_sent = 0
    horizon = 1000

    counts = np.zeros(3)
    rewards = np.zeros(3)

    while total_sent < horizon:
        arm = ts.select_arm()
        successes, trials = env.send(arm, morty_count=1)

        ts.update(arm, successes, trials)

        counts[arm] += 1
        rewards[arm] += successes
        total_sent += 1

        if total_sent % 100 == 0:
            print(f"[{total_sent}/{horizon}] counts={counts} "
                  f"rates={(rewards/(counts+1e-9))}")

    print("\n=== FINAL STATS ===")
    for i in range(3):
        print(f"Planet {i}: chosen {counts[i]} times, "
              f"mean survival = {rewards[i]/counts[i]:.3f}")

if __name__ == "__main__":
    main()
