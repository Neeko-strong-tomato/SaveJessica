# analyze_measurements.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sinus_tools import fit_sinusoidal_model

CSV_FILE = "planet_measurements.csv"

COLUMN_NAMES = [
    "planet",
    "run",
    "trip_index",
    "survived",
    "morties_sent",
    "morties_in_citadel",
    "morties_lost",
    "rewards",
    "steps_taken"
]

def load_csv():
    with open(CSV_FILE, "r") as f:
        first_line = f.readline()

    if "planet" not in first_line:
        print(">>> Adding header to CSV automatically")
        df = pd.read_csv(CSV_FILE, names=COLUMN_NAMES)
    else:
        df = pd.read_csv(CSV_FILE)

    if df["survived"].dtype == object:
        df["survived"] = df["survived"].astype(str).map(
            {"True": 1, "False": 0, "1": 1, "0": 0}
        ).astype(int)

    return df


def analyze_planet(df, planet):
    dfp = df[df.planet == planet]

    runs = sorted(dfp.run.unique())

    if len(runs) == 0:
        print(f"No data for planet {planet}")
        return

    all_runs = []
    for r in runs:
        s = dfp[dfp.run == r].sort_values("trip_index").survived.to_numpy()
        all_runs.append(s)

    all_runs = np.array(all_runs)
    mean_curve = all_runs.mean(axis=0)
    std_curve = all_runs.std(axis=0)

    # Fit sinusoïde sur la moyenne
    t = np.arange(len(mean_curve))
    fit = fit_sinusoidal_model(t, mean_curve)


    plt.figure(figsize=(10, 5))
    plt.title(f"Planet {planet} — Mean survivals with std")
    plt.plot(t, mean_curve, label="mean", linewidth=2)
    plt.fill_between(t, mean_curve - std_curve, mean_curve + std_curve, alpha=0.3)
    plt.xlabel("Trip index")
    plt.ylabel("Survival probability")
    plt.legend()
    plt.grid(True)
    plt.show()


    if fit:
        print(f"\nPlanet {planet} sinusoid fit:")
        print(f"  w = {fit.w:.4f}")
        print(f"  amplitude = {fit.amplitude:.4f}")
        print(f"  phase = {fit.phase:.4f}")
        print(f"  c0 = {fit.c0:.4f}")
    else:
        print(f"Planet {planet}: sinusoidal fit failed.")


def main():
    print("=== ANALYZING planet_measurements.csv ===")
    df = load_csv()

    for planet in [0, 1, 2]:
        print(f"\n====== PLANET {planet} ======")
        analyze_planet(df, planet)


if __name__ == "__main__":
    main()
