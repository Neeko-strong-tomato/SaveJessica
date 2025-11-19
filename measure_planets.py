# measure_planets.py
import os
import csv
from api_client import SphinxAPIClient


CSV_FILE = "planet_measurements.csv"
RUNS_PER_PLANET = 5
TRIPS_PER_RUN = 1000
MORTIES_PER_TRIP = 3


def append_row(row):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def do_single_run(planet, run_id):
    print(f"\n=== RUN planet {planet}, run {run_id} ===")

    client = SphinxAPIClient()
    client.start_episode()

    for trip in range(1, TRIPS_PER_RUN + 1):
        try:
            res = client.send_morties(planet, MORTIES_PER_TRIP)
        except Exception as e:
            print("API error:", e)
            break

        row = {
            "planet": planet,
            "run": run_id,
            "trip_index": trip,
            "survived": res["survived"],
            "morties_sent": res["morties_sent"],
            "morties_lost": res["morties_lost"],
            "morties_in_citadel": res["morties_in_citadel"],
            "morties_on_planet_jessica": res["morties_on_planet_jessica"],
            "steps_taken": res["steps_taken"]
        }
        append_row(row)

        if trip % 100 == 0:
            print(f"  progress: {trip}/{TRIPS_PER_RUN}")

    print("Run completed.")


def main():
    print("=== PLANET MEASUREMENT SCRIPT ===")
    print(f"Data will be appended to {CSV_FILE}")

    for planet in [0, 1, 2]:
        for r in range(RUNS_PER_PLANET):
            do_single_run(planet, run_id=r)

    print("\nAll runs complete. Data saved.")


if __name__ == "__main__":
    main()
