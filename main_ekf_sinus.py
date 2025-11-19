# main_ekf_sinus.py
from api_client import SphinxAPIClient
from strategy_ekf_sinus import EKFSinusStrategy
import pandas as pd

def main():
    print("=== EKF SINUS STRATEGY ===")

    client = SphinxAPIClient()
    client.start_episode()

    strat = EKFSinusStrategy(client, explore_steps=80)

    print("\nExploration phase...")
    strat.explore()

    print("\nExploitation phase...")
    strat.exploit(batch=3)

    df = pd.DataFrame(strat.df)
    df.to_csv("ekf_results.csv", index=False)
    print("\nSaved to ekf_results.csv")

    status = client.get_status()
    print("\nFinal status:")
    print(status)


if __name__ == "__main__":
    main()
