# main_changeaware.py
import pandas as pd
from api_client import SphinxAPIClient
from data_collector import DataCollector
from visualizations import (
    plot_survival_rates,
    plot_survival_by_planet,
    plot_moving_average,
    plot_risk_evolution,
    plot_episode_summary
)

from strat_second import ChangeAwareStickyTS

def main():
    try:
        client = SphinxAPIClient()
        collector = DataCollector(client)

        print("="*60)
        print("MORTY EXPRESS CHALLENGE - CHANGE-AWARE TS STRATEGY")
        print("="*60)

        client.start_episode()

        window_buf = 40
        cusum_h = 8.0
        cusum_k = 0.02
        forced_explore_after_reset = 6
        partial_reset = True

        strategy = ChangeAwareStickyTS(
            n_arms=3,
            prior_a=1, prior_b=1,
            epsilon_probe=0.05,
            min_stick=2,
            switch_margin=0.05,
            buffer_size=25,
            cusum_h=4.0,
            cusum_k=0.015,
            forced_explore_after_reset=8,
            partial_reset=False

        )

        total_morties_sent = 0
        morties_per_batch = 3
        all_data = []

        # send until 1000 morties or until API limit stops us
        while total_morties_sent < 1000:
            arm = strategy.select_arm()

            try:
                result = client.send_morties(arm, morties_per_batch)

                trip_data = {
                    'planet': arm,
                    'planet_name': client.get_planet_name(arm),
                    'morties_sent': result['morties_sent'],
                    'survived': result['survived'],  # boolean if all survived
                    'steps_taken': result['steps_taken'],
                    'morties_in_citadel': result['morties_in_citadel'],
                    'morties_on_planet_jessica': result['morties_on_planet_jessica'],
                    'morties_lost': result['morties_lost']
                }

                all_data.append(trip_data)

                # Pour SphinxAPI, "survived" = nombre de mortys vivants dans CE batch
                successes = result.get("survived", 0)
                trials = result.get("morties_sent", morties_per_batch)


                # Update the strategy
                strategy.observe(arm, successes, trials)

                total_morties_sent += morties_per_batch

                print(f"[{total_morties_sent}/1000] Sent {morties_per_batch} to {client.get_planet_name(arm)} "
                    f"(saved {successes}/{trials})")


            except Exception as e:
                print(f"Error sending Morties: {e}")
                break

        df = pd.DataFrame(all_data)

        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        for planet_name in df['planet_name'].unique():
            pdata = df[df['planet_name'] == planet_name]
            survival_rate = (pdata['morties_in_citadel'].sum() / (pdata['morties_sent'].sum())) * 100 if pdata['morties_sent'].sum()>0 else 0
            total_saved = pdata['morties_in_citadel'].sum()
            total_sent = pdata['morties_sent'].sum()

            print(f"\n{planet_name}:")
            print(f"  Batches Sent: {len(pdata)}")
            print(f"  Total Morties Sent: {total_sent}")
            print(f"  Survival Rate (by Morties): {survival_rate:.2f}%")
            print(f"  Morties Saved: {total_saved}")
            print(f"  Morties Lost: {pdata['morties_lost'].iloc[-1] if len(pdata)>0 else 0}")

        overall_saved = df['morties_in_citadel'].sum() if 'morties_in_citadel' in df else 0
        overall_sent = df['morties_sent'].sum() if 'morties_sent' in df else total_morties_sent
        overall_survival_pct = (overall_saved / overall_sent) * 100 if overall_sent>0 else 0
        print("\nOverall Survival Rate (by Morties): {:.2f}%".format(overall_survival_pct))

        # Save & visualize
        collector.trips_data = all_data
        collector.save_data("morties_changeaware_strategy.csv")

        print("\nGenerating plots...")
        if len(df) > 0:
            plot_survival_rates(df)
            plot_survival_by_planet(df)
            plot_moving_average(df, window=10)
            plot_risk_evolution(df)
            plot_episode_summary(df)

        print("\nFinal Status:")
        status = client.get_status()
        for k, v in status.items():
            print(f"  {k}: {v}")

        print("\n" + "="*60)
        print("CHANGE-AWARE STRATEGY RUN COMPLETE!")
        print("="*60)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
