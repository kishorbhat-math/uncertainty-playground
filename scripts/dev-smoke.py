from pathlib import Path
import numpy as np
import pandas as pd

from playground.demo_models import hb_simulate, wbkc_simulate, dfs_simulate
from playground.utils import adapt_result_to_dataframe, compute_quantiles, plot_fan, fig_to_png_bytes, quantiles_to_csv_bytes

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

def run_case(name, arr):
    df = adapt_result_to_dataframe(arr)
    qdf = compute_quantiles(df)
    fig, _ = plot_fan(qdf, title=name)
    (ART / f"{name}_quantiles.csv").write_bytes(quantiles_to_csv_bytes(qdf))
    (ART / f"{name}_fan.png").write_bytes(fig_to_png_bytes(fig))
    print(f"[ok] {name}: T={len(qdf['time'])}, saved PNG/CSV to artifacts/")

def main():
    # Small, fast runs
    run_case("hb_demo", hb_simulate(baseline_hb=12.0, iron_absorption=1.0, days=60))
    run_case("wbkc_demo", wbkc_simulate(time=60))
    run_case("dfs_demo", dfs_simulate(coverage=0.6, efficacy=0.3, duration_days=90))

if __name__ == "__main__":
    main()
