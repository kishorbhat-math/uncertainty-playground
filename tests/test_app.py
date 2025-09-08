import numpy as np
import pandas as pd
from playground.utils import adapt_result_to_dataframe, compute_quantiles

def test_adapt_and_quantiles():
    arr = np.random.randn(50, 100)  # T x n_sims
    df = adapt_result_to_dataframe(arr)
    qdf = compute_quantiles(df)
    assert "time" in qdf.columns
    for q in [5, 25, 50, 75, 95]:
        assert f"q{q:02d}" in qdf.columns
