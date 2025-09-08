from __future__ import annotations

import sys
import pathlib
import numpy as np
import streamlit as st

# Ensure "src" is on sys.path when running via Streamlit
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from playground.utils import (
    ParamSpec,
    safe_import_func,
    adapt_result_to_dataframe,
    compute_quantiles,
    plot_fan,
    fig_to_png_bytes,
    quantiles_to_csv_bytes,
)

st.set_page_config(page_title="Uncertainty Playground", page_icon="🎛️", layout="wide")
st.title("🎛️ Uncertainty Playground")
st.caption("Run simulate() from your model repos (or demo stubs) and visualize uncertainty as a fan chart.")

# --- Demo model stubs live in playground.demo_models for now ---
MODELS = {
    "Hemoglobin ODE (hb_model)": {
        "module": "playground.demo_models",
        "func": "hb_simulate",
        "params": [
            ParamSpec("baseline_hb", "float", 6.0, 18.0, 12.0, 0.1),
            ParamSpec("iron_absorption", "float", 0.0, 5.0, 1.0, 0.1),
            ParamSpec("days", "int", 7, 180, 60, 1),
        ],
    },
    "TBK Calibration MC (wbkc)": {
        "module": "playground.demo_models",
        "func": "wbkc_simulate",
        "params": [
            ParamSpec("time", "int", 5, 180, 60, 1),
        ],
    },
    "DFS Intervention (dfs_model)": {
        "module": "playground.demo_models",
        "func": "dfs_simulate",
        "params": [
            ParamSpec("coverage", "float", 0.0, 1.0, 0.6, 0.01),
            ParamSpec("efficacy", "float", 0.0, 1.0, 0.3, 0.01),
            ParamSpec("duration_days", "int", 30, 365, 180, 1),
        ],
    },
}

with st.sidebar:
    st.header("⚙️ Controls")
    model_name = st.selectbox("Model", list(MODELS.keys()))
    spec = MODELS[model_name]

    func_name = spec.get("func", "simulate")
    simulate = safe_import_func(spec["module"], func_name)

    params = {}
    for p in spec["params"]:
        if p.kind == "int":
            params[p.name] = st.slider(p.name, int(p.low), int(p.high), int(p.default), int(p.step or 1))
        else:
            params[p.name] = st.slider(p.name, float(p.low), float(p.high), float(p.default), float(p.step or 0.01))

    run = st.button("Run simulate()", type="primary")

# Fallback demo if import fails (shouldn't right now, since we point to demo_models)
def _demo_simulate(**kwargs):
    T = int(kwargs.get("days", kwargs.get("duration_days", kwargs.get("time", 60))))
    n_sims = 200
    arr = np.zeros((n_sims, T))
    for i in range(n_sims):
        x = 0.0
        for t in range(T):
            x = 0.9 * x + np.random.normal(0, 1)
            arr[i, t] = x
    return arr

if run:
    st.subheader(model_name)

    if simulate is None:
        st.warning(f"Could not import {func_name}() from {spec['module']}. Using a demo series.")
        result = _demo_simulate(**params)
    else:
        try:
            result = simulate(**params)
        except TypeError as e:
            st.error(f"Parameter mismatch calling {func_name}(): {e}")
            st.stop()
        except Exception as e:
            st.error(f"{func_name}() raised an error: {e}")
            st.stop()

    try:
        sim_df = adapt_result_to_dataframe(result)
        qdf = compute_quantiles(sim_df)
    except Exception as e:
        st.error(f"Could not adapt result to fan chart: {e}")
        st.stop()

    fig, _ = plot_fan(qdf, title=model_name)
    st.pyplot(fig, use_container_width=True)

    with st.expander("Show quantiles table"):
        st.dataframe(qdf.head(50))

    png_bytes = fig_to_png_bytes(fig)
    csv_bytes = quantiles_to_csv_bytes(qdf)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Download PNG", data=png_bytes,
                           file_name=f"{model_name.replace(' ', '_').lower()}_fan.png", mime="image/png")
    with c2:
        st.download_button("⬇️ Download CSV", data=csv_bytes,
                           file_name=f"{model_name.replace(' ', '_').lower()}_quantiles.csv", mime="text/csv")
else:
    st.info("Pick a model and click **Run simulate()** from the sidebar.")
