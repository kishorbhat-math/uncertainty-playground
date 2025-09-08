from __future__ import annotations
import importlib
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str  # "int" | "float"
    low: float
    high: float
    default: float
    step: Optional[float] = None


def safe_import_func(module_path: str, func_name: str = "simulate") -> Optional[Callable[..., Any]]:
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, func_name)
    except Exception:
        return None


def _to_draws_by_time(arr: np.ndarray, time_len_hint: Optional[int] = None) -> np.ndarray:
    """
    Normalize a 2D array to shape (n_sims, T).
    Heuristics:
      - If arr.ndim == 1 -> (1, T)
      - If time_len_hint is provided, align the T dimension to that.
      - Otherwise, if arr.shape[0] >= arr.shape[1], assume arr is (T, n_sims) and transpose.
      - Else assume (n_sims, T).
    """
    arr = np.atleast_2d(arr)
    if arr.ndim != 2:
        raise ValueError("Expected a 1D/2D array-like result.")
    if time_len_hint is not None:
        if arr.shape[0] == time_len_hint:
            arr = arr.T  # (T, n_sims) -> (n_sims, T)
        elif arr.shape[1] == time_len_hint:
            pass  # already (n_sims, T)
        # else leave heuristics below
    if arr.shape[0] >= arr.shape[1]:
        # likely (T, n_sims) -> transpose
        arr = arr.T
    return arr  # (n_sims, T)


def adapt_result_to_dataframe(result: Any) -> pd.DataFrame:
    """
    Normalize simulate() result into a DataFrame with columns:
      - 'time'
      - 'sim_0', 'sim_1', ... (Monte Carlo draws) or a single series
    Accepts:
      - pd.DataFrame with/without 'time' column
      - 1D/2D numpy arrays
      - Generic iterables
    """
    # Pandas DataFrame
    if isinstance(result, pd.DataFrame):
        df = result.copy()
        if "time" in df.columns:
            time = df["time"].to_numpy()
            value_cols = [c for c in df.columns if c != "time"]
            if not value_cols:
                raise ValueError("DataFrame has 'time' but no value columns.")
            vals = df[value_cols].to_numpy()
            if vals.ndim == 1:
                vals = vals[None, :]
            draws_by_time = _to_draws_by_time(vals, time_len_hint=len(time))
            out = pd.DataFrame({"time": np.arange(draws_by_time.shape[1])})
            for i in range(draws_by_time.shape[0]):
                out[f"sim_{i}"] = draws_by_time[i, :]
            if out.shape[0] == len(time):
                out["time"] = time
            return out
        else:
            time = np.arange(len(df))
            vals = df.to_numpy()
            if vals.ndim == 1:
                vals = vals[None, :]
            draws_by_time = _to_draws_by_time(vals, time_len_hint=len(time))
            out = pd.DataFrame({"time": np.arange(draws_by_time.shape[1])})
            for i in range(draws_by_time.shape[0]):
                out[f"sim_{i}"] = draws_by_time[i, :]
            if out.shape[0] == len(time):
                out["time"] = time
            return out

    # Numpy array
    if isinstance(result, np.ndarray):
        arr = np.atleast_2d(result)
        draws_by_time = _to_draws_by_time(arr)
        out = pd.DataFrame({"time": np.arange(draws_by_time.shape[1])})
        for i in range(draws_by_time.shape[0]):
            out[f"sim_{i}"] = draws_by_time[i, :]
        return out

    # Iterable fallback (e.g., list/series)
    if hasattr(result, "__iter__"):
        arr = np.asarray(list(result), dtype=float)
        return pd.DataFrame({"time": np.arange(arr.size), "sim_0": arr})

    raise TypeError("Unsupported result type from simulate().")


def compute_quantiles(sim_df: pd.DataFrame, qs: Iterable[float] = (0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame:
    """
    Compute fan chart quantiles per time step.
    Works whether sim_df has many 'sim_*' columns (stochastic) or a single series.
    """
    sim_cols = [c for c in sim_df.columns if c.startswith("sim_")]
    qdf = pd.DataFrame({"time": sim_df["time"].to_numpy()})
    if not sim_cols:
        other_cols = [c for c in sim_df.columns if c != "time"]
        vals = sim_df[other_cols[0]].to_numpy()
        for q in qs:
            qdf[f"q{int(q*100):02d}"] = vals
        return qdf

    vals = sim_df[sim_cols].to_numpy()  # (T, n_sims) OR (T, 1)
    # Ensure vals is (T, n_sims)
    if vals.shape[0] != len(sim_df):
        vals = vals.T  # swap if needed

    # Now vals is (T, n_sims); take quantiles across the sims axis
    qvals = np.quantile(vals, q=list(qs), axis=1)  # -> (len(qs), T)
    for q, row in zip(qs, qvals):
        qdf[f"q{int(q*100):02d}"] = row
    return qdf


def plot_fan(qdf: pd.DataFrame, title: str = "") -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    t = qdf["time"].to_numpy()
    needed = {"q05", "q25", "q50", "q75", "q95"}
    if needed.issubset(qdf.columns):
        ax.fill_between(t, qdf["q05"], qdf["q95"], alpha=0.2, linewidth=0)
        ax.fill_between(t, qdf["q25"], qdf["q75"], alpha=0.3, linewidth=0)
        ax.plot(t, qdf["q50"], linewidth=2)
    else:
        for c in qdf.columns:
            if c != "time":
                ax.plot(t, qdf[c], linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig, ax


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    return buf.read()


def quantiles_to_csv_bytes(qdf: pd.DataFrame) -> bytes:
    return qdf.to_csv(index=False).encode("utf-8")
