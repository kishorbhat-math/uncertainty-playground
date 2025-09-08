import numpy as np

def hb_simulate(baseline_hb: float, iron_absorption: float, days: int):
    T = int(days)
    n = 300
    base = baseline_hb + np.linspace(0, iron_absorption, T)
    draws = base + np.random.normal(0, 0.3, size=(n, T))
    return draws

def wbkc_simulate(time: int):
    T = int(time)
    n = 400
    draws = np.cumsum(np.random.normal(0, 1, size=(n, T)), axis=1)
    return draws

def dfs_simulate(coverage: float, efficacy: float, duration_days: int):
    T = int(duration_days)
    n = 250
    mu = coverage * efficacy * 10
    trend = np.linspace(0, mu, T)
    draws = trend + np.random.normal(0, 0.5, size=(n, T))
    return draws
