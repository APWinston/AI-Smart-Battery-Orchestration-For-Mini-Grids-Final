#!/usr/bin/env python
# coding: utf-8
"""
Phase 5 — Fleet Benchmark: PPO vs four baselines, all project KPIs
==================================================================
Evaluates the trained residual-PPO policy against four controllers spanning the
sophistication ladder, across the whole fleet:

    Controller     Foresight             Needs grid model   Role
    -------------  --------------------  -----------------  ---------------------
    Simple RBC     none (reactive)       no                 fixed-threshold SoC
    Advanced RBC   fixed time schedule   no                 scheduled reactive
    MPC            LSTM forecast         battery model      receding-horizon LP
    DP             perfect hindsight     battery model      optimal lower bound
    PPO            learned correction    no                 the trained policy

Grid:
    6 locations  x  N years (auto-detected)  x  system-size tiers
    For each cell, every requested controller runs on the SAME data, so the
    comparison is apples-to-apples.

KPIs (all five project metrics, plus renewable fraction), computed BY THE ENV
through one code path for every controller:
    ENS (kWh, lower)   LOLP (%, lower)   EFC (cycles, lower)
    SoH (%, higher)    SoC std (%, lower)

Design — every controller acts THROUGH the env
    Each controller only DECIDES a battery power each hour; the env applies the
    real physics (efficiency, SoC limits, degradation) and accumulates the KPIs.
    No KPI is recomputed by hand, so MPC/DP can only get the DECISION wrong,
    never the scoring, and all columns are consistent by construction.

    * PPO runs in RESIDUAL mode (as trained): its action is a correction to the
      load-following anchor that lives INSIDE the env.
    * The four baselines run in ABSOLUTE mode (env.RESIDUAL = False): the action
      IS the battery power command.
    Load-following is the residual anchor only; it is NOT a column here. Simple
    RBC is a distinct fixed-threshold SoC controller.

Validation note
    MPC is the one controller whose forecast path could not be exercised at
    build time. It de-normalises the env's LSTM forecast by inverting scaler_y
    (fit on ['ssrd_wm2','load_kw']); if scaler_y was fit differently, adjust the
    three lines marked (*) in mpc_forecast(). `--mpc-mode perfect` sidesteps the
    forecast entirely (perfect-foresight MPC) and is a useful ablation. Validate
    incrementally:
        python phase5_evaluation_param.py --quick
        python phase5_evaluation_param.py --controllers simple dp --quick
        python phase5_evaluation_param.py --best
        python phase5_evaluation_param.py --best            # full grid

Inputs:
    ../models/ppo_srep_param.zip      (or --best -> best/best_model.zip)
    ../models/vecnormalize_param.pkl
    ../data/master_dataset_raw.csv

Outputs:
    ../data/phase5_per_episode.csv    every (site, size, year, controller) row
    ../data/phase5_by_site.csv        mean over years, per site + size
    ../data/phase5_summary.csv        grid mean, per controller
    ../data/phase5_summary.png        grouped bar chart per KPI
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from phase3_parametrised_env import SREPMiniGridEnv

warnings.filterwarnings('ignore')

try:
    from scipy.optimize import linprog
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

DATA_PATH = '../data/master_dataset_raw.csv'
MODEL_DIR = '../models'
OUT_DIR   = '../data'
EVAL_SEED = 12345

# The six fleet locations present in the dataset (3 train + 3 held-out).
LOCATIONS = ['Tamale', 'Kumasi', 'Axim', 'Bolgatanga', 'Accra', 'Akosombo']

# SREP size tiers: (solar_kwp, battery_kwh, design_mean_load_kw), mirroring
# SREPMiniGridEnv.TIERS. All three run by default; trim with e.g. --sizes 75.
SIZE_TIERS = [(50, 160, 5.6), (75, 237, 8.4), (120, 378, 13.4)]

KPI_COLS = ['ENS_kwh', 'LOLP_pct', 'Served_pct', 'EFC_cycles', 'SOH_pct',
            'SOC_std_pct', 'RenFrac']
KPI_META = [
    ('Energy Served',                   '%',      'Served_pct',  'Higher'),
    ('Energy Not Served (ENS)',         'kWh',    'ENS_kwh',     'Lower'),
    ('Loss of Load Probability (LOLP)', '%',      'LOLP_pct',    'Lower'),
    ('Equivalent Full Cycles (EFC)',    'Cycles', 'EFC_cycles',  'Lower'),
    ('Final State of Health (SoH)',     '%',      'SOH_pct',     'Higher'),
    ('SoC Standard Deviation',          '%',      'SOC_std_pct', 'Lower'),
]


# ============================================================
# KPI EXTRACTION  (straight from the env's own episode stats)
# ============================================================

def kpis_from_env(env):
    s = env.get_episode_stats()
    return {
        'ENS_kwh':     s['unmet_kwh'],
        'LOLP_pct':    100.0 * s['lolp'],
        'Served_pct':  100.0 * s['served_kwh'] / max(s['load_kwh'], 1.0),
        'EFC_cycles':  s['efc'],
        'SOH_pct':     100.0 * s['final_soh'],
        'SOC_std_pct': 100.0 * s['soc_std'],
        'RenFrac':     s['renewable_fraction'],
    }


# ============================================================
# DATA HELPERS
# ============================================================

def physical_series(env, t0, n):
    """Realized solar_kw & load_kw for hours [t0, t0+n) — exogenous, so they are
    independent of any control action (same construction as env.step)."""
    df    = env.loc_df
    end   = min(t0 + n, len(df))
    ssrd  = df['ssrd_wm2'].values[t0:end]
    loadr = df['load_kw'].values[t0:end]
    solar = (ssrd / 1000.0) * env.solar_kwp * env.PV_DERATING
    load  = loadr * env.load_scale
    return solar.astype(np.float64), load.astype(np.float64)


def n_years_for(env, location, episode_len):
    df = env._loc_frames[location]
    return max(1, (len(df) - env.LOOKBACK) // episode_len)


def start_episode(env, site, year, carry=False):
    """Position the env on `site` at the start of calendar `year` and reset the
    per-year KPI accumulators (KPIs always cover exactly this one year).

    carry=False : fresh battery — reset SoC and SoH to INIT (used at year 0).
    carry=True  : KEEP the SoC and SoH the pack ended the previous year with, so
                  degradation compounds over the project lifetime. Because the
                  calendar years are contiguous, carrying SoC across the boundary
                  is also physically continuous. The env is NOT re-loaded here
                  (it is already on this site from year 0), which is what lets the
                  ageing persist."""
    if not carry:
        env.reset(seed=EVAL_SEED, options=site)
        env.soc = env.SOC_INIT
        env.soh = env.SOH_INIT
    env.t          = env.LOOKBACK + year * env.episode_len
    env.step_count = 0
    env._prev_soc      = env.soc
    env._rf_direction  = 0
    env._rf_half_start = env.soc
    env._reset_accumulators()


# ============================================================
# CONTROLLERS  (each returns an action in [-1, 1])
# ============================================================

def simple_rbc_action(env, obs=None, ctx=None):
    """Fixed-threshold bang-bang SoC controller (non-load-following).

    A coarse mode controller that switches between discharge and charge by SoC
    against the env's OWN setpoints, with hysteresis, and acts at a FIXED/FULL
    rate rather than metering to the exact net load:
        discharge -> charge   flip when SoC <= env.SOC_MIN
        charge    -> discharge flip when SoC >= env.SOC_MAX

    In 'discharge' mode it dumps the battery at full rate to serve a deficit,
    over-discharging relative to demand (the excess is wasted), which drains it
    fast. In 'charge' mode it banks surplus only and sheds deficits until the
    battery refills to SOC_MAX. No metering, no anticipation -> distinctly weaker
    than exact-matching dispatch, on every hour, not just after a drain.
    """
    if env.step_count == 0:
        env._rbc_mode = 'discharge'
    row      = env.loc_df.iloc[env.t]
    solar_kw = (row['ssrd_wm2'] / 1000.0) * env.solar_kwp * env.PV_DERATING
    load_kw  = row['load_kw'] * env.load_scale
    surplus  = solar_kw - load_kw                      # + surplus, - deficit

    mode = getattr(env, '_rbc_mode', 'discharge')
    if mode == 'discharge' and env.soc <= env.SOC_MIN + 1e-6:
        mode = 'charge'                                # drained -> recharge lockout
    elif mode == 'charge' and env.soc >= env.SOC_MAX - 1e-6:
        mode = 'discharge'                             # refilled -> serve again
    env._rbc_mode = mode

    if surplus >= 0:
        power = surplus                                # bank surplus (can't exceed it)
    elif mode == 'discharge':
        power = -env.max_power_kw                       # full-rate discharge (coarse)
    else:
        power = 0.0                                     # charge lockout: shed deficit
    return np.array([np.clip(power / env.max_power_kw, -1.0, 1.0)], dtype=np.float32)


def advanced_rbc_action(env, obs=None, ctx=None):
    """Reactive controller with a fixed time-of-day schedule (no forecast).

    Preserves charge through the solar window for the evening peak, discharges
    freely in the peak, and holds a reserve floor overnight. Reactive within each
    window; the schedule itself is fixed.
    """
    row      = env.loc_df.iloc[env.t]
    solar_kw = (row['ssrd_wm2'] / 1000.0) * env.solar_kwp * env.PV_DERATING
    load_kw  = row['load_kw'] * env.load_scale
    hour     = int(row['hour'])
    net      = solar_kw - load_kw          # + surplus -> charge, - deficit -> discharge
    power    = net                         # exact-match starting point

    if 8 <= hour <= 16:                    # solar window: bank charge for evening
        if net < 0:                        # shallow daytime deficit -> ration discharge
            power = net * 0.5
    elif 17 <= hour <= 22:                 # evening peak: serve fully
        power = net
    else:                                  # overnight: hold a reserve floor
        if net < 0 and env.soc < 0.30:
            power = net * 0.5
    return np.array([np.clip(power / env.max_power_kw, -1.0, 1.0)], dtype=np.float32)


def ppo_action(env, obs, ctx):
    """The trained residual policy. obs is the 60-dim residual observation."""
    norm = ctx['vecnorm'].normalize_obs(obs.reshape(1, -1))
    act, _ = ctx['model'].predict(norm, deterministic=True)
    return act[0]


# ---- MPC ---------------------------------------------------

def mpc_forecast(env, horizon, mode):
    """Return (solar_kw, load_kw) forecasts over `horizon` hours.

    mode='perfect' : realized future from loc_df (perfect-foresight MPC; an
                     ablation showing the ceiling foresight alone could reach).
    mode='lstm'    : the env's own LSTM forecast.

    NOTE (verify): _get_lstm_forecast() returns values clipped to [0,1]. We
    invert scaler_y (fit on LSTM_TARGETS = ['ssrd_wm2','load_kw']) to recover
    physical units, then rebuild kW exactly as env.step does. If scaler_y was fit
    differently, fix the three lines marked (*).
    """
    if mode == 'perfect':
        return physical_series(env, env.t, horizon)
    sN, lN = env._get_lstm_forecast()                       # (48,), normalised
    phys   = env.scaler_y.inverse_transform(
        np.stack([sN, lN], axis=1))                         # (*) -> [ssrd_wm2, load_kw]
    H      = min(horizon, phys.shape[0])
    solar  = (np.clip(phys[:H, 0], 0, None) / 1000.0) * env.solar_kwp * env.PV_DERATING  # (*)
    load   =  np.clip(phys[:H, 1], 0, None) * env.load_scale                              # (*)
    return solar, load


def mpc_lp_power(env, solar, load):
    """Receding-horizon LP: choose charge/discharge over the horizon to minimise
    unmet energy (+ tiny throughput cost, + small terminal-SoC reward), return
    THIS hour's battery power (+charge / -discharge). Converges to DP as the
    horizon -> full year with a perfect forecast.
    """
    H = len(solar)
    if H == 0:
        return 0.0
    cap  = env.battery_kwh * env.soh
    Pmax = env.max_power_kw
    ec, ed = env.ETA_CHARGE, env.ETA_DISCHARGE
    e0   = env.soc * cap
    emin = env.SOC_MIN * cap
    emax = env.SOC_MAX * cap
    net  = load - solar
    lf0  = float(np.clip(solar[0] - load[0], -Pmax, Pmax))   # exact-match fallback

    if not _HAVE_SCIPY:
        return lf0

    n = 3 * H
    cI = lambda t: t            # charge
    dI = lambda t: H + t        # discharge
    uI = lambda t: 2 * H + t    # unmet

    obj = np.zeros(n)
    obj[2 * H:3 * H] = 1.0                       # minimise total unmet
    obj[0:2 * H]    += 1e-4                       # tiny throughput regulariser
    wterm = 0.05                                  # value retained energy at horizon end
    for t in range(H):
        obj[cI(t)] += -wterm * ec
        obj[dI(t)] += -wterm * (-1.0 / ed)

    A, b = [], []
    # balance:  u - c + d >= net   ->   -u + c - d <= -net
    for t in range(H):
        r = np.zeros(n); r[uI(t)] = -1; r[cI(t)] = 1; r[dI(t)] = -1
        A.append(r); b.append(-net[t])
    # SoC bounds for e_t, t = 1..H   (e_t = e0 + sum_{k<t}(ec*c - d/ed))
    for t in range(1, H + 1):
        r = np.zeros(n)
        for k in range(t):
            r[cI(k)] += ec; r[dI(k)] += -1.0 / ed
        A.append(r.copy()); b.append(emax - e0)      # e_t <= emax
        A.append(-r);       b.append(-(emin - e0))   # e_t >= emin
    bounds = [(0, Pmax)] * H + [(0, Pmax)] * H + [(0, None)] * H

    try:
        res = linprog(obj, A_ub=np.array(A), b_ub=np.array(b),
                      bounds=bounds, method='highs')
        if res.success:
            return float(res.x[cI(0)] - res.x[dI(0)])
    except Exception:
        pass
    return lf0


def make_mpc_action(horizon, mode):
    def _act(env, obs=None, ctx=None):
        solar, load = mpc_forecast(env, horizon, mode)
        power = mpc_lp_power(env, solar, load)
        return np.array([np.clip(power / env.max_power_kw, -1.0, 1.0)], dtype=np.float32)
    return _act


# ---- Dynamic Programming (perfect-hindsight lower bound) ----

def dp_plan(env, site, year, n_levels):
    """Backward DP over a discretised SoC grid with the realized full-year series
    -> the action sequence that MINIMISES total unmet energy (ENS) given perfect
    hindsight. The absolute-best-case reference: no causal controller can beat it
    on ENS. Returns an action array (length episode_len) to replay through the env.
    """
    t0 = env.LOOKBACK + year * env.episode_len
    T  = env.episode_len
    solar, load = physical_series(env, t0, T)
    if len(solar) < T:                                   # short-tail safety
        T = len(solar)

    cap  = env.battery_kwh * env.soh        # current (carried) health, not INIT
    Pmax = env.max_power_kw
    ec, ed = env.ETA_CHARGE, env.ETA_DISCHARGE
    smin, smax = env.SOC_MIN, env.SOC_MAX

    soc_grid = np.linspace(smin, smax, n_levels)
    E   = soc_grid * cap
    dE  = E[None, :] - E[:, None]                        # (N,N) energy change i->j
    charge_kw = np.where(dE >= 0,  dE / ec, 0.0)
    disch_kw  = np.where(dE <  0, -dE * ed, 0.0)
    bp  = charge_kw - disch_kw                           # signed battery power (+charge)
    feasible = (charge_kw <= Pmax + 1e-9) & (disch_kw <= Pmax + 1e-9)
    INF = 1e18

    V = np.zeros(n_levels)                               # terminal cost-to-go
    nxt = np.zeros((T, n_levels), dtype=np.int16)
    for t in range(T - 1, -1, -1):
        supply = solar[t] + np.maximum(0.0, -bp)         # (N,N)
        demand = load[t]  + np.maximum(0.0,  bp)
        unmet  = np.clip(demand - supply, 0.0, load[t])
        cost   = np.where(feasible, unmet + V[None, :], INF)
        j      = np.argmin(cost, axis=1)
        V      = cost[np.arange(n_levels), j]
        nxt[t] = j

    i = int(np.argmin(np.abs(soc_grid - env.SOC_INIT)))
    plan = np.zeros(T, dtype=np.float32)
    for t in range(T):
        j = nxt[t, i]
        plan[t] = np.clip(bp[i, j] / Pmax, -1.0, 1.0)
        i = j
    return plan


def make_dp_action(plan):
    def _act(env, obs=None, ctx=None):
        k = min(env.step_count, len(plan) - 1)
        return np.array([plan[k]], dtype=np.float32)
    return _act


# ============================================================
# ROLLOUT
# ============================================================

CONTROLLERS = {
    'simple':   dict(label='Simple RBC',   residual=False, needs_obs=False),
    'advanced': dict(label='Advanced RBC', residual=False, needs_obs=False),
    'mpc':      dict(label='MPC',          residual=False, needs_obs=False),
    'dp':       dict(label='DP (bound)',   residual=False, needs_obs=False),
    'ppo':      dict(label='PPO',          residual=True,  needs_obs=True),
}


def run_episode(env, site, year, action_fn, residual, needs_obs, ctx=None, carry=False):
    env.RESIDUAL = residual                  # instance override of the class flag
    start_episode(env, site, year, carry=carry)
    obs = env._build_obs() if needs_obs else None
    done = False
    while not done:
        a = action_fn(env, obs, ctx)
        obs, _, term, trunc, _ = env.step(a)
        done = term or trunc
    return kpis_from_env(env)


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--controllers', nargs='+',
                   default=['simple', 'advanced', 'mpc', 'dp', 'ppo'],
                   choices=list(CONTROLLERS.keys()))
    p.add_argument('--locations', nargs='+', default=LOCATIONS)
    p.add_argument('--sizes', nargs='+', type=int, default=[t[0] for t in SIZE_TIERS],
                   help='solar_kwp tiers to evaluate (50 75 120)')
    p.add_argument('--years', nargs='+', type=int, default=None,
                   help='specific year indices; default = all detected')
    p.add_argument('--episode-len', type=int, default=8760)
    p.add_argument('--best', action='store_true',
                   help='use models/best/best_model.zip')
    p.add_argument('--mpc-horizon', type=int, default=24)
    p.add_argument('--mpc-mode', choices=['lstm', 'perfect'], default='lstm')
    p.add_argument('--dp-levels', type=int, default=41,
                   help='SoC discretisation for DP (higher = tighter bound, slower)')
    p.add_argument('--quick', action='store_true',
                   help='smoke: 1 location, 1 size, 1 year, controllers as given')
    args = p.parse_args()

    if args.quick:
        args.locations = args.locations[:1]
        args.sizes     = args.sizes[:1]
        args.years     = [0]

    print("=" * 64)
    print("  PHASE 5 — FLEET BENCHMARK")
    print("=" * 64)
    print(f"  controllers : {', '.join(CONTROLLERS[c]['label'] for c in args.controllers)}")
    print(f"  locations   : {', '.join(args.locations)}")
    print(f"  sizes (kWp) : {args.sizes}")
    if args.mpc_mode == 'lstm' and 'mpc' in args.controllers:
        print(f"  MPC         : horizon {args.mpc_horizon} h, LSTM forecast")
    if 'mpc' in args.controllers and not _HAVE_SCIPY:
        print("  WARNING: scipy not found -> MPC falls back to exact-match dispatch.")

    # ---- size tier lookup (solar_kwp -> battery_kwh, design load) ----
    tier_by_kwp = {t[0]: t for t in SIZE_TIERS}

    # ---- PPO model (only if requested) ----
    ctx = {}
    if 'ppo' in args.controllers:
        mp = (f'{MODEL_DIR}/best/best_model.zip' if args.best
              else f'{MODEL_DIR}/ppo_srep_param.zip')
        vp = (f'{MODEL_DIR}/best/vecnormalize_param.pkl' if args.best
              else f'{MODEL_DIR}/vecnormalize_param.pkl')
        print(f"  PPO model   : {mp}")
        ctx['model'] = PPO.load(mp, device='cpu')
        dummy = DummyVecEnv([lambda: SREPMiniGridEnv(
            data_path=DATA_PATH, model_dir=MODEL_DIR, episode_len=args.episode_len)])
        vn = VecNormalize.load(vp, dummy)
        vn.training = False; vn.norm_reward = False
        ctx['vecnorm'] = vn

    # ---- one reusable env ----
    env = SREPMiniGridEnv(data_path=DATA_PATH, model_dir=MODEL_DIR,
                          episode_len=args.episode_len)

    rows = []
    for loc in args.locations:
        n_years = n_years_for(env, loc, args.episode_len)
        years   = args.years if args.years is not None else list(range(n_years))
        years   = [y for y in years if y < n_years]
        for kwp in args.sizes:
            _, bat, load = tier_by_kwp[kwp]
            site = {'solar_kwp': kwp, 'battery_kwh': bat,
                    'mean_load_kw': load, 'location': loc}
            tag = f"{loc} {kwp}kWp"
            # Each controller runs the years SEQUENTIALLY on its own battery: SoH
            # (and SoC) carry from one year into the next, so the pack ages over
            # the project lifetime and a harder-cycling controller wears it
            # faster. The battery is reset to full health only at year 0, so
            # later-year conditions become controller-dependent by design.
            for cname in args.controllers:
                spec = CONTROLLERS[cname]
                # lifetime running totals for this controller (reset at year 0):
                # ENS and EFC accumulate; LOLP is recomputed as total LOL hours
                # over total hours, i.e. a project-to-date reliability figure.
                cum_ens = cum_efc = 0.0
                cum_lol = cum_steps = 0
                cum_served = cum_load = 0.0
                for yi, year in enumerate(years):
                    carry = yi > 0
                    if cname == 'simple':     fn = simple_rbc_action
                    elif cname == 'advanced': fn = advanced_rbc_action
                    elif cname == 'ppo':      fn = ppo_action
                    elif cname == 'mpc':      fn = make_mpc_action(args.mpc_horizon, args.mpc_mode)
                    elif cname == 'dp':
                        # DP must be planned on the pack as it is NOW (carried
                        # SoH), so position the env for this year first, then plan.
                        env.RESIDUAL = False
                        start_episode(env, site, year, carry=carry)
                        fn = make_dp_action(dp_plan(env, site, year, args.dp_levels))
                    k = run_episode(env, site, year, fn, spec['residual'],
                                    spec['needs_obs'], ctx, carry=carry)
                    # accumulate over the controller's lifetime
                    cum_ens    += k['ENS_kwh']
                    cum_efc    += k['EFC_cycles']
                    cum_lol    += env.ep_lol_hours
                    cum_steps  += env.ep_steps
                    cum_served += env.ep_served_kwh
                    cum_load   += env.ep_load_kwh
                    cum_lolp    = 100.0 * cum_lol / max(cum_steps, 1)
                    cum_served_pct = 100.0 * cum_served / max(cum_load, 1.0)
                    rows.append({'location': loc, 'size_kwp': kwp, 'year': year,
                                 'controller': spec['label'], **k,
                                 'ENS_cum_kwh':     cum_ens,
                                 'EFC_cum_cycles':  cum_efc,
                                 'LOLP_cum_pct':    cum_lolp,
                                 'Served_cum_pct':  cum_served_pct})
                    print(f"  {tag}  {spec['label']:<13} yr{year}: "
                          f"Served={k['Served_pct']:5.1f}%  ENS={k['ENS_kwh']:8.0f}  "
                          f"cumENS={cum_ens:9.0f}  SoH={k['SOH_pct']:5.2f}%")

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(f'{OUT_DIR}/phase5_per_episode.csv', index=False)

    # ---- aggregate: mean over years, per site + size ----
    by_site = (df.groupby(['location', 'size_kwp', 'controller'])[KPI_COLS]
                 .mean().reset_index())
    by_site.to_csv(f'{OUT_DIR}/phase5_by_site.csv', index=False)

    # ---- aggregate: grid mean, per controller ----
    summary = df.groupby('controller')[KPI_COLS].mean().reset_index()
    summary.to_csv(f'{OUT_DIR}/phase5_summary.csv', index=False)

    # ---- printed summary table (PPO vs each baseline) ----
    order = [CONTROLLERS[c]['label'] for c in args.controllers]
    summary = summary.set_index('controller').reindex(order)
    print("\n" + "=" * 78)
    print(f"  GRID-MEAN KPIs  ({len(df)} episodes: "
          f"{df.location.nunique()} loc x {df.size_kwp.nunique()} size x "
          f"{df.year.nunique()} yr)")
    print("  " + "-" * 76)
    print(f"  {'KPI':<30}{'Unit':<7}" + "".join(f"{o.split()[0]:>11}" for o in order))
    print("  " + "-" * 76)
    for name, unit, col, _ in KPI_META:
        print(f"  {name:<30}{unit:<7}" + "".join(
            f"{summary.loc[o, col]:>11.2f}" for o in order))
    print("=" * 78)
    if 'DP (bound)' in order:
        print("  DP = perfect-hindsight lower bound on ENS/LOLP "
              "(no causal controller can beat it).")

    # ---- lifetime totals: take the LAST year's cumulative row per controller,
    #      then average those lifetime totals across sites/sizes ----
    last = (df.sort_values('year')
              .groupby(['location', 'size_kwp', 'controller'], as_index=False)
              .last())
    life = (last.groupby('controller')
                .agg(Served_life_pct=('Served_cum_pct', 'mean'),
                     ENS_total_kwh=('ENS_cum_kwh', 'mean'),
                     EFC_total=('EFC_cum_cycles', 'mean'),
                     LOLP_life_pct=('LOLP_cum_pct', 'mean'),
                     SoH_final_pct=('SOH_pct', 'mean'))
                .reindex(order))
    # mean per-year LOLP (average of the yearly LOLP values, all years/sites)
    life['LOLP_mean_pct'] = df.groupby('controller')['LOLP_pct'].mean().reindex(order)
    life = life[['Served_life_pct', 'ENS_total_kwh', 'LOLP_mean_pct',
                 'LOLP_life_pct', 'SoH_final_pct', 'EFC_total']]
    life.to_csv(f'{OUT_DIR}/phase5_lifetime.csv')
    n_life = df.year.nunique()
    print("\n" + "=" * 78)
    print(f"  LIFETIME TOTALS  (cumulative over {n_life} years, "
          f"mean across {last.location.nunique()} loc x {last.size_kwp.nunique()} size)")
    print("  " + "-" * 76)
    print(f"  {'Controller':<14}{'Served':>9}{'ENS total':>12}{'LOLP mean':>11}"
          f"{'LOLP life':>11}{'SoH final':>11}{'EFC total':>11}")
    print(f"  {'':<14}{'%':>9}{'kWh':>12}{'%':>11}{'%':>11}{'%':>11}{'cycles':>11}")
    print("  " + "-" * 76)
    for o in order:
        print(f"  {o:<14}{life.loc[o,'Served_life_pct']:>9.2f}"
              f"{life.loc[o,'ENS_total_kwh']:>12.0f}"
              f"{life.loc[o,'LOLP_mean_pct']:>11.2f}"
              f"{life.loc[o,'LOLP_life_pct']:>11.2f}"
              f"{life.loc[o,'SoH_final_pct']:>11.2f}"
              f"{life.loc[o,'EFC_total']:>11.1f}")
    print("=" * 78)

    # ---- KPI bar charts: one full-size figure per KPI + a 2x3 overview ----
    from matplotlib.ticker import ScalarFormatter
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(order)))
    short  = [o.split()[0] for o in order]

    def _fmt_val(v):
        if abs(v) >= 1000: return f"{v:,.0f}"
        if abs(v) >= 10:   return f"{v:.0f}"
        return f"{v:.1f}"

    def _draw_kpi(ax, name, unit, col, direction, legend=False, big=False):
        vals = [summary.loc[o, col] for o in order]
        bars = ax.bar(short, vals, color=colors)
        pos  = [v for v in vals if v > 0]
        # ENS/LOLP span orders of magnitude (naive RBC dwarfs the rest) -> log,
        # but force PLAIN number tick labels (e.g. 1000, not 10^3).
        use_log = col in ('ENS_kwh', 'LOLP_pct') and pos and min(pos) > 0
        if use_log:
            ax.set_yscale('log')
            f = ScalarFormatter(); f.set_scientific(False); f.set_useOffset(False)
            ax.yaxis.set_major_formatter(f)
            ax.yaxis.set_minor_formatter(plt.NullFormatter())
            ax.set_ylim(bottom=min(pos) * 0.6)             # avoid spurious sub-1 "0" ticks
        fs = 13 if big else 10
        ax.set_title(f"{name}\n({unit}, {direction} better)", fontsize=fs, fontweight='bold')
        ax.set_xlabel('Controller', fontsize=fs - 1)
        ax.set_ylabel(f"{name} ({unit})", fontsize=fs - 1)
        ax.grid(True, axis='y', alpha=0.35)
        ax.tick_params(axis='x', labelrotation=20, labelsize=fs - 2)
        ax.tick_params(axis='y', labelsize=fs - 2)
        for b, v in zip(bars, vals):                       # actual value on each bar
            ax.annotate(_fmt_val(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha='center', va='bottom', fontsize=fs - 2,
                        xytext=(0, 2), textcoords='offset points')
        ax.set_ylim(top=max(vals) * (2.2 if use_log else 1.18))
        if legend:
            handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(order))]
            ax.legend(handles, order, title='Controller', fontsize=8, framealpha=0.9)

    kpi_file = {'Served_pct': 'served', 'ENS_kwh': 'ens', 'LOLP_pct': 'lolp',
                'EFC_cycles': 'efc', 'SOH_pct': 'soh', 'SOC_std_pct': 'socstd'}
    # individual, full-size, self-explanatory figures
    for name, unit, col, direction in KPI_META:
        figk, axk = plt.subplots(figsize=(9, 6))
        _draw_kpi(axk, name, unit, col, direction, legend=True, big=True)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/phase5_kpi_{kpi_file[col]}.png", dpi=150)
        plt.close(figk)

    # combined 2x3 overview (big panels, no per-panel legend -> one shared legend)
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    for ax, (name, unit, col, direction) in zip(axes.flat, KPI_META):
        _draw_kpi(ax, name, unit, col, direction, legend=False)
    for ax in axes.flat[len(KPI_META):]:
        ax.axis('off')
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(order))]
    fig.legend(handles, order, title='Controller', loc='lower center',
               ncol=len(order), fontsize=10, framealpha=0.95)
    fig.suptitle('Phase 5 — PPO vs baselines (grid mean over fleet)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f'{OUT_DIR}/phase5_summary.png', dpi=150)
    plt.close(fig)

    # ---- lifetime & improvement summary (the "everything" figure) ----
    EOL      = 0.80          # standard end-of-life SoH threshold
    SOH0     = 1.0
    base     = 'Simple RBC' if 'Simple RBC' in order else order[0]
    cum_by_yr = df.groupby(['controller', 'year'])['ENS_cum_kwh'].mean().unstack('controller')
    soh_by_yr = df.groupby(['controller', 'year'])['SOH_pct'].mean().unstack('controller')
    srv_by_yr = df.groupby(['controller', 'year'])['Served_pct'].mean().unstack('controller')
    yrs       = sorted(df.year.unique())

    # battery lifespan = years to reach EOL, linear-extrapolated from the mean
    # annual SoH loss each controller actually inflicted on its pack.
    ann_loss  = ((SOH0 - life['SoH_final_pct'] / 100.0) / max(n_life, 1)).clip(lower=1e-6)
    lifespan  = (SOH0 - EOL) / ann_loss
    ens_red   = 100.0 * (1 - life['ENS_total_kwh'] / life.loc[base, 'ENS_total_kwh'])
    life_ext  = 100.0 * (lifespan / lifespan[base] - 1)
    srv_gain  = life['Served_life_pct'] - life.loc[base, 'Served_life_pct']   # pts
    _lolp_b   = life.loc[base, 'LOLP_life_pct']
    lolp_red  = 100.0 * (1 - life['LOLP_life_pct'] / _lolp_b) if _lolp_b else life['LOLP_life_pct'] * 0.0

    fig2, ax = plt.subplots(2, 3, figsize=(20, 10))
    short = [o.split()[0] for o in order]
    cols  = plt.cm.viridis(np.linspace(0.1, 0.9, len(order)))

    for o in order:                                  # A: cumulative ENS
        ax[0, 0].plot(yrs, [cum_by_yr.loc[y, o] for y in yrs], marker='o', label=o.split()[0])
    ax[0, 0].set_title('Cumulative energy not served over project life')
    ax[0, 0].set_xlabel('year'); ax[0, 0].set_ylabel('cumulative ENS (kWh)')
    ax[0, 0].set_yscale('log'); ax[0, 0].grid(True, alpha=0.3); ax[0, 0].legend(fontsize=8)

    for o in order:                                  # B: % energy served per year
        ax[0, 1].plot(yrs, [srv_by_yr.loc[y, o] for y in yrs], marker='o', label=o.split()[0])
    ax[0, 1].set_title('Energy served over project life')
    ax[0, 1].set_xlabel('year'); ax[0, 1].set_ylabel('% of demand served')
    ax[0, 1].grid(True, alpha=0.3); ax[0, 1].legend(fontsize=8)

    for o in order:                                  # C: SoH decay
        ax[0, 2].plot(yrs, [soh_by_yr.loc[y, o] for y in yrs], marker='o', label=o.split()[0])
    ax[0, 2].axhline(EOL * 100, ls='--', color='red', lw=1, label='EOL 80%')
    ax[0, 2].set_title('Battery state of health over project life')
    ax[0, 2].set_xlabel('year'); ax[0, 2].set_ylabel('SoH (%)')
    ax[0, 2].grid(True, alpha=0.3); ax[0, 2].legend(fontsize=8)

    # D: lifetime % energy served (HIGHLIGHT)
    ax[1, 0].bar(short, [life.loc[o, 'Served_life_pct'] for o in order], color=cols)
    ax[1, 0].set_title('Lifetime energy served (higher better)')
    ax[1, 0].set_ylabel('% of demand served'); ax[1, 0].set_ylim(0, 105)
    ax[1, 0].grid(True, axis='y', alpha=0.3)
    ax[1, 0].tick_params(axis='x', labelrotation=20, labelsize=8)
    for i, o in enumerate(order):
        ax[1, 0].text(i, life.loc[o, 'Served_life_pct'],
                      f"{life.loc[o,'Served_life_pct']:.1f}%", ha='center', va='bottom', fontsize=8)

    ax[1, 1].bar(short, [lifespan[o] for o in order], color=cols)   # E: lifespan
    ax[1, 1].set_title('Estimated battery lifespan (years to 80% SoH)')
    ax[1, 1].set_ylabel('years'); ax[1, 1].grid(True, axis='y', alpha=0.3)
    ax[1, 1].tick_params(axis='x', labelrotation=20, labelsize=8)
    for i, o in enumerate(order):
        ax[1, 1].text(i, lifespan[o], f"{lifespan[o]:.1f}", ha='center', va='bottom', fontsize=8)

    x = np.arange(len(order)); w = 0.2              # F: improvement vs baseline
    ax[1, 2].bar(x - 1.5 * w, [srv_gain[o] for o in order], w, label='Energy served gain (pts)', color='#264653')
    ax[1, 2].bar(x - 0.5 * w, [ens_red[o]  for o in order], w, label='ENS reduction %',          color='#2a9d8f')
    ax[1, 2].bar(x + 0.5 * w, [lolp_red[o] for o in order], w, label='LOLP reduction %',          color='#8ab17d')
    ax[1, 2].bar(x + 1.5 * w, [life_ext[o] for o in order], w, label='Lifespan extension %',      color='#e9c46a')
    ax[1, 2].axhline(0, color='k', lw=0.8)
    ax[1, 2].set_title(f'Improvement vs {base.split()[0]} (naive dispatch)')
    ax[1, 2].set_xticks(x); ax[1, 2].set_xticklabels(short, rotation=20, fontsize=8)
    ax[1, 2].set_ylabel('improvement'); ax[1, 2].grid(True, axis='y', alpha=0.3)
    ax[1, 2].legend(fontsize=7)

    fig2.suptitle('Phase 5 — lifetime & improvement summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/phase5_lifetime_summary.png', dpi=150)

    # ====================================================================
    # FIGURE 3b — standalone improvement-vs-baseline (naive Simple dropped)
    # ====================================================================
    to_imp = [o for o in order if o != base]
    imp_series = []
    if 'Served_pct' in df.columns:
        imp_series.append(('Energy served gain (pts)', srv_gain, '#264653'))
    imp_series += [('ENS reduction %', ens_red, '#2a9d8f'),
                   ('LOLP reduction %', lolp_red, '#8ab17d'),
                   ('Lifespan extension %', life_ext, '#e9c46a')]
    xi = np.arange(len(to_imp)); wI = 0.8 / len(imp_series)
    offs = np.linspace(-(len(imp_series) - 1) / 2, (len(imp_series) - 1) / 2, len(imp_series)) * wI
    figI, axI = plt.subplots(figsize=(11, 7))
    for (lbl, vals, c), off in zip(imp_series, offs):
        barsI = axI.bar(xi + off, [vals[o] for o in to_imp], wI, label=lbl, color=c)
        for b, o in zip(barsI, to_imp):
            axI.annotate(f"{vals[o]:.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                         ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')
    axI.axhline(0, color='k', lw=0.8)
    axI.set_xticks(xi); axI.set_xticklabels([o.split()[0] for o in to_imp], fontsize=11)
    axI.set_ylabel('Improvement', fontsize=12)
    axI.set_title(f'Improvement over {base.split()[0]} (naive dispatch)', fontsize=14, fontweight='bold', pad=12)
    axI.grid(True, axis='y', alpha=0.3)
    axI.legend(fontsize=10, ncol=len(imp_series), loc='upper center', bbox_to_anchor=(0.5, -0.09),
               frameon=False, columnspacing=1.8, handlelength=1.3, handletextpad=0.5)
    plt.savefig(f'{OUT_DIR}/phase5_improvement.png', dpi=150, bbox_inches='tight')
    # ====================================================================
    hm_metrics = [                       # (label, column, lower_is_better)
        ('Energy served', 'Served_pct',  False),
        ('ENS',           'ENS_kwh',     True),
        ('LOLP',          'LOLP_pct',    True),
        ('EFC',           'EFC_cycles',  True),
        ('SoH final',     'SOH_pct',     False),
        ('SoC stability', 'SOC_std_pct', True),
    ]
    M = np.zeros((len(order), len(hm_metrics) + 1))
    for j, (lbl, col, lower) in enumerate(hm_metrics):
        b = summary.loc[base, col]
        for i, o in enumerate(order):
            v = summary.loc[o, col]
            M[i, j] = (100.0 * (1 - v / b) if lower else 100.0 * (v / b - 1)) if b else 0.0
    for i, o in enumerate(order):                       # lifespan (higher better)
        M[i, -1] = 100.0 * (lifespan[o] / lifespan[base] - 1)
    hm_labels = [m[0] for m in hm_metrics] + ['Lifespan']

    fig3, axh = plt.subplots(figsize=(1.5 * len(hm_labels) + 2, 0.7 * len(order) + 2))
    im = axh.imshow(M, cmap='RdYlGn', vmin=-100, vmax=100, aspect='auto')
    axh.set_xticks(range(len(hm_labels))); axh.set_xticklabels(hm_labels, rotation=25, ha='right', fontsize=9)
    axh.set_yticks(range(len(order)));     axh.set_yticklabels(short, fontsize=9)
    for i in range(len(order)):
        for j in range(len(hm_labels)):
            axh.text(j, i, f"{M[i, j]:.0f}", ha='center', va='center', fontsize=8)
    axh.set_title(f'% improvement over {base.split()[0]}  (green = better, capped at \u00b1100 for colour)',
                  fontsize=11)
    fig3.colorbar(im, ax=axh, label='% improvement', shrink=0.85)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/phase5_heatmap.png', dpi=150)

    # ====================================================================
    # FIGURE 4 — reliability vs longevity tradeoff (two reliability framings:
    # energy-served on the left, LOLP on the right). The PPO money-graph.
    # The naive baseline is dropped: at ~50% served / ~50% LOLP it stretches the
    # axes so the deployable controllers collapse into an unreadable cluster.
    # ====================================================================
    to   = [o for o in order if o != base]
    tcol = [colors[order.index(o)] for o in to]
    ys   = [lifespan[o] for o in to]
    fig4, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    xs1 = [life.loc[o, 'Served_life_pct'] for o in to]            # served % axis
    axL.scatter(xs1, ys, s=180, c=tcol, zorder=3, edgecolor='k', linewidth=0.6)
    for o, xx, yy in zip(to, xs1, ys):
        axL.annotate(f"{o.split()[0]}\n{yy:.1f} yr", (xx, yy),
                     textcoords='offset points', xytext=(8, 4), fontsize=9)
    axL.set_xlabel('Lifetime energy served (%)   \u2192  more reliable')
    axL.set_ylabel('Estimated lifespan (yrs to 80% SoH)   \u2192  longer-lived')
    axL.set_title('Energy served vs longevity  (upper-right = best)')
    axL.grid(True, alpha=0.3)

    xs2 = [life.loc[o, 'LOLP_life_pct'] for o in to]              # LOLP axis
    axR.scatter(xs2, ys, s=180, c=tcol, zorder=3, edgecolor='k', linewidth=0.6)
    for o, xx, yy in zip(to, xs2, ys):
        axR.annotate(f"{o.split()[0]}\n{xx:.2f}%", (xx, yy),
                     textcoords='offset points', xytext=(8, 4), fontsize=9)
    axR.invert_xaxis()      # lower LOLP is better -> keep "good = right" like the left panel
    axR.set_xlabel('Lifetime LOLP (%)   \u2190  more reliable (axis inverted)')
    axR.set_ylabel('Estimated lifespan (yrs to 80% SoH)   \u2192  longer-lived')
    axR.set_title('LOLP vs longevity  (upper-right = best)')
    axR.grid(True, alpha=0.3)

    fig4.suptitle(f'Reliability vs battery longevity  (competent controllers; naive '
                  f'{base.split()[0]} excluded: {life.loc[base,"Served_life_pct"]:.1f}% served, '
                  f'{lifespan[base]:.1f} yr)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/phase5_tradeoff.png', dpi=150)

    # ====================================================================
    # FIGURE 5 — fleet robustness: distribution of each KPI across the
    # site x size cells (does the result hold fleet-wide, not just one cell?)
    # ====================================================================
    cell_last = (df.sort_values('year')
                   .groupby(['location', 'size_kwp', 'controller'], as_index=False).last())
    cell_last['lifespan'] = (SOH0 - EOL) / (
        ((SOH0 - cell_last['SOH_pct'] / 100.0) / max(n_life, 1)).clip(lower=1e-6))
    box_metrics = [('Lifetime energy served (%)', 'Served_cum_pct'),
                   ('Lifetime LOLP (%)',          'LOLP_cum_pct'),
                   ('Final SoH (%)',              'SOH_pct'),
                   ('Battery lifespan (yrs)',     'lifespan')]
    n_cells = cell_last.groupby('controller').size().max()
    fig5, axb = plt.subplots(2, 2, figsize=(14, 10))
    for axx, (lbl, col) in zip(axb.flat, box_metrics):
        data = [cell_last[cell_last.controller == o][col].values for o in order]
        axx.boxplot(data, showmeans=True)
        axx.set_xticks(range(1, len(order) + 1)); axx.set_xticklabels(short, rotation=20, fontsize=8)
        axx.set_title(f'{lbl}', fontsize=10)
        axx.grid(True, axis='y', alpha=0.3)
    fig5.suptitle(f'Phase 5 — fleet robustness across {cell_last.location.nunique()} loc '
                  f'x {cell_last.size_kwp.nunique()} size ({n_cells} cells/controller)',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/phase5_robustness.png', dpi=150)

    # ====================================================================
    # FIGURE 6 — battery degradation to end-of-life (SoH decay -> 80%);
    # x is battery AGE so the simulated end-of-year points lie on the linear
    # model (year y -> age y+1) and the 80% crossing == the lifespan bars.
    # ====================================================================
    fig6, ax6 = plt.subplots(figsize=(11, 7))
    sim_ages = [y + 1 for y in yrs]
    for o, c in zip(order, colors):
        rate = (SOH0 * 100 - life.loc[o, 'SoH_final_pct']) / max(n_life, 1)   # %/yr
        rate = max(rate, 1e-6)
        L = (100 - EOL * 100) / rate                                          # age at 80%
        ax6.plot(sim_ages, [soh_by_yr.loc[y, o] for y in yrs], '-o', color=c, lw=2, ms=5,
                 label=f"{o.split()[0]}  (\u2212{rate:.2f}%/yr \u2192 {L:.1f} yr)")
        t_ext = np.linspace(sim_ages[-1], L, 50)
        ax6.plot(t_ext, 100 - rate * t_ext, '--', color=c, lw=1.5, alpha=0.9)
        ax6.plot(L, EOL * 100, 'X', color=c, ms=11, mec='k', mew=0.6, zorder=5)
        ax6.annotate(f"{L:.1f} yr", (L, EOL * 100), textcoords='offset points',
                     xytext=(0, -16), ha='center', fontsize=8, color=c, fontweight='bold')
    ax6.axhline(EOL * 100, ls='--', color='red', lw=1.3, label='End of life (80% SoH)')
    ax6.axvspan(1, n_life, color='grey', alpha=0.08)
    ax6.set_xlabel('Battery age (years)', fontsize=11)
    ax6.set_ylabel('Battery state of health (%)', fontsize=11)
    ax6.set_title('Battery degradation to end-of-life — lifespan = 80% crossing',
                  fontsize=13, fontweight='bold')
    ax6.set_ylim(EOL * 100 - 2, 101)
    ax6.grid(True, alpha=0.3); ax6.legend(fontsize=9, loc='lower left')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/phase5_lifespan_decay.png', dpi=150)

    print(f"\nSaved: {OUT_DIR}/phase5_per_episode.csv")
    print(f"Saved: {OUT_DIR}/phase5_by_site.csv")
    print(f"Saved: {OUT_DIR}/phase5_summary.csv")
    print(f"Saved: {OUT_DIR}/phase5_lifetime.csv")
    print(f"Saved: {OUT_DIR}/phase5_summary.png")
    for col in ['Served_pct', 'ENS_kwh', 'LOLP_pct', 'EFC_cycles', 'SOH_pct', 'SOC_std_pct']:
        print(f"Saved: {OUT_DIR}/phase5_kpi_{kpi_file[col]}.png")
    print(f"Saved: {OUT_DIR}/phase5_lifetime_summary.png")
    print(f"Saved: {OUT_DIR}/phase5_improvement.png")
    print(f"Saved: {OUT_DIR}/phase5_heatmap.png")
    print(f"Saved: {OUT_DIR}/phase5_tradeoff.png")
    print(f"Saved: {OUT_DIR}/phase5_robustness.png")
    print(f"Saved: {OUT_DIR}/phase5_lifespan_decay.png")
    print("\n" + "=" * 64)
    print("  PHASE 5 COMPLETE")
    print("=" * 64)


if __name__ == '__main__':
    main()
