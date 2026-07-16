#!/usr/bin/env python
# coding: utf-8
"""
Phase 3 — Parameterised Mini-Grid RL Environment
=================================================
Defines SREPMiniGridEnv: a Gymnasium environment that simulates a
solar + battery mini-grid for the Ghana SREP fleet.

Parameters sampled at each reset (or fixed for evaluation):
  location     : Tamale | Kumasi | Axim
  solar_kwp    : 50 | 75 | 120  (SREP tier)
  battery_kwh  : 200 | 300 | 480  (matched to tier)
  mean_load_kw : 5 to 40  (community size range)

Observation (57-dim):
  soc, soh,
  hour_sin, hour_cos, month_sin, month_cos,
  lstm_solar_24h (24 steps, normalised),
  lstm_load_24h  (24 steps, normalised),
  solar_kwp_norm, battery_kwh_norm, mean_load_kw_norm

Action (1-dim, continuous [-1, 1]):
  -1 = max discharge, +1 = max charge

Reward per step (merged reliability + reserve + longevity design):
  + W_SERVED   * fraction of load served          (ENS  : serve load)
  - W_UNMET    * fraction of load unmet           (ENS  : shortfall size)
  - W_LOLP     * 1 if any load unmet this hour    (LOLP : shortfall frequency)
  + W_RESERVE  * max(0, SoC-SOC_RESERVE)*served   (self-sufficiency reserve)
  - SoC band guards (high / low / floor)          (operating-band keeping)
  - W_DEGRADE  * normalised SoH loss              (SoH  : physics health)
  - W_EFC      * cycle throughput                 (EFC  : cycling)
  - W_CYCLE_WASTE * cycle_depth*(1-served)        (anti over-discharge)
  - W_DOD      * daily DoD above DOD_CAP          (cycle-life cap, once/day)
  - W_CURTAIL  * fraction of solar curtailed      (renewable utilisation)
  reward is clipped to REWARD_CLIP for PPO stability.
  Reliability dominates; a high-SoC reserve is favoured for self-sufficiency.

Inputs:
  ../data/master_dataset_raw.csv
  ../models/best_lstm_param.pth
  ../models/scaler_X_param.pkl
  ../models/scaler_y_param.pkl

Run directly to verify with random + rule-based agents and save plots.

Run order:
  python phase1_build_dataset_param.py
  python phase2_train_lstm_param.py
  python phase3_parametrised_env.py    <-- you are here
  python phase4_ppo_training_param.py
  python phase5_evaluation_param.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

warnings.filterwarnings('ignore')

print("=" * 60)
print("  PHASE 3 — PARAMETERISED MINI-GRID ENVIRONMENT")
print("=" * 60)

# ============================================================
# LSTM ARCHITECTURE (must match Phase 2 exactly)
# ============================================================

class MiniGridLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, forecast, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.forecast    = forecast
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, forecast * output_size)

    def forward(self, x):
        h0  = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0  = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :]).view(-1, self.forecast, self.output_size)


# ============================================================
# ENVIRONMENT
# ============================================================

class SREPMiniGridEnv(gym.Env):
    """
    Parameterised Gymnasium environment for Ghana SREP solar mini-grids.

    Supports three SREP site tiers and a continuous range of community
    load sizes.  A single PPO policy trained on this environment can
    control any site in the fleet without retraining.
    """

    metadata = {'render_modes': []}

    # Ghana SREP site tiers: (solar_kwp, battery_kwh, design_mean_load_kw)
    # Solar matches the real SREP fleet (11x50, 11x75, 13x120 kWp). Battery is
    # sized for ~1 day of autonomy at the design load, matching the observed
    # Lake Volta island mini-grids (~0.9 d at max load):
    #     battery_kwh ~= 1 * design_load_kw * 24 / (SOC_MAX - SOC_MIN)
    # Design load assumes solar generates ~1.4x daily load energy at a Volta
    # region ~5 peak-sun-hours with PV_DERATING. Verify autonomy:
    #     0.85 * 160 / (5.6*24) = 1.01 d   (50 kWp)
    #     0.85 * 237 / (8.4*24) = 1.00 d   (75 kWp)
    #     0.85 * 378 / (13.4*24)= 1.00 d   (120 kWp)
    TIERS = [(50, 160, 5.6), (75, 237, 8.4), (120, 378, 13.4)]

    # Load scatter around the tier design load (community-size variation on
    # standardised hardware); actual autonomy then ranges ~0.8-1.25 days.
    LOAD_SCATTER   = (0.80, 1.20)
    ORIG_MEAN_LOAD = 192.9   # Nigeria baseline (kW)

    # Battery physics
    ETA_CHARGE    = 0.95   # charge efficiency
    ETA_DISCHARGE = 0.95   # discharge efficiency
    SOC_MIN       = 0.10   # minimum allowed SoC
    SOC_MAX       = 0.95   # maximum allowed SoC
    SOC_INIT      = 0.50   # starting SoC each episode
    SOH_INIT      = 1.00   # starting SoH each episode
    SOH_FLOOR     = 0.50   # battery replaced below this SoH
    N_CYCLES      = 4000   # lifetime full cycles (20 % degradation at end)

    # PV derating (soiling, temperature, mismatch losses)
    PV_DERATING   = 0.75

    # Normalisation bounds for site params in observation
    MAX_SOLAR_KWP = 120.0
    MAX_BAT_KWH   = 400.0
    MAX_LOAD_KW   = 20.0

    # ---- Residual RL ----------------------------------------------------
    # The policy output is a CORRECTION added to the load-following baseline,
    # not the absolute battery power:
    #     act = clip(rule_based_action + RESIDUAL_SCALE * policy_output, -1, 1)
    # At policy_output = 0 the agent reproduces the baseline EXACTLY, so the
    # baseline's ~2-3% LOLP becomes the floor it starts from rather than a
    # target it must rediscover. PPO then only learns small foresight
    # corrections (e.g. rationing discharge ahead of a forecast cloudy spell).
    # This sidesteps the ~16% precision floor an absolute-action policy hit.
    # NOTE: incompatible with the BC/DAgger warm-start (which clones ABSOLUTE
    # actions) -- run residual as a COLD start; the architecture itself starts
    # the policy at the baseline. Set RESIDUAL=False to restore the old env.
    RESIDUAL       = True
    RESIDUAL_SCALE = 0.25   # max correction magnitude around the baseline

    # LSTM window
    LOOKBACK   = 24       # hours of history fed to the LSTM
    FORECAST   = 48       # hours the LSTM predicts ahead (2 days)
    OBS_HOURLY = 24       # hours of hourly detail exposed in the observation
    # Hours 24:48 of the forecast are compressed to a 2-number day-2 summary
    # (mean solar, mean load) — captures "is a cloudy spell coming?" without
    # feeding the policy 24 noisy far-horizon hourly values.

    # LSTM input features (must match Phase 2)
    LSTM_FEATURES = ['ssrd_wm2', 'tp', 'temp_c', 'load_kw',
                     'hour', 'month', 'dayofweek']   # Option 1: no location_code
    LSTM_TARGETS  = ['ssrd_wm2', 'load_kw']  # (solar W/m2, load kW)

    # Reward weights — merged design:
    #   reliability-first (ENS/LOLP) + high-SoC reserve for self-sufficiency
    #   (no diesel) + physics-based longevity (degrade_cost/EFC) + daily-DoD cap.
    # Reliability (dominant)
    W_SERVED      = 2.5    # ENS: reward fraction of load served
    W_LOLP        = 3.5    # LOLP: flat penalty for ANY unmet load this hour
    W_UNMET       = 4.0    # ENS: penalty scaled to shortfall magnitude
    # SoC reserve / operating band (self-sufficiency), repositioned to 0.10-0.95
    W_RESERVE     = 0.0    # REMOVED: a high-SoC bonus rewarded hoarding and added
    #                        ruggedness; reliability + curtail terms drive recharge.
    W_SOC_HIGH    = 0.5    # reduced 2.0->0.5 (daytime fill-up is necessary; physics
    #                        degradation term already captures high-SoC calendar ageing)
    W_SOC_LOW     = 0.0    # REMOVED: the 0.10-0.20 band is legitimate operation for a
    #                        1-day battery; SoC is already hard-clipped at SOC_MIN=0.10
    W_SOC_FLOOR   = 2.0    # reduced 10.0->2.0: a gentle nudge at the physical floor only
    SOC_RESERVE   = 0.40   # (unused now that W_RESERVE=0)
    SOC_HIGH      = 0.90   # above this -> high-SoC ageing penalty
    SOC_LOW       = 0.20   # (unused now that W_SOC_LOW=0)
    SOC_FLOOR     = 0.11   # lowered 0.15->0.11: penalise only true over-discharge
    #   NOTE: the steep guards above were taxing the optimal deep-cycling policy by
    #   ~9000/episode vs only ~1800 in actual reliability penalties, which FLATTENED
    #   the reward gradient (10%->3% LOLP gained only +395). Softening them restores a
    #   strong serving gradient (+4868 for the same improvement). Validated in-env.
    # Battery longevity
    W_DEGRADE     = 0.5    # SoH: physics-based per-step health loss
    W_EFC         = 0.05   # EFC: light cycle-throughput term (overlaps degrade)
    W_DOD         = 1.5    # reduced 3.0->1.5: allow near-daily deep cycling
    DOD_CAP       = 0.85   # raised 0.75->0.85: a 1-day battery swings most of its band
    W_CYCLE_WASTE = 2.0    # penalise cycling that did NOT serve load (anti-over-discharge)
    # Renewable utilisation
    W_CURTAIL     = 0.5    # penalise wasted solar
    # Per-step reward bound (prevents any single step dominating PPO targets)
    REWARD_CLIP   = (-15.0, 5.0)

    LOL_EPS    = 1e-3      # below this unmet (kWh) an hour is NOT loss-of-load

    def __init__(self,
                 data_path='../data/master_dataset_raw.csv',
                 model_dir='../models',
                 episode_len=8760,
                 seed=None):
        super().__init__()

        # ---- load dataset ----
        self.df = pd.read_csv(data_path)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        # Load EVERY location present (so held-out eval sites can be requested
        # via options). PPO training still samples only TRAIN_LOCS (see _sample_site).
        self._loc_frames = {
            loc: self.df[self.df['location'] == loc].reset_index(drop=True)
            for loc in self.df['location'].unique()
        }

        # ---- load LSTM + scalers ----
        with open(f'{model_dir}/scaler_X_param.pkl', 'rb') as fh:
            self.scaler_X = pickle.load(fh)
        with open(f'{model_dir}/scaler_y_param.pkl', 'rb') as fh:
            self.scaler_y = pickle.load(fh)

        self.lstm = MiniGridLSTM(7, 128, 2, self.FORECAST, 2)
        self.lstm.load_state_dict(torch.load(
            f'{model_dir}/best_lstm_param.pth',
            map_location='cpu', weights_only=True))
        self.lstm.eval()

        self.episode_len = episode_len
        self.rng = np.random.default_rng(seed)

        # ---- spaces ----
        # obs: soc, soh, hour_sin, hour_cos, month_sin, month_cos,
        #       24 solar forecast, 24 load forecast, 3 site params
        # obs: soc,soh (2) + time (4) + 24h solar + 24h load + day-2 summary (2) + site params (3)
        obs_dim = (2 + 4 + self.OBS_HOURLY + self.OBS_HOURLY + 2 + 3
                   + (1 if self.RESIDUAL else 0))   # +1: baseline-action reference
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32)

        # action: normalised charge rate in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # initialise placeholders
        self.loc_df       = self._loc_frames['Tamale']
        self.solar_kwp    = 120.0
        self.battery_kwh  = 378.0
        self.mean_load_kw = 13.4
        self.load_scale   = self.mean_load_kw / self.ORIG_MEAN_LOAD
        self.max_power_kw = 48.0
        self.lifetime_kwh = self.N_CYCLES * self.battery_kwh * 2
        self.t            = self.LOOKBACK
        self.soc          = self.SOC_INIT
        self.soh          = self.SOH_INIT
        self.step_count   = 0
        self._reset_accumulators()

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _reset_accumulators(self):
        self.ep_solar_kwh  = 0.0
        self.ep_load_kwh   = 0.0
        self.ep_served_kwh = 0.0
        self.ep_curtail_kwh= 0.0
        self.ep_unmet_kwh  = 0.0
        self.ep_efc        = 0.0   # equivalent full cycles
        self.ep_lol_hours  = 0     # hours with any unmet load (for LOLP)
        self.ep_steps      = 0     # total hours stepped (for LOLP denominator)
        self.ep_soc_sum    = 0.0   # running sum of SoC      (for SoC std)
        self.ep_soc_sq     = 0.0   # running sum of SoC^2    (for SoC std)
        self._day_soc_max  = self.SOC_INIT   # for daily depth-of-discharge
        self._day_soc_min  = self.SOC_INIT

    def _sample_site(self, options):
        """Apply options dict or sample randomly."""
        opts = options or {}

    def _sample_site(self, options):
        """Apply options dict or sample a coherent SREP-sized site.

        Sites are no longer sampled with battery and load independent. We pick
        a real solar tier (50/75/120 kWp), take its matched ~1-day-autonomy
        battery, then draw the community load by scattering around that tier's
        design load. Any field can still be overridden via `options`.
        """
        opts = options or {}

        # --- Solar tier + matched 1-day-autonomy battery ---
        if 'solar_kwp' in opts:
            self.solar_kwp = float(opts['solar_kwp'])
            tier = min(self.TIERS, key=lambda t: abs(t[0] - self.solar_kwp))
        else:
            tier = self.TIERS[int(self.rng.integers(len(self.TIERS)))]
            self.solar_kwp = float(tier[0])
        self.battery_kwh = float(opts.get('battery_kwh', tier[1]))
        design_load = float(tier[2])

        # --- Community load: scattered around the tier design load ---
        if 'mean_load_kw' in opts:
            self.mean_load_kw = float(opts['mean_load_kw'])
        else:
            lo, hi = self.LOAD_SCATTER
            self.mean_load_kw = float(design_load * self.rng.uniform(lo, hi))
        self.load_scale = self.mean_load_kw / self.ORIG_MEAN_LOAD

        # --- Location ---
        locs = ['Tamale', 'Kumasi', 'Axim']
        loc  = opts.get('location', str(self.rng.choice(locs)))
        self.location = loc
        self.loc_df   = self._loc_frames[loc]

        # --- Derived ---
        # Max power: 0.5C rate, capped at solar peak (inverter sizing)
        self.max_power_kw = min(self.battery_kwh * 0.5, self.solar_kwp * 0.8)
        # Lifetime throughput: N_cycles × capacity × 2 (charge + discharge)
        self.lifetime_kwh = self.N_CYCLES * self.battery_kwh * 2.0

    def _get_lstm_forecast(self):
        """Return (solar_fc, load_fc) each shape (FORECAST,) = (48,), normalised [0,1]."""
        window = self.loc_df.iloc[self.t - self.LOOKBACK:self.t].copy()
        window['load_kw'] = window['load_kw'] * self.load_scale

        X = window[self.LSTM_FEATURES].values.astype(np.float32)
        X_scaled = self.scaler_X.transform(X)         # (24, 7)

        with torch.no_grad():
            inp = torch.FloatTensor(X_scaled).unsqueeze(0)  # (1, 24, 7)
            out = self.lstm(inp).squeeze(0).numpy()          # (48, 2)

        # out[:,0] = normalised solar, out[:,1] = normalised load
        return np.clip(out[:, 0], 0.0, 1.0), np.clip(out[:, 1], 0.0, 1.0)

    def _build_obs(self):
        row      = self.loc_df.iloc[self.t]
        hour     = int(row['hour'])
        month    = int(row['month'])
        solar_fc, load_fc = self._get_lstm_forecast()   # each length FORECAST (48)
        h = self.OBS_HOURLY
        # day-2 summary: mean normalised solar / load over forecast hours 24:48
        day2_solar = float(solar_fc[h:].mean()) if len(solar_fc) > h else float(solar_fc.mean())
        day2_load  = float(load_fc[h:].mean())  if len(load_fc)  > h else float(load_fc.mean())

        parts = [
            [self.soc, self.soh],
            [np.sin(2*np.pi*hour/24),  np.cos(2*np.pi*hour/24)],
            [np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)],
            solar_fc[:h],
            load_fc[:h],
            [day2_solar, day2_load],
            [self.solar_kwp   / self.MAX_SOLAR_KWP,
             self.battery_kwh / self.MAX_BAT_KWH,
             self.mean_load_kw/ self.MAX_LOAD_KW],
        ]
        if self.RESIDUAL:
            # the load-following reference the policy is correcting
            parts.append([float(rule_based_action(self).flat[0])])
        obs = np.concatenate(parts, dtype=np.float32)

        return obs

    def _compute_degradation(self, temp_c, action):
        """
        LiFePO4 degradation — 3 mechanisms:

        1. Rainflow cycle aging [Xu et al. 2016]
           L(DOD) = 3500 × (1/DOD)^1.5
           Each direction reversal closes a half-cycle.

        2. Calendar aging — Arrhenius + asymmetric SOC stress [Wang et al. 2014]
           Cell temp = ambient + 5 °C enclosure + |action| × 3 °C self-heating.
           SOC stress above 50% punished harder (cathode oxidation).

        3. Low-SOC lithium plating below 30% SOC [Schmalstieg et al. 2014]
        """
        usable = self.SOC_MAX - self.SOC_MIN

        # ── 1. Rainflow cycle aging ───────────────────────────────
        soc_change = self.soc - self._prev_soc
        deg_cycle  = 0.0
        if abs(soc_change) > 1e-4:
            new_dir = 1 if soc_change > 0 else -1
            if new_dir != self._rf_direction and self._rf_direction != 0:
                half_dod = abs(self.soc - self._rf_half_start) / usable
                if half_dod > 1e-4:
                    cycle_life = 3500.0 * (1.0 / max(half_dod, 0.01)) ** 1.5
                    deg_cycle  = (0.20 / cycle_life) * 0.5
                    self.ep_efc += half_dod * 0.5
                self._rf_half_start = self.soc
            self._rf_direction = new_dir
        self._prev_soc = self.soc

        # ── 2. Calendar aging (Arrhenius + cell temp + asymmetric SOC) ──
        cell_temp = temp_c + 5.0 + abs(action) * 3.0
        T_kelvin  = max(cell_temp + 273.15, 273.15)
        k_cal     = 14876.0 * np.exp(-24500.0 / (8.314 * T_kelvin))
        if self.soc >= 0.50:
            soc_stress = 0.70 + 0.60 * (self.soc - 0.50) ** 2
        else:
            soc_stress = 0.70 + 0.30 * (self.soc - 0.50) ** 2
        # LFP has higher activation energy (~31 kJ/mol vs 24.5 for NMC) and
        # slower calendar fade; pre-factor reduced accordingly
        deg_cal = 6e-7 * k_cal * soc_stress

        # ── 3. Low-SOC lithium plating (LFP-adjusted — LFP is far more
        #    tolerant of deep discharge than NMC; coefficient scaled down
        #    from Schmalstieg 2014 to match LFP empirical data)
        deg_low_soc = 4e-5 * max(0.0, 0.30 - self.soc) ** 2

        return deg_cycle + deg_cal + deg_low_soc

    # ----------------------------------------------------------
    # Gymnasium API
    # ----------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._sample_site(options)

        # Choose a random start inside the location data
        max_start = len(self.loc_df) - self.episode_len - 1
        min_start = self.LOOKBACK
        if max_start <= min_start:
            self.t = min_start
        else:
            self.t = int(self.rng.integers(min_start, max_start))

        self.soc        = self.SOC_INIT
        self.soh        = self.SOH_INIT
        self.step_count = 0
        self._rf_direction  = 0
        self._rf_half_start = self.SOC_INIT
        self._prev_soc      = self.SOC_INIT
        self._reset_accumulators()

        return self._build_obs(), {}

    def step(self, action):
        row      = self.loc_df.iloc[self.t]
        solar_kw = (row['ssrd_wm2'] / 1000.0) * self.solar_kwp * self.PV_DERATING
        load_kw  = row['load_kw'] * self.load_scale
        soc_before = self.soc   # for per-step cycle depth

        # ---- battery action ----
        if self.RESIDUAL:
            # policy output is a CORRECTION to the load-following baseline
            base  = float(rule_based_action(self).flat[0])
            delta = float(np.asarray(action).flat[0]) * self.RESIDUAL_SCALE
            act   = float(np.clip(base + delta, -1.0, 1.0))
        else:
            act = float(np.clip(np.asarray(action).flat[0], -1.0, 1.0))
        desired_power = act * self.max_power_kw   # kW; + = charge, - = discharge

        eff_cap      = self.battery_kwh * self.soh
        stored_energy = self.soc * eff_cap

        if desired_power >= 0.0:
            free_cap      = (self.SOC_MAX - self.soc) * eff_cap
            max_charge_kw = free_cap / self.ETA_CHARGE
            battery_power = min(desired_power, max_charge_kw)
        else:
            avail_energy  = (self.soc - self.SOC_MIN) * eff_cap
            max_disch_kw  = avail_energy * self.ETA_DISCHARGE
            battery_power = max(desired_power, -max_disch_kw)

        # ---- bus balance (kWh over 1-hour step) ----
        supply = solar_kw + max(0.0, -battery_power)   # solar + discharge
        demand = load_kw  + max(0.0,  battery_power)   # load  + charge

        if supply >= demand:
            served_kwh  = load_kw
            curtail_kwh = supply - demand
            unmet_kwh   = 0.0
        else:
            gap         = demand - supply
            unmet_kwh   = min(gap, load_kw)            # load is the only "flexible" sink
            served_kwh  = load_kw - unmet_kwh
            curtail_kwh = 0.0

        # ---- SoC update ----
        actual_charge    = max(0.0,  battery_power)
        actual_discharge = max(0.0, -battery_power)
        delta_energy     = (actual_charge * self.ETA_CHARGE
                           - actual_discharge / self.ETA_DISCHARGE)
        new_energy       = stored_energy + delta_energy
        self.soc         = float(np.clip(new_energy / eff_cap,
                                         self.SOC_MIN, self.SOC_MAX))

        # ---- SoH degradation ----
        efc_before = self.ep_efc
        delta_soh = self._compute_degradation(row['temp_c'], act)
        self.soh  = float(np.clip(self.soh - delta_soh, self.SOH_FLOOR, 1.0))
        efc_increment = self.ep_efc - efc_before   # half-cycles closed this step

        # ---- accumulators ----
        self.ep_solar_kwh   += solar_kw
        self.ep_load_kwh    += load_kw
        self.ep_served_kwh  += served_kwh
        self.ep_curtail_kwh += curtail_kwh
        self.ep_unmet_kwh   += unmet_kwh
        self.ep_steps       += 1
        if unmet_kwh > self.LOL_EPS:
            self.ep_lol_hours += 1
        self.ep_soc_sum     += self.soc
        self.ep_soc_sq      += self.soc * self.soc
        self._day_soc_max    = max(self._day_soc_max, self.soc)
        self._day_soc_min    = min(self._day_soc_min, self.soc)

        # ---- reward (merged: reliability + SoC reserve + longevity) ----
        served_frac  = served_kwh  / max(load_kw,  0.01)
        unmet_frac   = unmet_kwh   / max(load_kw,  0.01)
        # Curtailment normalised by dispatched supply (solar + discharge) so it
        # stays in [0,1] (dividing by solar alone explodes at night).
        curtail_frac = curtail_kwh / max(supply, 0.01)
        lol_flag     = 1.0 if unmet_kwh > self.LOL_EPS else 0.0

        # SoH: physics health loss normalised against one nominal full cycle.
        one_cycle_deg = 0.20 / self.N_CYCLES
        degrade_cost  = delta_soh / max(one_cycle_deg, 1e-10)

        # per-step cycle depth (SoC moved this hour)
        cycle_depth = abs(self.soc - soc_before)

        reward  = self.W_SERVED * served_frac          # ENS: serve load
        reward -= self.W_LOLP   * lol_flag             # LOLP: any shortfall
        reward -= self.W_UNMET  * unmet_frac           # ENS: shortfall size

        # SoC reserve for self-sufficiency — only rewarded while serving load,
        # so it biases against UNNECESSARY discharge, not against using the
        # battery when load actually needs it.
        reward += self.W_RESERVE * max(0.0, self.soc - self.SOC_RESERVE) * served_frac

        # SoC operating-band guards (repositioned to the 0.10-0.95 band)
        reward -= self.W_SOC_HIGH  * (self.soc > self.SOC_HIGH)
        reward -= self.W_SOC_LOW   * (self.soc < self.SOC_LOW)
        reward -= self.W_SOC_FLOOR * (self.soc < self.SOC_FLOOR)

        # Longevity: physics per-step + light EFC + cycling-that-didn't-serve.
        reward -= self.W_DEGRADE     * degrade_cost
        reward -= self.W_EFC         * efc_increment
        reward -= self.W_CYCLE_WASTE * cycle_depth * (1.0 - served_frac)

        # Daily depth-of-discharge cap (evaluated once per day, then reset).
        daily_dod = self._day_soc_max - self._day_soc_min
        if self.step_count % 24 == 23 and daily_dod > self.DOD_CAP:
            reward -= self.W_DOD * (daily_dod - self.DOD_CAP)
            self._day_soc_max = self.soc
            self._day_soc_min = self.soc

        # Renewable utilisation
        reward -= self.W_CURTAIL * curtail_frac

        # Bound the per-step reward (PPO stability).
        reward = float(np.clip(reward, self.REWARD_CLIP[0], self.REWARD_CLIP[1]))

        # ---- advance ----
        self.t          += 1
        self.step_count += 1

        terminated = self.step_count >= self.episode_len
        truncated  = self.t >= len(self.loc_df) - 1

        done = terminated or truncated
        obs  = (self._build_obs() if not done
                else np.zeros(self.observation_space.shape, dtype=np.float32))

        info = {
            'solar_kw'   : solar_kw,
            'load_kw'    : load_kw,
            'battery_kw' : battery_power,
            'served_kwh' : served_kwh,
            'unmet_kwh'  : unmet_kwh,
            'curtail_kwh': curtail_kwh,
            'soc'        : self.soc,
            'soh'        : self.soh,
            'delta_soh'  : delta_soh,
        }

        return obs, float(reward), terminated, truncated, info

    def get_episode_stats(self):
        """Call after an episode ends for summary metrics."""
        rf = self.ep_served_kwh  / max(self.ep_load_kwh,  1.0)
        cf = self.ep_curtail_kwh / max(self.ep_solar_kwh, 1.0)
        uf = self.ep_unmet_kwh   / max(self.ep_load_kwh,  1.0)
        n  = max(self.ep_steps, 1)
        lolp = self.ep_lol_hours / n
        soc_mean = self.ep_soc_sum / n
        soc_var  = max(self.ep_soc_sq / n - soc_mean ** 2, 0.0)
        soc_std  = soc_var ** 0.5
        return {
            'renewable_fraction'  : round(rf, 4),
            'curtailment_fraction': round(cf, 4),
            'unmet_fraction'      : round(uf, 4),
            'lolp'                : round(lolp, 4),       # Loss-of-load probability
            'soc_std'             : round(soc_std, 4),    # SoC standard deviation
            'final_soh'           : round(self.soh, 6),
            'efc'                 : round(self.ep_efc, 3),
            'solar_kwh'           : round(self.ep_solar_kwh,   1),
            'load_kwh'            : round(self.ep_load_kwh,    1),
            'served_kwh'          : round(self.ep_served_kwh,  1),
            'curtail_kwh'         : round(self.ep_curtail_kwh, 1),
            'unmet_kwh'           : round(self.ep_unmet_kwh,   1),
        }


# ============================================================
# RULE-BASED CONTROLLERS (baselines for comparison)
# ============================================================

def rule_based_action(env):
    """Realistic baseline: solar-first load-following (self-consumption).

    The standard reactive EMS deployed on off-grid solar-battery mini-grids:
      - solar serves the load directly,
      - any surplus charges the battery,
      - any deficit is covered by discharging the battery.
    It uses real-time MEASURED solar and load (no forecast), and the
    environment enforces battery power / SoC / efficiency limits. When the
    battery is empty the shortfall becomes ENS; when it is full the excess
    solar is curtailed. This is a strong, fair baseline: it is handed perfect
    current measurement, so any PPO improvement must come from foresight
    (pre-charging ahead of cloudy spells) and degradation-aware dispatch,
    not from better information about the present.
    """
    row      = env.loc_df.iloc[env.t]
    solar_kw = (row['ssrd_wm2'] / 1000.0) * env.solar_kwp * env.PV_DERATING
    load_kw  = row['load_kw'] * env.load_scale
    net      = solar_kw - load_kw      # + surplus -> charge, - deficit -> discharge
    action   = net / env.max_power_kw  # env clips to [-1, 1] and physical limits
    return np.array([np.clip(action, -1.0, 1.0)], dtype=np.float32)


def heuristic_action(obs, solar_kwp, mean_load_kw):
    """
    Legacy forecast heuristic (kept for reference, not the headline baseline):
      - If LSTM solar forecast for next 6h is high → charge
      - Else → discharge at a fixed rate
    Note: discharges at a fixed rate regardless of load, so it over-discharges
    and is NOT a fair baseline. Use rule_based_action for comparisons.
    """
    solar_fc = obs[6:30]    # 24h solar forecast (normalised)
    avg_solar_next6 = solar_fc[:6].mean()

    soc   = obs[0]
    # Charge during sunny periods, discharge otherwise
    if avg_solar_next6 > 0.15:
        # Daytime: charge at a rate proportional to forecast surplus
        action = min(1.0, avg_solar_next6 * 2.0)
    else:
        # Night / cloudy: discharge to serve load
        action = -0.5

    return np.array([action], dtype=np.float32)


# ============================================================
# SANITY CHECK — runs when executed as a script
# ============================================================

if __name__ == '__main__':
    import time

    EVAL_STEPS = 24 * 30   # 30 days for a quick check

    # ---- instantiate ----
    print("\nInstantiating SREPMiniGridEnv...")
    env = SREPMiniGridEnv(episode_len=EVAL_STEPS)
    print(f"  Observation space : {env.observation_space.shape}")
    print(f"  Action space      : {env.action_space.shape}")

    # ----------------------------------------------------------
    # 1. RANDOM AGENT
    # ----------------------------------------------------------
    print("\n[1/2] Random agent (30-day episode)...")
    obs, _ = env.reset(seed=42, options={
        'solar_kwp': 75, 'battery_kwh': 237,
        'mean_load_kw': 8.4, 'location': 'Kumasi'
    })
    print(f"  Site  : {env.location}  |  {env.solar_kwp} kWp  |  "
          f"{env.battery_kwh} kWh  |  {env.mean_load_kw:.1f} kW mean load")

    rand_log = {'soc':[], 'solar':[], 'load':[], 'bat':[], 'reward':[]}
    t0 = time.time()

    for _ in range(EVAL_STEPS):
        act = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(act)
        rand_log['soc'].append(info['soc'])
        rand_log['solar'].append(info['solar_kw'])
        rand_log['load'].append(info['load_kw'])
        rand_log['bat'].append(info['battery_kw'])
        rand_log['reward'].append(rew)
        if term or trunc:
            break

    rand_stats = env.get_episode_stats()
    print(f"  Elapsed : {time.time()-t0:.1f}s")
    print(f"  Renewable fraction : {rand_stats['renewable_fraction']:.1%}")
    print(f"  Energy not served  : {rand_stats['unmet_kwh']:.0f} kWh")
    print(f"  Unmet load         : {rand_stats['unmet_fraction']:.1%}")
    print(f"  LOLP               : {rand_stats['lolp']:.1%}")
    print(f"  Curtailment        : {rand_stats['curtailment_fraction']:.1%}")
    print(f"  EFC                : {rand_stats['efc']:.2f} cycles")
    print(f"  Final SoH          : {rand_stats['final_soh']:.4f}")
    print(f"  SoC std            : {rand_stats['soc_std']:.1%}")

    # ----------------------------------------------------------
    # 2. RULE-BASED AGENT (solar-first load-following)
    # ----------------------------------------------------------
    print("\n[2/2] Rule-based agent — load-following (same 30-day episode)...")
    obs, _ = env.reset(seed=42, options={
        'solar_kwp': 75, 'battery_kwh': 237,
        'mean_load_kw': 8.4, 'location': 'Kumasi'
    })

    heur_log = {'soc':[], 'solar':[], 'load':[], 'bat':[], 'reward':[]}

    for _ in range(EVAL_STEPS):
        # Under residual mode the env reads the action as a CORRECTION to the
        # baseline, so a ZERO action reproduces load-following exactly. Stepping
        # rule_based_action directly would double-count to (1+SCALE)x the command.
        act = (np.zeros(1, dtype=np.float32) if env.RESIDUAL
               else rule_based_action(env))
        obs, rew, term, trunc, info = env.step(act)
        heur_log['soc'].append(info['soc'])
        heur_log['solar'].append(info['solar_kw'])
        heur_log['load'].append(info['load_kw'])
        heur_log['bat'].append(info['battery_kw'])
        heur_log['reward'].append(rew)
        if term or trunc:
            break

    heur_stats = env.get_episode_stats()
    print(f"  Renewable fraction : {heur_stats['renewable_fraction']:.1%}")
    print(f"  Energy not served  : {heur_stats['unmet_kwh']:.0f} kWh")
    print(f"  Unmet load         : {heur_stats['unmet_fraction']:.1%}")
    print(f"  LOLP               : {heur_stats['lolp']:.1%}")
    print(f"  Curtailment        : {heur_stats['curtailment_fraction']:.1%}")
    print(f"  EFC                : {heur_stats['efc']:.2f} cycles")
    print(f"  Final SoH          : {heur_stats['final_soh']:.4f}")
    print(f"  SoC std            : {heur_stats['soc_std']:.1%}")

    # ----------------------------------------------------------
    # 3. COMPARISON TABLE
    # ----------------------------------------------------------
    print("\n" + "-" * 54)
    print(f"  {'Metric':<28}  {'Random':>8}  {'Rule-Based':>11}")
    print(f"  {'-'*52}")
    # fmt: 'pct' = percentage, 'kwh' = raw kWh, 'cyc' = cycles, 'soh' = 4-dp
    for key, label, fmt in [
        ('renewable_fraction',   'Renewable fraction', 'pct'),
        ('unmet_kwh',            'Energy not served',  'kwh'),
        ('unmet_fraction',       'Unmet load',         'pct'),
        ('lolp',                 'LOLP',               'pct'),
        ('curtailment_fraction', 'Curtailment',        'pct'),
        ('efc',                  'EFC',                'cyc'),
        ('final_soh',            'Final SoH',          'soh'),
        ('soc_std',              'SoC std',            'pct'),
    ]:
        rv = rand_stats[key]
        hv = heur_stats[key]
        if fmt == 'soh':
            print(f"  {label:<28}  {rv:>8.4f}  {hv:>10.4f}")
        elif fmt == 'kwh':
            print(f"  {label:<28}  {rv:>6.0f}kWh  {hv:>7.0f}kWh")
        elif fmt == 'cyc':
            print(f"  {label:<28}  {rv:>8.2f}  {hv:>10.2f}")
        else:  # pct
            print(f"  {label:<28}  {rv:>7.1%}  {hv:>9.1%}")
    print("-" * 54)

    # ----------------------------------------------------------
    # 4. PLOT — first 7 days
    # ----------------------------------------------------------
    days  = 7
    steps = days * 24
    hours = np.arange(steps)

    fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
    titles_r = ['Random — Solar & Load (kW)', 'Random — Battery Power (kW)',
                'Random — SoC',               'Random — Cumulative Reward']
    titles_h = ['Rule-Based — Solar & Load (kW)', 'Rule-Based — Battery Power (kW)',
                'Rule-Based — SoC',               'Rule-Based — Cumulative Reward']

    for col, (log, titles) in enumerate([(rand_log, titles_r), (heur_log, titles_h)]):
        solar = np.array(log['solar'][:steps])
        load  = np.array(log['load'][:steps])
        bat   = np.array(log['bat'][:steps])
        soc   = np.array(log['soc'][:steps])
        cum_r = np.cumsum(log['reward'][:steps])

        ax = axes[0, col]
        ax.fill_between(hours, solar, alpha=0.4, color='#f59e0b', label='Solar (kW)')
        ax.plot(hours, load, color='#ef4444', linewidth=1.2, label='Load (kW)')
        ax.set_title(titles[0], fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        ax.bar(hours, bat, color=['#22c55e' if b >= 0 else '#3b82f6' for b in bat],
               width=1.0, alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_title(titles[1], fontsize=10)
        ax.set_ylabel('kW')
        ax.grid(True, alpha=0.3)

        ax = axes[2, col]
        ax.plot(hours, soc, color='#6366f1', linewidth=1.5)
        ax.axhline(env.SOC_MIN, color='red',  linewidth=0.8, linestyle='--', label='SoC min')
        ax.axhline(env.SOC_MAX, color='gray', linewidth=0.8, linestyle='--', label='SoC max')
        ax.set_ylim(0, 1.05)
        ax.set_title(titles[2], fontsize=10)
        ax.set_ylabel('SoC')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[3, col]
        ax.plot(hours, cum_r, color='#0ea5e9', linewidth=1.5)
        ax.set_title(titles[3], fontsize=10)
        ax.set_xlabel('Hour')
        ax.set_ylabel('Cumulative reward')
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Phase 3 — Environment Sanity Check  |  '
        f'Kumasi  |  75 kWp  |  237 kWh  |  8.4 kW mean load  |  First {days} days',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    out_path = '../data/phase3_env_sanity_check.png'
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"\nPlot saved: {out_path}")

    # ----------------------------------------------------------
    # 5. MULTI-SITE RESET CHECK
    # ----------------------------------------------------------
    print("\nMulti-site reset check (5 random configs)...")
    print(f"  {'Location':<10} {'Solar':>8} {'Battery':>10} {'Load':>10} {'MaxPwr':>10}")
    print(f"  {'-'*52}")
    for i in range(5):
        obs, _ = env.reset(seed=i)
        print(f"  {env.location:<10} "
              f"{env.solar_kwp:>6.0f} kWp "
              f"{env.battery_kwh:>7.0f} kWh "
              f"{env.mean_load_kw:>7.1f} kW "
              f"{env.max_power_kw:>7.1f} kW")

    # ---- final summary ----
    print("\n" + "=" * 60)
    print("  PHASE 3 COMPLETE")
    print("=" * 60)
    print(f"  Environment : SREPMiniGridEnv")
    print(f"  Obs dim     : {env.observation_space.shape[0]}")
    print(f"  Action dim  : {env.action_space.shape[0]}")
    print(f"  Tiers       : {env.TIERS}")
    print(f"  Load range  : {env.TIERS[0][2]*env.LOAD_SCATTER[0]:.1f}-"
          f"{env.TIERS[-1][2]*env.LOAD_SCATTER[1]:.1f} kW mean (1-day autonomy)")
    print(f"  Locations   : Tamale | Kumasi | Axim")
    print(f"\n  Rule-based baseline (load-following) — the 5 project KPIs:")
    print(f"    Energy not served (ENS) : {heur_stats['unmet_kwh']:.0f} kWh")
    print(f"    Loss of load prob (LOLP): {heur_stats['lolp']:.1%}")
    print(f"    Equiv. full cycles (EFC): {heur_stats['efc']:.2f}")
    print(f"    Final state of health   : {heur_stats['final_soh']:.1%}")
    print(f"    SoC standard deviation  : {heur_stats['soc_std']:.1%}")
    print(f"\nNext: run phase4_ppo_training_param.py")
