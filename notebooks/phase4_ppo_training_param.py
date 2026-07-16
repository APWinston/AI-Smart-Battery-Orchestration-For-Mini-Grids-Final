#!/usr/bin/env python
# coding: utf-8
"""
Phase 4 — PPO Training (Parameterised Fleet Policy)
===================================================
Trains a SINGLE PPO policy on SREPMiniGridEnv that can control any site in
the Ghana SREP fleet (3 tiers x continuous load range x 3 locations) without
retraining. Each environment reset samples a random site, so the policy must
generalise across the whole fleet rather than memorise one configuration.

Why PPO:
  - The action is continuous (battery charge/discharge rate in [-1, 1]).
  - The horizon is long (up to 8760 hourly steps) with delayed consequences
    (charging now to serve load tonight), which PPO's GAE handles well.
  - On-policy stability matters because the reward mixes several competing
    terms (served / unmet / LOLP / curtailment / degradation / SoC band).

Outputs:
  ../models/ppo_srep_param.zip            final policy
  ../models/vecnormalize_param.pkl        final obs/reward normalisation stats
  ../models/best/best_model.zip           best policy by fleet eval reward
  ../models/best/vecnormalize_param.pkl   normalisation stats AT the best model
  ../data/phase4_ppo_training_curve.png   reward + KPI curves
  ../data/phase4_ppo_progress.csv         per-eval metrics

NOTE on in-training KPI numbers: to keep evaluation cheap, the KPI curves are
measured over SHORT episodes (default 90 days), so EFC/ENS read ~1/4 of an
annual figure. They are for watching TRENDS during training. The real annual
KPI table comes from phase5_evaluation_param.py (full 8760-hour episodes).

Run order:
  phase1 -> phase2 -> phase3 -> phase4 (here) -> phase5

Usage:
  python phase4_ppo_training_param.py                       # full run
  python phase4_ppo_training_param.py --timesteps 500000    # shorter run
  python phase4_ppo_training_param.py --smoke               # quick pipeline test
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor)
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from phase3_parametrised_env import SREPMiniGridEnv, rule_based_action

warnings.filterwarnings('ignore')

DATA_PATH = '../data/master_dataset_raw.csv'
MODEL_DIR = '../models'
OUT_DIR   = '../data'

# Fixed, representative evaluation sites — one per tier, spread across
# locations. The SAME list is used in Phase 5, so best-model selection here is
# consistent with the final KPI table.
EVAL_SITES = [
    # SREP tiers sized for ~1-day autonomy (must match phase3 TIERS).
    {'solar_kwp': 50,  'battery_kwh': 160, 'mean_load_kw': 5.6,  'location': 'Tamale'},
    {'solar_kwp': 75,  'battery_kwh': 237, 'mean_load_kw': 8.4,  'location': 'Kumasi'},
    {'solar_kwp': 120, 'battery_kwh': 378, 'mean_load_kw': 13.4, 'location': 'Axim'},
]
EVAL_SEED = 12345   # fixes the episode start for reproducible evaluation


# ============================================================
# ENV FACTORIES
# ============================================================

class _FixedSiteWrapper(gym.Wrapper):
    """Forces a specific site config on every reset (stable evaluation)."""
    def __init__(self, env, site):
        super().__init__(env)
        self._site = dict(site)

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=EVAL_SEED, options=self._site)


def make_train_env(episode_len, seed):
    """Training env: samples a RANDOM fleet site on every reset."""
    def _init():
        return SREPMiniGridEnv(data_path=DATA_PATH, model_dir=MODEL_DIR,
                               episode_len=episode_len, seed=seed)
    return _init


def make_eval_env(episode_len, site):
    """Eval env: a single fixed site, fixed start."""
    def _init():
        env = SREPMiniGridEnv(data_path=DATA_PATH, model_dir=MODEL_DIR,
                              episode_len=episode_len)
        return _FixedSiteWrapper(env, site)
    return _init


# ============================================================
# KPI LOGGING CALLBACK
# ============================================================

class KPICallback(BaseCallback):
    """
    Every eval_freq timesteps, rolls the current policy out deterministically
    on each fixed eval site and records the five project KPIs. The eval envs
    are built ONCE (not per evaluation) to avoid repeatedly re-reading the
    dataset / reloading the LSTM. Writes a CSV and keeps history for plotting.
    """
    def __init__(self, eval_episode_len, eval_freq, out_csv, verbose=1):
        super().__init__(verbose)
        self.eval_episode_len = eval_episode_len
        self.eval_freq = eval_freq
        self.out_csv = out_csv
        self.history = []
        self._last_eval = 0
        self._envs = None

    def _on_training_start(self):
        # Build one raw env per site, reused across all evaluations.
        self._envs = [SREPMiniGridEnv(data_path=DATA_PATH, model_dir=MODEL_DIR,
                                      episode_len=self.eval_episode_len)
                      for _ in EVAL_SITES]
        # Log the pristine starting point (under residual zero-init the policy IS
        # the baseline here) so the curve shows where it actually starts, not the
        # post-drift first eval at eval_freq.
        if not self.history:
            self._log_eval(0)

    def _rollout(self, env, site):
        vecnorm = self.model.get_vec_normalize_env()
        obs, _ = env.reset(seed=EVAL_SEED, options=site)
        soc_series, unmet_flags = [], []
        done = False
        while not done:
            norm_obs = vecnorm.normalize_obs(obs.reshape(1, -1))
            act, _ = self.model.predict(norm_obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act[0])
            soc_series.append(info['soc'])
            unmet_flags.append(1.0 if info['unmet_kwh'] > 1e-6 else 0.0)
            done = term or trunc
        s = env.get_episode_stats()
        soc = np.asarray(soc_series, dtype=np.float64)
        return {
            'ens_kwh':     s['unmet_kwh'],
            'lolp_pct':    100.0 * np.mean(unmet_flags) if len(unmet_flags) else 0.0,
            'efc':         s['efc'],
            'soh_pct':     100.0 * s['final_soh'],
            'soc_std_pct': 100.0 * soc.std() if len(soc) else 0.0,
            'renewable_fraction': s['renewable_fraction'],
        }

    def _log_eval(self, timesteps):
        per_site = [self._rollout(env, site)
                    for env, site in zip(self._envs, EVAL_SITES)]
        keys = ['ens_kwh', 'lolp_pct', 'efc', 'soh_pct', 'soc_std_pct',
                'renewable_fraction']
        row = {'timesteps': timesteps}
        for k in keys:
            row[k] = float(np.mean([d[k] for d in per_site]))
        self.history.append(row)

        if self.verbose:
            tag = "start" if timesteps == 0 else f"{timesteps:>8} steps"
            print(f"  [KPI @ {tag}]  "
                  f"ENS={row['ens_kwh']:.0f} kWh  "
                  f"LOLP={row['lolp_pct']:.1f}%  "
                  f"EFC={row['efc']:.1f}  "
                  f"SoH={row['soh_pct']:.2f}%  "
                  f"SoC_std={row['soc_std_pct']:.1f}%  "
                  f"RenFrac={row['renewable_fraction']:.1%}")

        pd.DataFrame(self.history).to_csv(self.out_csv, index=False)

    def _on_step(self):
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps
        self._log_eval(self.num_timesteps)
        return True


class SaveVecNormalize(BaseCallback):
    """Saves the training VecNormalize stats whenever a new best model is
    found, so Phase 5 (--best) loads normalisation matched to that policy."""
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self):
        vn = self.model.get_vec_normalize_env()
        if vn is not None:
            vn.save(self.save_path)
        return True


# ============================================================
# MAIN
# ============================================================

def collect_expert_data(n_samples, episode_len, seed):
    """Roll out the load-following baseline to gather aligned (obs, action).

    Uses a single RAW env (no vec wrappers) so rule_based_action can read the
    env's internal state directly. The obs returned by reset/step is the raw,
    UN-normalised observation the VecNormalize wrapper would later scale.
    """
    env = SREPMiniGridEnv(data_path=DATA_PATH, model_dir=MODEL_DIR,
                          episode_len=episode_len, seed=seed)
    obs, _ = env.reset(seed=seed)
    O, A = [], []
    for _ in range(n_samples):
        a = rule_based_action(env)                      # expert action at state obs
        O.append(np.asarray(obs, dtype=np.float32))
        A.append(np.asarray(a, dtype=np.float32).reshape(-1))
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            obs, _ = env.reset()
    return np.asarray(O, np.float32), np.asarray(A, np.float32)


def behavior_clone(model, venv, n_samples, epochs, episode_len, seed, batch=256):
    """Supervised warm-start: clone the baseline into the PPO policy.

    1. Collect expert (obs, action) pairs.
    2. Fit the VecNormalize obs stats IN PLACE (so the shared eval obs_rms sees
       them) and normalise the obs the same way training will.
    3. Maximise the policy's log-likelihood of the expert actions (this also
       fits the Gaussian log_std). Only policy params get gradients; the value
       head stays at init and PPO calibrates it during fine-tuning.
    """
    import torch
    print(f"\n  [warm-start] collecting {n_samples:,} expert steps...")
    O, A = collect_expert_data(n_samples, episode_len, seed)

    # Fit obs normalisation in place (preserve the obs_rms object identity so
    # eval_venv, which shares it by reference, sees the same stats).
    venv.obs_rms.mean  = O.mean(axis=0)
    venv.obs_rms.var   = O.var(axis=0) + 1e-8
    venv.obs_rms.count = float(O.shape[0])
    On = venv.normalize_obs(O).astype(np.float32)

    dev  = model.device
    On_t = torch.as_tensor(On, device=dev)
    A_t  = torch.as_tensor(A,  device=dev)
    opt  = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    n    = On_t.shape[0]

    print(f"  [warm-start] cloning policy: {epochs} epochs x {n:,} samples")
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot, nb = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            _, log_prob, _ = model.policy.evaluate_actions(On_t[idx], A_t[idx])
            loss = -log_prob.mean()                      # max-likelihood BC
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        print(f"    epoch {ep + 1:2d}/{epochs}  -logL: {tot / max(nb, 1):8.4f}")

    with torch.no_grad():
        k = min(4096, n)
        pred = model.policy._predict(On_t[:k], deterministic=True).cpu().numpy()
    mse = float(np.mean((pred.reshape(-1) - A[:k].reshape(-1)) ** 2))
    print(f"  [warm-start] done. deterministic action MSE vs baseline: {mse:.4f}")

    # Max-likelihood BC against a DETERMINISTIC expert collapses the Gaussian
    # spread (log_std -> -inf). Reset it to a usable exploration level so PPO
    # can search BEYOND the baseline instead of freezing at it. The cloned
    # action MEAN is untouched; only the exploration noise is restored.
    with torch.no_grad():
        model.policy.log_std.data.fill_(float(np.log(0.3)))
    print("  [warm-start] reset policy log_std -> exploration std ~0.3")


def _policy_lolp(model, venv, episode_len, seed, horizon=720):
    """Deterministic-policy LOLP on a fresh rollout (covariate-shift probe)."""
    env = SREPMiniGridEnv(data_path=DATA_PATH, model_dir=MODEL_DIR,
                          episode_len=episode_len, seed=seed)
    obs, _ = env.reset(seed=seed)
    for _ in range(horizon):
        nobs = venv.normalize_obs(np.asarray(obs, np.float32))
        a, _ = model.predict(nobs, deterministic=True)
        obs, _, te, tr, _ = env.step(a)
        if te or tr:
            break
    return 100.0 * env.ep_lol_hours / max(env.ep_steps, 1)


def dagger_warmstart(model, venv, n_iters, samples_per_iter, epochs_per_iter,
                     episode_len, seed, batch=256):
    """DAgger warm-start — cures the covariate shift that plain BC suffers.

    Iteration 0 is plain BC (roll out the expert, clone it). Each later
    iteration rolls out the CURRENT policy (mixing in expert actions with
    probability beta), labels every visited state with the baseline expert,
    aggregates into the dataset, and retrains. Training on the states the
    POLICY actually visits is what stops it drifting off-distribution.

    The printed per-iter 'policy-rollout LOLP' should fall across iterations:
    that is the covariate shift closing.
    """
    import torch
    dev = model.device
    raw_env = SREPMiniGridEnv(data_path=DATA_PATH, model_dir=MODEL_DIR,
                              episode_len=episode_len, seed=seed)
    allO, allA = [], []

    for it in range(n_iters):
        beta = 0.5 ** it                       # 1.0, 0.5, 0.25, ... expert mix
        obs, _ = raw_env.reset(seed=seed + 1000 + it)
        Oi, Ai = [], []
        for _ in range(samples_per_iter):
            expert_a = np.asarray(rule_based_action(raw_env),
                                  np.float32).reshape(-1)
            Oi.append(np.asarray(obs, np.float32))
            Ai.append(expert_a)
            # action used to ADVANCE the env: expert w.p. beta, else the learner
            if np.random.rand() < beta:
                step_a = expert_a
            else:
                nobs = venv.normalize_obs(np.asarray(obs, np.float32))
                step_a, _ = model.predict(nobs, deterministic=False)
            obs, _, te, tr, _ = raw_env.step(step_a)
            if te or tr:
                obs, _ = raw_env.reset()
        allO.append(np.asarray(Oi, np.float32))
        allA.append(np.asarray(Ai, np.float32))

        # refit obs normalisation on ALL aggregated data (in place, shared)
        O = np.concatenate(allO); A = np.concatenate(allA)
        venv.obs_rms.mean  = O.mean(axis=0)
        venv.obs_rms.var   = O.var(axis=0) + 1e-8
        venv.obs_rms.count = float(O.shape[0])
        On = venv.normalize_obs(O).astype(np.float32)

        # retrain the policy on the aggregated dataset
        On_t = torch.as_tensor(On, device=dev)
        A_t  = torch.as_tensor(A,  device=dev)
        opt  = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
        n = On_t.shape[0]
        for _ep in range(epochs_per_iter):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                _, logp, _ = model.policy.evaluate_actions(On_t[idx], A_t[idx])
                loss = -logp.mean()
                opt.zero_grad(); loss.backward(); opt.step()

        pl = _policy_lolp(model, venv, episode_len, seed + 7000 + it)
        print(f"  [DAgger {it + 1}/{n_iters}] beta={beta:4.2f}  "
              f"dataset={n:>7,}  policy-rollout LOLP={pl:5.1f}%")

    # restore exploration spread for the PPO handoff
    with torch.no_grad():
        model.policy.log_std.data.fill_(float(np.log(0.3)))
    print("  [DAgger] done. reset policy log_std -> exploration std ~0.3")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--timesteps', type=int, default=1_500_000)
    p.add_argument('--n-envs', type=int, default=8,
                   help='parallel training envs (each holds a copy of the '
                        'dataset+LSTM, so memory scales with this)')
    p.add_argument('--episode-len', type=int, default=8760,
                   help='training episode length in hours (1 yr = 8760)')
    p.add_argument('--eval-episode-len', type=int, default=8760,
                   help='evaluation episode length (hours) for BOTH the '
                        'monitoring curve and best-model selection. Full year '
                        '(8760) so the saved best model is the best on the '
                        'DEPLOYMENT distribution, not on cloudy 90-day windows.')
    p.add_argument('--eval-freq', type=int, default=100_000,
                   help='timesteps between evaluations')
    p.add_argument('--subproc', action='store_true',
                   help='use SubprocVecEnv (true multiprocessing)')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true',
                   help='continue training from the saved checkpoint '
                        '(../models/ppo_srep_param.zip + vecnormalize_param.pkl). '
                        'With --resume, --timesteps is the number of ADDITIONAL '
                        'steps to train (e.g. 1500000 takes 1.5M -> 3.0M).')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--warmstart', action='store_true',
                   help='behavior-clone the policy from the load-following '
                        'baseline BEFORE PPO, so training starts near the '
                        'baseline instead of cold. Ignored with --resume.')
    p.add_argument('--bc-samples', type=int, default=300_000,
                   help='expert (obs,action) pairs to collect for cloning')
    p.add_argument('--bc-epochs', type=int, default=10,
                   help='supervised passes over the BC dataset')
    p.add_argument('--dagger', action='store_true',
                   help='DAgger warm-start: BC, then iteratively retrain on the '
                        'states the policy visits (labelled by the baseline). '
                        'Cures the covariate shift plain --warmstart suffers. '
                        'Implies a warm start; ignored with --resume.')
    p.add_argument('--dagger-iters', type=int, default=6)
    p.add_argument('--dagger-samples', type=int, default=50_000,
                   help='policy-visited states labelled per DAgger iteration')
    p.add_argument('--dagger-epochs', type=int, default=5,
                   help='passes over the aggregated dataset per DAgger iteration')
    args = p.parse_args()

    if args.smoke:
        args.timesteps        = min(args.timesteps, 8_000)
        args.n_envs           = min(args.n_envs, 2)
        args.episode_len      = 24 * 14
        args.eval_episode_len = 24 * 14
        args.eval_freq        = 4_000
        args.bc_samples       = min(args.bc_samples, 5_000)
        args.bc_epochs        = min(args.bc_epochs, 3)
        args.dagger_iters     = min(args.dagger_iters, 3)
        args.dagger_samples   = min(args.dagger_samples, 2_000)
        args.dagger_epochs    = min(args.dagger_epochs, 2)

    # Residual RL is incompatible with the warm-start: BC/DAgger clone ABSOLUTE
    # baseline actions, but under residual the policy outputs a CORRECTION to the
    # baseline (zero output already == baseline). Disable warm-start so the two
    # don't fight; residual is meant to run as a cold start.
    if SREPMiniGridEnv.RESIDUAL and (args.warmstart or args.dagger):
        print("  NOTE: residual env active -> ignoring --warmstart/--dagger "
              "(residual already starts the policy at the baseline).")
        args.warmstart = False
        args.dagger = False

    print("=" * 60)
    print("  PHASE 4 — PPO TRAINING (FLEET POLICY)")
    print("=" * 60)
    print(f"  timesteps         : {args.timesteps:,}")
    print(f"  parallel envs     : {args.n_envs}")
    print(f"  train episode len : {args.episode_len} steps")
    print(f"  eval episode len  : {args.eval_episode_len} steps (curve + best-model selection)")
    print(f"  eval frequency    : every {args.eval_freq:,} steps")
    print(f"  smoke test        : {args.smoke}")
    print(f"  residual RL       : {SREPMiniGridEnv.RESIDUAL}"
          + (f" (scale {SREPMiniGridEnv.RESIDUAL_SCALE})"
             if SREPMiniGridEnv.RESIDUAL else ""))
    if args.dagger and not args.resume:
        print(f"  warm-start (DAgger): {args.dagger_iters} iters x "
              f"{args.dagger_samples:,} samples from baseline expert")
    elif args.warmstart and not args.resume:
        print(f"  warm-start (BC)   : {args.bc_samples:,} samples x "
              f"{args.bc_epochs} epochs from load-following baseline")

    os.makedirs(f'{MODEL_DIR}/best', exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- vectorised training env (VecMonitor records RAW reward) ----
    VecCls = SubprocVecEnv if args.subproc else DummyVecEnv
    venv = VecCls([make_train_env(args.episode_len, args.seed + i)
                   for i in range(args.n_envs)])
    venv = VecMonitor(venv)

    final_model_path = f'{MODEL_DIR}/ppo_srep_param'          # (.zip added by SB3)
    final_vn_path    = f'{MODEL_DIR}/vecnormalize_param.pkl'

    if args.resume:
        if not (os.path.exists(final_model_path + '.zip')
                and os.path.exists(final_vn_path)):
            raise FileNotFoundError(
                f"--resume needs a prior checkpoint: {final_model_path}.zip and "
                f"{final_vn_path}. Run a normal (non-resume) training first.")
        # Restore obs/reward normalisation so the policy keeps seeing
        # consistently scaled observations (a fresh VecNormalize would reset
        # the running stats and wreck the loaded policy).
        venv = VecNormalize.load(final_vn_path, venv)
        venv.training = True
        venv.norm_reward = True
        model = PPO.load(final_model_path, env=venv, device='auto')
        print(f"\n  RESUMING from {final_model_path}.zip "
              f"@ {model.num_timesteps:,} steps")
        print(f"  Training {args.timesteps:,} MORE steps "
              f"-> {model.num_timesteps + args.timesteps:,} total")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, gamma=0.999)
        # ---- PPO (tuned for long-horizon continuous control) ----
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        # Exploration for the warm start comes from resetting log_std after BC
        # (see behavior_clone), NOT from an entropy bonus. A nonzero ent_coef was
        # observed to INFLATE the action std (~0.9), turning rollouts near-random
        # and washing the clone off the baseline. With ent_coef=0, std anneals
        # downward as PPO exploits, preserving and refining the cloned policy.
        model = PPO(
            'MlpPolicy', venv,
            learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
            gamma=0.999, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=policy_kwargs, seed=args.seed, verbose=1,
        )
        if SREPMiniGridEnv.RESIDUAL:
            # Zero the action head so the initial correction is EXACTLY zero ->
            # the policy starts at the baseline itself, then PPO learns small
            # corrections away from it. Also soften exploration: default std=1
            # with RESIDUAL_SCALE corrections is large enough to knock the
            # policy off the baseline before it learns, so start gentler.
            import torch
            with torch.no_grad():
                model.policy.action_net.weight.zero_()
                model.policy.action_net.bias.zero_()
                model.policy.log_std.data.fill_(float(np.log(0.2)))
            print("  residual: zero-init action head + log_std=0.2 "
                  "-> starts at baseline, explores gently")

    # ---- best-model selection across ALL fleet eval sites ----
    eval_venv = DummyVecEnv([make_eval_env(args.eval_episode_len, s)
                             for s in EVAL_SITES])
    eval_venv = VecMonitor(eval_venv)
    eval_venv = VecNormalize(eval_venv, norm_obs=True, norm_reward=False,
                             clip_obs=10.0, training=False, gamma=0.999)
    eval_venv.obs_rms = venv.obs_rms   # share running obs stats by reference

    # ---- warm-start (fresh runs only) ----
    # Done here (after eval_venv exists) so we can report the cloned policy's
    # reliability BEFORE PPO touches it. Both paths mutate venv.obs_rms in
    # place, which eval_venv shares by reference.
    if (args.dagger or args.warmstart) and not args.resume:
        if args.dagger:
            dagger_warmstart(model, venv, args.dagger_iters, args.dagger_samples,
                             args.dagger_epochs, args.episode_len, args.seed)
        else:
            behavior_clone(model, venv, args.bc_samples, args.bc_epochs,
                           args.episode_len, args.seed)
        from stable_baselines3.common.evaluation import evaluate_policy
        mr, sr = evaluate_policy(model, eval_venv,
                                 n_eval_episodes=len(EVAL_SITES),
                                 deterministic=True)
        tag = 'DAgger' if args.dagger else 'BC'
        print(f"  [{tag}] post-warmstart eval reward (pre-PPO, per "
              f"{args.eval_episode_len}-step episode): {mr:.1f} +/- {sr:.1f}")
        print(f"  [{tag}] (strongly positive => clone generalises to eval sites)")

    save_best_vn = SaveVecNormalize(f'{MODEL_DIR}/best/vecnormalize_param.pkl')
    eval_cb = EvalCallback(
        eval_venv,
        best_model_save_path=f'{MODEL_DIR}/best',
        log_path=OUT_DIR,
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=len(EVAL_SITES),
        deterministic=True, render=False, verbose=1,
        callback_on_new_best=save_best_vn,
    )

    # ---- baseline FLOOR for best-model selection (residual fresh runs) ----
    # The zero-init policy IS the load-following baseline. Evaluate it now, save it
    # as the initial best, and seed EvalCallback's best_mean_reward with its score
    # so a later DRIFTED policy can only overwrite "best" if it genuinely beats the
    # baseline. Without this, the first eval at eval_freq saves whatever the policy
    # has drifted to (worse than baseline), which is exactly what happened before.
    if SREPMiniGridEnv.RESIDUAL and not args.resume:
        from stable_baselines3.common.evaluation import evaluate_policy
        base_r, _ = evaluate_policy(model, eval_venv,
                                    n_eval_episodes=len(EVAL_SITES),
                                    deterministic=True)
        os.makedirs(f'{MODEL_DIR}/best', exist_ok=True)
        model.save(f'{MODEL_DIR}/best/best_model')
        vn0 = model.get_vec_normalize_env()
        if vn0 is not None:
            vn0.save(f'{MODEL_DIR}/best/vecnormalize_param.pkl')
        eval_cb.best_mean_reward = float(base_r)
        print(f"  residual: saved step-0 baseline as best-model floor "
              f"(eval reward {base_r:.1f}); best replaced only if beaten")

    kpi_cb = KPICallback(args.eval_episode_len, args.eval_freq,
                         f'{OUT_DIR}/phase4_ppo_progress.csv')
    if args.resume and os.path.exists(f'{OUT_DIR}/phase4_ppo_progress.csv'):
        prev = pd.read_csv(f'{OUT_DIR}/phase4_ppo_progress.csv')
        kpi_cb.history = prev.to_dict('records')
        if len(prev):
            kpi_cb._last_eval = int(prev['timesteps'].max())
        print(f"  Loaded {len(prev)} prior KPI rows; training curve will extend.")

    # ---- train ----
    print("\nTraining...\n")
    model.learn(total_timesteps=args.timesteps,
                callback=[eval_cb, kpi_cb], progress_bar=False,
                reset_num_timesteps=not args.resume)

    # ---- save final artifacts ----
    model.save(f'{MODEL_DIR}/ppo_srep_param')
    venv.save(f'{MODEL_DIR}/vecnormalize_param.pkl')
    print(f"\nSaved final policy   : {MODEL_DIR}/ppo_srep_param.zip")
    print(f"Saved final VecNorm  : {MODEL_DIR}/vecnormalize_param.pkl")
    print(f"Saved best policy    : {MODEL_DIR}/best/best_model.zip")
    print(f"Saved best VecNorm   : {MODEL_DIR}/best/vecnormalize_param.pkl")

    # ---- training-curve plot ----
    if kpi_cb.history:
        hist = pd.DataFrame(kpi_cb.history)
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        panels = [
            ('ens_kwh',     'Energy Not Served (kWh, 90d)',  '#ef4444'),
            ('lolp_pct',    'Loss of Load Probability (%)',  '#f97316'),
            ('efc',         'Equivalent Full Cycles (90d)',  '#8b5cf6'),
            ('soh_pct',     'Final State of Health (%)',     '#22c55e'),
            ('soc_std_pct', 'SoC Std Dev (%)',               '#3b82f6'),
            ('renewable_fraction', 'Renewable Fraction',     '#0ea5e9'),
        ]
        for ax, (col, title, color) in zip(axes.flat, panels):
            ax.plot(hist['timesteps'], hist[col], color=color,
                    linewidth=1.8, marker='o', markersize=3)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Timesteps')
            ax.grid(True, alpha=0.3)
        fig.suptitle('Phase 4 — PPO Training Progress (mean over eval sites)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/phase4_ppo_training_curve.png', dpi=150)
        print(f"Saved curve          : {OUT_DIR}/phase4_ppo_training_curve.png")

    print("\n" + "=" * 60)
    print("  PHASE 4 COMPLETE")
    print("=" * 60)
    print("Next: run phase5_evaluation_param.py --best")


if __name__ == '__main__':
    main()
