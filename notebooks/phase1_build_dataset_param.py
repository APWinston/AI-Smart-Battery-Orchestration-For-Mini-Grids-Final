#!/usr/bin/env python
# coding: utf-8
"""
Phase 1 — Build Raw Dataset for Parameterised Ghana SREP Mini-Grid
===================================================================
Prepares master_dataset.csv for the parameterised training pipeline.

Unlike the original Phase 1, this version does NOT scale the load to
a single fixed system. The raw Nigeria load profile is preserved so
that Phase 2 can apply multiple load scales covering the full Ghana
SREP fleet distribution (5 to 40 kW mean load).

The Ghana SREP programme deploys 35 mini-grids across three site tiers:
  11 sites x  50 kWp
  11 sites x  75 kWp
  13 sites x 120 kWp
  Total: 4.525 MWp across island and lakeside communities, Volta Lake

This script validates the raw dataset, adds time features, checks data
quality, and saves master_dataset_raw.csv ready for Phase 2.

Input:  ../data/master_dataset.csv
Output: ../data/master_dataset_raw.csv

Run order:
  python phase1_build_dataset_param.py
  python phase2_train_lstm_param.py
  python phase3_parametrised_env.py
  python phase4_ppo_training_param.py
  python phase5_evaluation_param.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  PHASE 1 — BUILD RAW DATASET (PARAMETERISED PIPELINE)")
print("=" * 60)

# ============================================================
# GHANA SREP FLEET — reference only, not used for scaling
# ============================================================
SREP_SITES      = 35
SREP_TOTAL_MWP  = 4.525
SREP_TIERS      = {
    '50 kWp'  : 11,
    '75 kWp'  : 11,
    '120 kWp' : 13,
}
SREP_AVG_KWP    = (SREP_TOTAL_MWP * 1000) / SREP_SITES   # 129.3 kWp
ORIG_MEAN_LOAD  = 192.9    # kW — original Nigeria national grid mean

print(f"\nGhana SREP fleet reference:")
print(f"  Total sites    : {SREP_SITES}")
print(f"  Total capacity : {SREP_TOTAL_MWP} MWp")
for tier, count in SREP_TIERS.items():
    print(f"  {count} sites x {tier}")
print(f"  Fleet average  : {SREP_AVG_KWP:.1f} kWp per site")
print(f"\nLoad scaling will be applied in Phase 2 across:")
print(f"  5 kW to 40 kW mean load (full SREP community size range)")

# ============================================================
# LOAD RAW DATASET
# ============================================================
print("\nLoading master_dataset.csv...")
df = pd.read_csv('../data/master_dataset.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
print(f"Loaded: {df.shape}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Locations: {sorted(df['location'].unique().tolist())}")

# ============================================================
# DATA QUALITY CHECKS
# ============================================================
print("\nData quality checks:")

# Missing values
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  Missing values: NONE")
else:
    print(f"  WARNING — missing values found:")
    print(missing[missing > 0])
    df = df.ffill()
    print("  Forward-filled missing values")

# Negative irradiance
neg_ssrd = (df['ssrd_wm2'] < 0).sum()
if neg_ssrd > 0:
    print(f"  WARNING — {neg_ssrd} negative ssrd values. Clipping to 0.")
    df['ssrd_wm2'] = df['ssrd_wm2'].clip(lower=0)
else:
    print("  Negative irradiance: NONE")

# Negative load
neg_load = (df['load_kw'] < 0).sum()
if neg_load > 0:
    print(f"  WARNING — {neg_load} negative load values. Clipping to 0.")
    df['load_kw'] = df['load_kw'].clip(lower=0)
else:
    print("  Negative load: NONE")

# Temperature range sanity check
t_min = df['temp_c'].min()
t_max = df['temp_c'].max()
print(f"  Temperature range: {t_min:.1f} to {t_max:.1f} degC")
if t_min < -10 or t_max > 60:
    print("  WARNING — temperature values outside expected range for Ghana")
else:
    print("  Temperature range: OK")

# ============================================================
# ADD TIME FEATURES
# ============================================================
print("\nAdding time features...")
# Train on 3 weather-diverse sites; hold out 3 for evaluation.
TRAIN_LOCS = ['Tamale', 'Kumasi', 'Axim']
EVAL_LOCS  = ['Accra', 'Bolgatanga', 'Akosombo']
ALL_LOCS   = sorted(df['location'].unique().tolist())
# location_code kept for reference only — Option 1 drops it as an LSTM feature
# so the forecaster generalises to unseen locations from weather alone.
df['location_code'] = df['location'].astype('category').cat.codes
df['hour']          = df['datetime'].dt.hour
df['month']         = df['datetime'].dt.month
df['dayofweek']     = df['datetime'].dt.dayofweek

print(f"  locations    : {ALL_LOCS}")
print(f"  training     : {TRAIN_LOCS}")
print(f"  held-out eval: {EVAL_LOCS}")
print(f"  hour range   : {df['hour'].min()} to {df['hour'].max()}")
print(f"  month range  : {df['month'].min()} to {df['month'].max()}")

# ============================================================
# PER LOCATION STATS
# ============================================================
print("\nPer-location raw statistics:")
print(f"  {'Location':<12} {'Rows':>8} {'Mean ssrd':>12} {'Mean load':>12} {'Mean temp':>12}")
print(f"  {'-'*58}")
for loc in ALL_LOCS:
    loc_df = df[df['location'] == loc]
    tag = 'train' if loc in TRAIN_LOCS else 'eval'
    print(f"  {loc:<12} {len(loc_df):>8,} "
          f"{loc_df['ssrd_wm2'].mean():>11.2f}W "
          f"{loc_df['load_kw'].mean():>10.2f}kW "
          f"{loc_df['temp_c'].mean():>10.1f}C  [{tag}]")

print(f"\n  Original Nigeria load mean: {df['load_kw'].mean():.2f} kW")
print(f"  This will be scaled in Phase 2 to cover 5-40 kW range")

# ============================================================
# LOAD SCALE PREVIEW
# ============================================================
print("\nLoad scale factors that Phase 2 will apply:")
target_means  = [5.0, 8.0, 12.0, 18.958, 28.0, 40.0]
print(f"  {'Target mean':>12}  {'Scale factor':>14}  {'Community size':>18}")
print(f"  {'-'*48}")
for tm in target_means:
    sf = tm / ORIG_MEAN_LOAD
    pop = tm * 69.5
    print(f"  {tm:>10.1f} kW  {sf:>14.5f}  ~{pop:>6.0f} people")

# ============================================================
# SAVE
# ============================================================
output_cols = ['datetime', 'location', 'location_code',
               'ssrd_wm2', 'tp', 'temp_c', 'load_kw',
               'hour', 'month', 'dayofweek']

df_out = df[output_cols].reset_index(drop=True)
output_path = '../data/master_dataset_raw.csv'
df_out.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")
print(f"Shape: {df_out.shape}")
print(f"Columns: {df_out.columns.tolist()}")

# ============================================================
# PLOT — raw solar and load for all locations
# ============================================================
nloc = len(ALL_LOCS)
fig, axes = plt.subplots(nloc, 2, figsize=(16, 3.2 * nloc))
show = 24 * 7   # first 7 days

for row_idx, loc in enumerate(ALL_LOCS):
    loc_df  = df_out[df_out['location'] == loc].reset_index(drop=True)
    sol_kw  = (loc_df['ssrd_wm2'] / 1000.0) * 100.0 * 0.75   # kW at 100m2 reference area
    tag = 'train' if loc in TRAIN_LOCS else 'EVAL'

    axes[row_idx, 0].plot(sol_kw.values[:show],
                          color='#f59e0b', linewidth=1.0, alpha=0.85)
    axes[row_idx, 0].set_title(f'{loc} [{tag}] — Solar PV at 100m2 (First 7 Days)', fontsize=10)
    axes[row_idx, 0].set_ylabel('kW')
    axes[row_idx, 0].grid(True, alpha=0.3)

    axes[row_idx, 1].plot(loc_df['load_kw'].values[:show],
                          color='#ef4444', linewidth=1.0, alpha=0.85)
    axes[row_idx, 1].set_title(f'{loc} [{tag}] — Load (Raw Nigeria Profile)', fontsize=10)
    axes[row_idx, 1].set_ylabel('kW (unscaled)')
    axes[row_idx, 1].grid(True, alpha=0.3)

plt.suptitle('Phase 1 — Raw Dataset: Solar Irradiance and Load by Location',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('../data/phase1_raw_dataset_overview.png', dpi=150)
plt.show()
print("Plot saved: ../data/phase1_raw_dataset_overview.png")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 1 COMPLETE")
print("=" * 60)
print(f"  Output  : ../data/master_dataset_raw.csv")
print(f"  Rows    : {len(df_out):,}")
print(f"  Locations: {', '.join(ALL_LOCS)}")
print(f"    training: {', '.join(TRAIN_LOCS)} | held-out eval: {', '.join(EVAL_LOCS)}")
print(f"  Period  : {df_out['datetime'].min()} to {df_out['datetime'].max()}")
print(f"  Features: {df_out.shape[1]} columns")
print(f"\n  NOTE: Load is NOT scaled in this file.")
print(f"  Phase 2 applies 6 scale factors to cover 5-40 kW mean load range.")
print(f"\nNext: run phase2_train_lstm_param.py")
