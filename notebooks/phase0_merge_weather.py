#!/usr/bin/env python
# coding: utf-8
"""
Phase 0 — Merge raw weather + load into master_dataset.csv (ALL locations)
==========================================================================
Combines the per-location ERA5 weather exports with the shared Nigeria load
profile into the merged intermediate that Phase 1 consumes.

Transforms (matching the original 3-location master_dataset.csv exactly):
  ssrd_wm2 = ssrd / 3600        # ERA5 accumulated J/m^2 over the hour -> W/m^2
  temp_c   = t2m  - 273.15      # Kelvin -> Celsius
  tp       = tp                 # passed through
  load_kw  = shared Nigeria national profile (identical across all sites)

Training locations : Tamale, Kumasi, Axim
Evaluation (held-out) locations : Accra, Bolgatanga, Akosombo

Input : ../data/{Solar Irradiance,2m temperature,Precipitation} <Location>.csv
        ../data/master_dataset.csv   (only to lift the shared load series)
Output: ../data/master_dataset.csv   (now containing all 6 locations)
"""

import pandas as pd
import numpy as np

DATA = '../data'
TRAIN_LOCS = ['Tamale', 'Kumasi', 'Axim']
EVAL_LOCS  = ['Accra', 'Bolgatanga', 'Akosombo']
ALL_LOCS   = TRAIN_LOCS + EVAL_LOCS

print("=" * 60)
print("  PHASE 0 — MERGE WEATHER + LOAD (ALL 6 LOCATIONS)")
print("=" * 60)

# --- shared Nigeria load profile (identical across locations) ---
old = pd.read_csv(f'{DATA}/master_dataset.csv', parse_dates=['datetime'])
load_series = (old[old['location'] == 'Tamale'][['datetime', 'load_kw']]
               .drop_duplicates('datetime')
               .reset_index(drop=True))
print(f"Load profile: {len(load_series)} hours, mean {load_series['load_kw'].mean():.2f} kW")


def build_location(loc):
    sol = pd.read_csv(f'{DATA}/Solar Irradiance {loc}.csv', parse_dates=['valid_time'])
    tmp = pd.read_csv(f'{DATA}/2m temperature {loc}.csv', parse_dates=['valid_time'])
    prc = pd.read_csv(f'{DATA}/Precipitation {loc}.csv',   parse_dates=['valid_time'])

    df = sol[['valid_time', 'ssrd']].rename(columns={'valid_time': 'datetime'})
    df['ssrd_wm2'] = df['ssrd'] / 3600.0
    df = df.drop(columns='ssrd')
    df = df.merge(tmp[['valid_time', 't2m']].rename(columns={'valid_time': 'datetime'}),
                  on='datetime', how='inner')
    df['temp_c'] = df['t2m'] - 273.15
    df = df.drop(columns='t2m')
    df = df.merge(prc[['valid_time', 'tp']].rename(columns={'valid_time': 'datetime'}),
                  on='datetime', how='inner')
    df = df.merge(load_series, on='datetime', how='inner')
    df['location'] = loc
    return df[['datetime', 'location', 'ssrd_wm2', 'tp', 'temp_c', 'load_kw']]


frames = []
for loc in ALL_LOCS:
    d = build_location(loc)
    tag = 'train' if loc in TRAIN_LOCS else 'EVAL '
    print(f"  [{tag}] {loc:<11} {len(d):>7} rows  "
          f"ssrd {d['ssrd_wm2'].mean():6.1f}  temp {d['temp_c'].mean():5.1f}C")
    frames.append(d)

merged = pd.concat(frames, ignore_index=True)

# --- verify we reproduce the original 3-location data exactly ---
chk = merged[merged['location'] == 'Tamale'].reset_index(drop=True)
ref = old[old['location'] == 'Tamale'].reset_index(drop=True)
n = min(len(chk), len(ref))
ok = np.allclose(chk['ssrd_wm2'][:n], ref['ssrd_wm2'][:n], atol=1e-3) and \
     np.allclose(chk['temp_c'][:n],   ref['temp_c'][:n],   atol=1e-3)
print(f"\nReproduces original Tamale merge exactly: {ok}")

out = f'{DATA}/master_dataset.csv'
merged.to_csv(out, index=False)
print(f"Saved: {out}  shape {merged.shape}")
print(f"Locations: {sorted(merged['location'].unique())}")
print("\nPHASE 0 COMPLETE — next: phase1_build_dataset_param.py")
