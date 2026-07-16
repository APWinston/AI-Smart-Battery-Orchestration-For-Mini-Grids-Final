#!/usr/bin/env python
# coding: utf-8
"""
pack_to_json.py
===============
Converts matlab_export/srep_matlab_pack.mat  ->  srep_weights.json
so the browser dashboard can run your real LSTM + PPO in pure JavaScript.

Run on your PC (needs scipy + numpy, which you already have):
    python pack_to_json.py
Then load the resulting srep_weights.json in the dashboard's "Load model" button.
"""
import os, json, numpy as np
from scipy.io import loadmat

SRC = "matlab_export/srep_matlab_pack.mat"
if not os.path.isfile(SRC):
    SRC = "srep_matlab_pack.mat"
m = loadmat(SRC)

out = {}
for k, v in m.items():
    if k.startswith("__"):
        continue
    a = np.asarray(v, dtype=np.float64)
    if a.size == 1:                                   # scalar
        out[k] = float(a.ravel()[0])
    elif a.ndim == 2 and (a.shape[0] == 1 or a.shape[1] == 1):
        out[k] = a.ravel().tolist()                   # vector -> flat list
    else:
        out[k] = a.tolist()                           # matrix -> list of lists

with open("srep_weights.json", "w") as f:
    json.dump(out, f)

mb = os.path.getsize("srep_weights.json") / 1e6
print(f"[ok] srep_weights.json  ({mb:.2f} MB, {len(out)} fields)")
print("Open the dashboard, click 'Load model', and pick this file.")
