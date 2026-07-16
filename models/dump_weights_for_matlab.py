#!/usr/bin/env python
# coding: utf-8
"""
dump_weights_for_matlab.py
==========================
Option 2: export raw network weights so MATLAB can run the LSTM + PPO with
plain matrix math (no ONNX, no Deep Learning Toolbox, no converter).

Run ON YOUR PC in your training env (needs: torch, stable-baselines3,
scikit-learn, scipy, numpy). Put next to:
    best_lstm_param.pth  ppo_srep_param.zip  vecnormalize_param.pkl
    scaler_X_param.pkl   scaler_y_param.pkl

Writes ./matlab_export/srep_matlab_pack.mat  (one file the MATLAB app loads).
"""
import os, pickle, numpy as np, torch, torch.nn as nn
from scipy.io import savemat
from stable_baselines3 import PPO

MODELS = "."
OUT    = "matlab_export"
os.makedirs(OUT, exist_ok=True)

# ---- LSTM weights (architecture must match phase2) ----
class MiniGridLSTM(nn.Module):
    def __init__(self, i, h, n, f, o):
        super().__init__()
        self.lstm = nn.LSTM(i, h, n, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(h, f * o)
    def forward(self, x):                       # unused; only need state_dict
        out, _ = self.lstm(x); return self.fc(out[:, -1, :])

m = MiniGridLSTM(7, 128, 2, 48, 2)
m.load_state_dict(torch.load(f"{MODELS}/best_lstm_param.pth",
                             map_location="cpu", weights_only=True))
sd = m.state_dict()
G  = lambda k: sd[k].cpu().numpy().astype(np.float64)

pack = dict(
    # PyTorch LSTM gate order in each weight block: [input, forget, cell(g), output]
    lstm_Wih0=G('lstm.weight_ih_l0'), lstm_Whh0=G('lstm.weight_hh_l0'),
    lstm_bih0=G('lstm.bias_ih_l0').reshape(1, -1), lstm_bhh0=G('lstm.bias_hh_l0').reshape(1, -1),
    lstm_Wih1=G('lstm.weight_ih_l1'), lstm_Whh1=G('lstm.weight_hh_l1'),
    lstm_bih1=G('lstm.bias_ih_l1').reshape(1, -1), lstm_bhh1=G('lstm.bias_hh_l1').reshape(1, -1),
    fc_W=G('fc.weight'), fc_b=G('fc.bias').reshape(1, -1),
    hidden=128, num_layers=2,
)

# ---- PPO policy weights (pi MLP [256,256] tanh + action head) ----
ppo = PPO.load(f"{MODELS}/ppo_srep_param.zip", device="cpu")
ps  = ppo.policy.state_dict()
Pp  = lambda k: ps[k].cpu().numpy().astype(np.float64)
pack.update(
    pi_W0=Pp('mlp_extractor.policy_net.0.weight'), pi_b0=Pp('mlp_extractor.policy_net.0.bias').reshape(1, -1),
    pi_W1=Pp('mlp_extractor.policy_net.2.weight'), pi_b1=Pp('mlp_extractor.policy_net.2.bias').reshape(1, -1),
    act_W=Pp('action_net.weight'), act_b=Pp('action_net.bias').reshape(1, -1),
)

# ---- scalers + VecNormalize + env constants ----
sx = pickle.load(open(f"{MODELS}/scaler_X_param.pkl", "rb"))
sy = pickle.load(open(f"{MODELS}/scaler_y_param.pkl", "rb"))
vn = pickle.load(open(f"{MODELS}/vecnormalize_param.pkl", "rb"))
mean = np.asarray(vn.obs_rms.mean, np.float64).reshape(1, -1)
var  = np.asarray(vn.obs_rms.var,  np.float64).reshape(1, -1)
assert mean.shape[1] == 60, f"obs dim {mean.shape[1]} != 60"

pack.update(
    sx_scale=sx.scale_.reshape(1, -1).astype(np.float64), sx_min=sx.min_.reshape(1, -1).astype(np.float64),
    sy_scale=sy.scale_.reshape(1, -1).astype(np.float64), sy_min=sy.min_.reshape(1, -1).astype(np.float64),
    vn_mean=mean, vn_var=var, vn_clip=float(vn.clip_obs), vn_eps=float(vn.epsilon),
    lookback=24, forecast=48, obs_hourly=24, obs_dim=60,
    residual_scale=0.25, pv_derating=0.75,
    max_solar_kwp=120.0, max_bat_kwh=400.0, max_load_kw=20.0, orig_mean_load=192.9,
    soc_min=0.10, soc_max=0.95, soc_init=0.50, eta=0.95,
    tiers=np.array([[50, 160, 5.6], [75, 237, 8.4], [120, 378, 13.4]], np.float64),
)
savemat(f"{OUT}/srep_matlab_pack.mat", pack)
print(f"[ok] {OUT}/srep_matlab_pack.mat   ({len(pack)} fields)")
for k in ['lstm_Wih0', 'lstm_Whh0', 'fc_W', 'pi_W0', 'pi_W1', 'act_W', 'vn_mean']:
    print(f"     {k}: {np.array(pack[k]).shape}")
print("Copy matlab_export/ next to the MATLAB app and run srep_minigrid_3d_twin.")
