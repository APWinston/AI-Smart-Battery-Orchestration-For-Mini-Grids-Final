#!/usr/bin/env python
# coding: utf-8
"""
export_srep_to_onnx.py  (v2 - distribution-free PPO export + legacy opset 13)
=============================================================================
Run ON YOUR PC in your training env (torch, stable-baselines3, scikit-learn,
scipy, numpy; onnx + onnxscript already installed).

Place next to:
    best_lstm_param.pth  ppo_srep_param.zip  vecnormalize_param.pkl
    scaler_X_param.pkl   scaler_y_param.pkl
(or edit MODELS below). Writes ./matlab_export/ with the two .onnx files
and srep_inference_params.mat.
"""
import os, pickle, numpy as np, torch, torch.nn as nn
from scipy.io import savemat

MODELS = "."
OUT    = "matlab_export"
os.makedirs(OUT, exist_ok=True)
OBS_DIM = 60

def export_onnx(model, dummy, path, in_names, out_names, dyn):
    """Prefer the legacy exporter (opset 13, MATLAB-friendly); fall back to dynamo."""
    try:
        torch.onnx.export(model, dummy, path, input_names=in_names,
                          output_names=out_names, dynamic_axes=dyn,
                          opset_version=13, dynamo=False)
        print(f"[ok] {os.path.basename(path)}  (legacy exporter, opset 13)")
    except TypeError:                      # very old torch without the dynamo kwarg
        torch.onnx.export(model, dummy, path, input_names=in_names,
                          output_names=out_names, dynamic_axes=dyn, opset_version=13)
        print(f"[ok] {os.path.basename(path)}  (opset 13)")
    except Exception as e:                 # legacy path removed -> dynamo, static batch
        print(f"[warn] legacy export failed ({type(e).__name__}: {e}); "
              f"retrying with dynamo=True, opset 17")
        torch.onnx.export(model, dummy, path, input_names=in_names,
                          output_names=out_names, opset_version=17, dynamo=True)
        print(f"[ok] {os.path.basename(path)}  (dynamo exporter, opset 17)")

# ----------------------------------------------------------------------
# 1) LSTM  ->  ONNX
# ----------------------------------------------------------------------
class MiniGridLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, forecast, output_size):
        super().__init__()
        self.forecast = forecast; self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, forecast * output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).view(-1, self.forecast, self.output_size)

lstm = MiniGridLSTM(7, 128, 2, 48, 2)
lstm.load_state_dict(torch.load(f"{MODELS}/best_lstm_param.pth",
                                map_location="cpu", weights_only=True))
lstm.eval()
export_onnx(lstm, torch.zeros(1, 24, 7), f"{OUT}/lstm_forecaster.onnx",
            ["window"], ["forecast"],
            {"window": {0: "batch"}, "forecast": {0: "batch"}})

# ----------------------------------------------------------------------
# 2) PPO policy  ->  ONNX   (deterministic action = mean; NO distribution)
# ----------------------------------------------------------------------
from stable_baselines3 import PPO
ppo = PPO.load(f"{MODELS}/ppo_srep_param.zip", device="cpu")
ppo.policy.set_training_mode(False); ppo.policy.eval()

class OnnxPPO(nn.Module):
    """obs -> mean action. Skips the Gaussian so the exporter never builds a
    data-dependent distribution. For PPO the deterministic action IS the mean,
    so this is numerically identical to predict(obs, deterministic=True)."""
    def __init__(self, policy):
        super().__init__(); self.policy = policy
    def forward(self, obs):
        latent_pi = self.policy.mlp_extractor.forward_actor(obs)  # FlattenExtractor is identity for flat Box obs
        return self.policy.action_net(latent_pi)

export_onnx(OnnxPPO(ppo.policy), torch.zeros(1, OBS_DIM), f"{OUT}/ppo_policy.onnx",
            ["obs"], ["action"], {"obs": {0: "batch"}, "action": {0: "batch"}})

# ----------------------------------------------------------------------
# 3) scalers + VecNormalize stats + constants  ->  .mat
# ----------------------------------------------------------------------
sx = pickle.load(open(f"{MODELS}/scaler_X_param.pkl", "rb"))
sy = pickle.load(open(f"{MODELS}/scaler_y_param.pkl", "rb"))
with open(f"{MODELS}/vecnormalize_param.pkl", "rb") as f:
    vn = pickle.load(f)

mean = np.asarray(vn.obs_rms.mean, dtype=np.float64).reshape(1, -1)
var  = np.asarray(vn.obs_rms.var,  dtype=np.float64).reshape(1, -1)
assert mean.shape[1] == OBS_DIM, f"VecNormalize obs dim {mean.shape[1]} != {OBS_DIM}"

params = dict(
    sx_scale=sx.scale_.astype(np.float64).reshape(1, -1),
    sx_min  =sx.min_.astype(np.float64).reshape(1, -1),
    sy_scale=sy.scale_.astype(np.float64).reshape(1, -1),
    sy_min  =sy.min_.astype(np.float64).reshape(1, -1),
    vn_mean=mean, vn_var=var,
    vn_clip=float(vn.clip_obs), vn_eps=float(vn.epsilon),
    lookback=24, forecast=48, obs_hourly=24, obs_dim=OBS_DIM,
    residual_scale=0.25, pv_derating=0.75,
    max_solar_kwp=120.0, max_bat_kwh=400.0, max_load_kw=20.0,
    orig_mean_load=192.9, soc_min=0.10, soc_max=0.95, soc_init=0.50,
    eta_charge=0.95, eta_discharge=0.95,
    tiers=np.array([[50, 160, 5.6], [75, 237, 8.4], [120, 378, 13.4]], dtype=np.float64),
    lstm_features=np.array(['ssrd_wm2', 'tp', 'temp_c', 'load_kw',
                            'hour', 'month', 'dayofweek'], dtype=object),
)
savemat(f"{OUT}/srep_inference_params.mat", params)
print("[ok] srep_inference_params.mat")
print(f"\nDone. obs_dim={OBS_DIM}, VecNorm length={mean.shape[1]}.")

# ----------------------------------------------------------------------
# 4) optional numeric self-check (needs onnxruntime)
# ----------------------------------------------------------------------
try:
    import onnxruntime as ort
    fc = ort.InferenceSession(f"{OUT}/lstm_forecaster.onnx").run(
        None, {"window": np.random.rand(1, 24, 7).astype(np.float32)})[0]
    ac = ort.InferenceSession(f"{OUT}/ppo_policy.onnx").run(
        None, {"obs": np.random.rand(1, 60).astype(np.float32)})[0]
    print(f"self-check: LSTM out {fc.shape} (expect (1,48,2)); "
          f"PPO action {ac.shape}={ac.ravel()} (expect 1 value)")
except Exception as e:
    print(f"(skipped self-check: {e})")
