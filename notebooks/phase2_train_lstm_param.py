#!/usr/bin/env python
# coding: utf-8
"""
Phase 2 — Train LSTM Forecaster (Parameterised — Multi-Scale Load)
====================================================================
Retrains the LSTM on load profiles augmented across the full Ghana SREP
fleet distribution (5 to 40 kW mean load). The Nigeria load shape is
preserved but scaled to six different community sizes, giving the LSTM
training examples at every scale it will encounter during PPO training.

Load scale factors applied to the base Nigeria profile:
  0.26x  ->  ~5  kW mean   (small island, ~100 people)
  0.40x  ->  ~8  kW mean   (small community, ~150 people)
  0.60x  ->  ~12 kW mean   (medium community, ~250 people)
  1.00x  ->  ~19 kW mean   (base SREP average, ~1,318 people)  <- old system
  1.50x  ->  ~28 kW mean   (larger site, ~550 people)
  2.10x  ->  ~40 kW mean   (large lakeside community, ~780 people)

Architecture: unchanged from old system
  8 inputs | 128 hidden | 2 layers | 24h lookback | 24h forecast
  Outputs: [solar W/m2, load kW]

Output: ../models/best_lstm_param.pth

Input:  ../data/master_dataset_raw.csv   (from phase1_build_dataset_param.py)
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

print(f"Libraries imported! PyTorch: {torch.__version__}")

# ============================================================
# CONFIGURATION
# ============================================================
LOOKBACK   = 24
FORECAST   = 48   # predict 2 days ahead (env uses 24h detail + day-2 summary)
EPOCHS     = 60
BATCH_SIZE = 256
LR         = 0.001
EARLY_STOP = 10

# Load scale factors — maps Nigeria base profile to different community sizes
# Base Nigeria profile has mean ~192.9 kW (national grid)
# Scaled to: 5, 8, 12, 19, 28, 40 kW mean loads
ORIG_MEAN_LOAD = 192.9   # kW — original Nigeria dataset mean
TARGET_MEANS   = [5.0, 8.0, 12.0, 18.958, 28.0, 40.0]   # kW
SCALE_FACTORS  = [t / ORIG_MEAN_LOAD for t in TARGET_MEANS]

print("\nLoad augmentation scales:")
for sf, tm in zip(SCALE_FACTORS, TARGET_MEANS):
    print(f"  x{sf:.4f}  ->  {tm:.1f} kW mean")

# ============================================================
# 1. LOAD BASE DATASET (unscaled)
# ============================================================
print("\nLoading master_dataset_raw.csv (unscaled, from Phase 1)...")
df = pd.read_csv('../data/master_dataset_raw.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# Option 1 / held-out protocol: the LSTM trains ONLY on the weather-diverse
# training sites. Accra / Bolgatanga / Akosombo are never seen here (not even
# by the scalers), so Phase 5 is a genuine zero-shot generalisation test.
TRAIN_LOCS = ['Tamale', 'Kumasi', 'Axim']
df = df[df['location'].isin(TRAIN_LOCS)].reset_index(drop=True)
print(f"Training locations only: {sorted(df['location'].unique())}")
print(f"Loaded: {df.shape}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Original load mean: {df['load_kw'].mean():.2f} kW")

# location_code, hour, month, dayofweek already added by Phase 1
# Verify they are present
assert all(c in df.columns for c in ['location_code','hour','month','dayofweek']), \
    "Missing time features — did you run phase1_build_dataset_param.py first?"
print("Time features confirmed from Phase 1")

input_features = ['ssrd_wm2', 'tp', 'temp_c', 'load_kw',
                  'hour', 'month', 'dayofweek']   # Option 1: no location_code
targets        = ['ssrd_wm2', 'load_kw']

# ============================================================
# 2. BUILD AUGMENTED DATASET
# ============================================================
print("\nBuilding multi-scale augmented dataset...")
augmented_frames = []

for sf, tm in zip(SCALE_FACTORS, TARGET_MEANS):
    df_scaled = df.copy()
    df_scaled['load_kw'] = df_scaled['load_kw'] * sf
    df_scaled['scale_mean_kw'] = tm   # metadata — not used as feature
    augmented_frames.append(df_scaled)
    print(f"  Scale x{sf:.4f} -> mean load: {df_scaled['load_kw'].mean():.3f} kW")

df_aug = pd.concat(augmented_frames, ignore_index=True)
print(f"\nAugmented dataset: {df_aug.shape}")
print(f"Load range: {df_aug['load_kw'].min():.2f} to {df_aug['load_kw'].max():.2f} kW")
print(f"Load mean across all scales: {df_aug['load_kw'].mean():.3f} kW")

# ============================================================
# 3. FIT SCALERS ON AUGMENTED DATA
# ============================================================
# Scalers must see the full range so they normalise correctly
# at every load scale the LSTM will encounter during PPO training
print("\nFitting scalers on augmented data...")
df_sorted  = df_aug.sort_values(['location', 'scale_mean_kw', 'datetime']).reset_index(drop=True)
scaler_X   = MinMaxScaler()
scaler_y   = MinMaxScaler()
X_scaled   = scaler_X.fit_transform(df_sorted[input_features])
y_scaled   = scaler_y.fit_transform(df_sorted[targets])
print(f"scaler_X range: {X_scaled.min():.3f} to {X_scaled.max():.3f}")
print(f"scaler_y range: {y_scaled.min():.3f} to {y_scaled.max():.3f}")

# Save scalers for use in phase3 and phase4
with open('../models/scaler_X_param.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('../models/scaler_y_param.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)
print("Scalers saved: scaler_X_param.pkl, scaler_y_param.pkl")

# ============================================================
# 4. CREATE SEQUENCES
# ============================================================
print("\nCreating sequences...")

def create_sequences(X, y, lookback, forecast):
    Xs, ys = [], []
    for i in range(len(X) - lookback - forecast + 1):
        Xs.append(X[i:i + lookback])
        ys.append(y[i + lookback:i + lookback + forecast])
    return np.array(Xs), np.array(ys)

all_X, all_y = [], []

# Create sequences per location per scale — no cross-contamination
for loc in ['Tamale', 'Kumasi', 'Axim']:
    for sf, tm in zip(SCALE_FACTORS, TARGET_MEANS):
        loc_idx = df_sorted[
            (df_sorted['location'] == loc) &
            (df_sorted['scale_mean_kw'] == tm)
        ].index

        if len(loc_idx) < LOOKBACK + FORECAST:
            continue

        X_s, y_s = create_sequences(
            X_scaled[loc_idx.values], y_scaled[loc_idx.values], LOOKBACK, FORECAST)
        all_X.append(X_s)
        all_y.append(y_s)

X_seq = np.concatenate(all_X)
y_seq = np.concatenate(all_y)
print(f"Total sequences: {X_seq.shape[0]:,}")
print(f"Sequence shape: X={X_seq.shape} y={y_seq.shape}")

# ============================================================
# 5. TRAIN / VAL / TEST SPLIT (70/15/15)
# ============================================================
np.random.seed(42)
idx   = np.random.permutation(len(X_seq))
X_seq = X_seq[idx]; y_seq = y_seq[idx]
n     = len(X_seq)
X_train, y_train = X_seq[:int(n * .70)],           y_seq[:int(n * .70)]
X_val,   y_val   = X_seq[int(n * .70):int(n * .85)], y_seq[int(n * .70):int(n * .85)]
X_test,  y_test  = X_seq[int(n * .85):],            y_seq[int(n * .85):]
print(f"\nSplit: Train={len(X_train):,} Val={len(X_val):,} Test={len(X_test):,}")

# ============================================================
# 6. DATALOADERS
# ============================================================
train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ============================================================
# 7. LSTM MODEL (identical architecture to old system)
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
model = MiniGridLSTM(7, 128, 2, FORECAST, 2).to(device)
print(f"LSTM parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 8. TRAINING
# ============================================================
criterion  = nn.MSELoss()
optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
             optimizer, patience=3, factor=0.5)

best_val    = float('inf')
patience_ct = 0
train_losses = []
val_losses   = []

print(f"\nTraining for up to {EPOCHS} epochs...")
print("-" * 60)

for epoch in range(EPOCHS):
    # Train
    model.train()
    t_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_loss += loss.item()
    t_loss /= len(train_loader)

    # Validate
    model.eval()
    v_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            v_loss += criterion(model(Xb), yb).item()
    v_loss /= len(val_loader)

    scheduler.step(v_loss)
    train_losses.append(t_loss)
    val_losses.append(v_loss)

    tag = ""
    if v_loss < best_val:
        best_val    = v_loss
        patience_ct = 0
        torch.save(model.state_dict(), '../models/best_lstm_param.pth')
        tag = "  <- saved"
    else:
        patience_ct += 1

    if (epoch + 1) % 5 == 0 or tag:
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}]  "
              f"Train: {t_loss:.6f}  Val: {v_loss:.6f}{tag}")

    if patience_ct >= EARLY_STOP:
        print(f"\nEarly stopping at epoch {epoch + 1}")
        break

print("-" * 60)
print(f"Best validation loss: {best_val:.6f}")
print("Saved: ../models/best_lstm_param.pth")

# ============================================================
# 9. TRAINING CURVE
# ============================================================
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Train', color='blue',   linewidth=2)
plt.plot(val_losses,   label='Val',   color='orange', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('LSTM Training — Parameterised Multi-Scale (5 to 40 kW mean load)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../data/lstm_param_training_curve.png', dpi=150)
plt.show()
print("Training curve saved: ../data/lstm_param_training_curve.png")

# ============================================================
# 10. TEST EVALUATION PER SCALE
# ============================================================
model.load_state_dict(torch.load('../models/best_lstm_param.pth',
                                  map_location='cpu', weights_only=True))
model.eval()
model.to('cpu')

print("\nTest Set Performance:")
print("-" * 50)
preds_h, actuals_h = [], []
with torch.no_grad():
    for i in range(0, len(X_test), BATCH_SIZE):
        out = model(torch.FloatTensor(X_test[i:i+BATCH_SIZE])).numpy()  # (B, 48, 2)
        lbl = y_test[i:i+BATCH_SIZE]                                    # (B, 48, 2)
        preds_h.append(out)
        actuals_h.append(lbl)

preds_h   = np.concatenate(preds_h,   0)   # (N, 48, 2) normalised
actuals_h = np.concatenate(actuals_h, 0)
N, H, _ = preds_h.shape

# inverse-transform to physical units, preserving horizon structure
preds   = scaler_y.inverse_transform(preds_h.reshape(-1, 2)).reshape(N, H, 2)
actuals = scaler_y.inverse_transform(actuals_h.reshape(-1, 2)).reshape(N, H, 2)

# ---- overall + day-1 vs day-2 split ----
for i, name in enumerate(['Solar (W/m2)', 'Load (kW)']):
    mean_v = max(actuals[:, :, i].mean(), 1.0)
    mae_all = np.abs(actuals[:, :, i] - preds[:, :, i]).mean()
    mae_d1  = np.abs(actuals[:, :24, i] - preds[:, :24, i]).mean()
    mae_d2  = np.abs(actuals[:, 24:, i] - preds[:, 24:, i]).mean()
    print(f"{name}:")
    print(f"    all 48h : MAE={mae_all:8.3f}  ({mae_all/mean_v*100:4.1f}% of mean)")
    print(f"    day 1   : MAE={mae_d1:8.3f}  ({mae_d1/mean_v*100:4.1f}% of mean)")
    print(f"    day 2   : MAE={mae_d2:8.3f}  ({mae_d2/mean_v*100:4.1f}% of mean)")

# ---- day-2 SUMMARY feature skill (the actual value fed to the policy) ----
# Compare the model's day-2 mean to a 'persistence' guess (day-2 mean = day-1 mean).
print("\nDay-2 summary feature (mean over hours 24:48) — is it better than 'tomorrow = today'?")
for i, name in enumerate(['Solar', 'Load']):
    pred_d2_mean = preds[:, 24:, i].mean(axis=1)     # model's day-2 mean per sample
    act_d2_mean  = actuals[:, 24:, i].mean(axis=1)
    act_d1_mean  = actuals[:, :24, i].mean(axis=1)   # persistence baseline
    mae_model = np.abs(act_d2_mean - pred_d2_mean).mean()
    mae_persist = np.abs(act_d2_mean - act_d1_mean).mean()
    verdict = "model adds skill" if mae_model < mae_persist else "no better than persistence"
    print(f"    {name:<6}: model MAE={mae_model:8.3f}  vs persistence MAE={mae_persist:8.3f}  -> {verdict}")

# ---- per-horizon MAE curve ----
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for i, name in enumerate(['Solar (W/m2)', 'Load (kW)']):
    per_h = np.abs(actuals[:, :, i] - preds[:, :, i]).mean(axis=0)  # (48,)
    axes[i].plot(range(1, H + 1), per_h, color='#2563eb', linewidth=1.8)
    axes[i].axvline(24, color='#ef4444', linestyle='--', alpha=0.7, label='day-1 / day-2')
    axes[i].set_title(f'{name} — MAE by forecast horizon')
    axes[i].set_xlabel('Hours ahead'); axes[i].set_ylabel('MAE'); axes[i].grid(True, alpha=0.3)
    axes[i].legend()
plt.tight_layout()
plt.savefig('../data/lstm_param_horizon_error.png', dpi=150)
print("\nPer-horizon error curve saved: ../data/lstm_param_horizon_error.png")

print("\nDone! Run phase3_parametrised_env.py next.")
