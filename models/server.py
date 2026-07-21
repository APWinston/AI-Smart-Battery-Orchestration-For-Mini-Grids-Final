#!/usr/bin/env python
# coding: utf-8
"""
server.py — SREP Mini-Grid AI Battery EMS (Route B backend)
===========================================================
Runs the REAL trained models (LSTM .pth + SB3 PPO .zip) in a background
real-time controller (1 wall-clock hour = 1 operating hour). The run lives in
THIS process — close the browser, reopen, switch devices: the run continues.

    pip install fastapi uvicorn torch stable-baselines3 scikit-learn numpy
    python server.py
    -> open http://localhost:8000

Endpoints:
    GET  /            the twin front-end (twin.html next to this file)
    POST /api/start   {site_name, lat, lon, kwp, kwh, load, soc}  -> starts a 24 h run
    GET  /api/state   full live state (engine truth) — poll this
    POST /api/stop    stop the current run
"""
import os, json, time, math, threading, urllib.request, datetime as dt
import numpy as np

# ============================================================
# PATHS  (script-dir relative; server.py lives in your models/ folder)
# ============================================================
HERE        = os.path.dirname(os.path.abspath(__file__))
LSTM_PTH    = os.path.join(HERE, "best_lstm_param.pth")
SCALER_X    = os.path.join(HERE, "scaler_X_param.pkl")
SCALER_Y    = os.path.join(HERE, "scaler_y_param.pkl")
PPO_CANDIDATES = [  # first pair found wins
    (os.path.join(HERE, "best", "best_model.zip"), os.path.join(HERE, "best", "vecnormalize_param.pkl")),
    (os.path.join(HERE, "ppo_srep_param.zip"),     os.path.join(HERE, "vecnormalize_param.pkl")),
]
TWIN_HTML   = os.path.join(HERE, "twin.html")
TICK_SEC    = 2.0
RUN_HOURS   = 24.0

# ============================================================
# CONSTANTS  (verified against phase3_parametrised_env.py)
# ============================================================
C = dict(residual_scale=0.25, pv_derating=0.75, soc_min=0.10, soc_max=0.95, eta=0.95,
         max_solar_kwp=120.0, max_bat_kwh=400.0, max_load_kw=20.0, obs_hourly=24,
         vn_clip=10.0, vn_eps=1e-8)
LOAD_SHAPE = [.45,.40,.38,.38,.42,.55,.85,1.05,.95,.82,.78,.80,
              .82,.80,.78,.82,.95,1.35,1.95,2.20,2.05,1.55,.95,.60]
FORECAST = 48

# ============================================================
# MODELS
# ============================================================
def _build_lstm():
    import torch.nn as nn, torch
    class MiniGridLSTM(nn.Module):
        def __init__(s, inp, hid, nl, fc, out):
            super().__init__(); s.hidden_size, s.num_layers = hid, nl
            s.lstm = nn.LSTM(inp, hid, nl, batch_first=True, dropout=0.2)
            s.fc = nn.Linear(hid, fc*out); s.fc_out, s.fc_h = fc, out
        def forward(s, x):
            import torch
            h0 = torch.zeros(s.num_layers, x.size(0), s.hidden_size)
            c0 = torch.zeros(s.num_layers, x.size(0), s.hidden_size)
            o, _ = s.lstm(x, (h0, c0))
            return s.fc(o[:, -1, :]).view(-1, s.fc_out, s.fc_h)
    return MiniGridLSTM(7, 128, 2, FORECAST, 2)

def load_models():
    M = {"lstm": None, "scaler_X": None, "scaler_y": None, "ppo": None,
         "vn_mean": None, "vn_var": None, "vn_clip": C["vn_clip"], "vn_eps": C["vn_eps"],
         "have_lstm": False, "have_ppo": False, "msg": []}
    try:
        import torch, pickle
        if all(os.path.exists(p) for p in (LSTM_PTH, SCALER_X, SCALER_Y)):
            m = _build_lstm()
            m.load_state_dict(torch.load(LSTM_PTH, map_location="cpu", weights_only=True)); m.eval()
            M["lstm"] = m
            M["scaler_X"] = pickle.load(open(SCALER_X, "rb"))
            M["scaler_y"] = pickle.load(open(SCALER_Y, "rb"))
            M["have_lstm"] = True
        else:
            M["msg"].append("LSTM/scalers not found — weather-derived forecast fallback.")
    except Exception as e:
        M["msg"].append(f"LSTM load failed: {e}")
    try:
        import pickle
        for zp, vp in PPO_CANDIDATES:
            if os.path.exists(zp) and os.path.exists(vp):
                from stable_baselines3 import PPO
                M["ppo"] = PPO.load(zp, device="cpu")
                vn = pickle.load(open(vp, "rb"))
                M["vn_mean"] = np.asarray(vn.obs_rms.mean, dtype=np.float64)
                M["vn_var"] = np.asarray(vn.obs_rms.var, dtype=np.float64)
                M["vn_clip"] = float(getattr(vn, "clip_obs", C["vn_clip"]))
                M["vn_eps"] = float(getattr(vn, "epsilon", C["vn_eps"]))
                M["have_ppo"] = True
                M["msg"].append(f"PPO loaded: {os.path.relpath(zp, HERE)}")
                break
        else:
            M["msg"].append("PPO/VecNormalize not found — baseline load-follower only.")
    except Exception as e:
        M["msg"].append(f"PPO load failed: {e}")
    return M

# ============================================================
# WEATHER  (72 h: yesterday | today | tomorrow; 15-min cache)
# ============================================================
_wx_cache = {}
def get_weather(lat, lon):
    key = (round(lat, 3), round(lon, 3))
    hit = _wx_cache.get(key)
    if hit and time.time() - hit[0] < 900:
        return hit[1]
    try:
        url = ("https://api.open-meteo.com/v1/forecast"
               f"?latitude={lat}&longitude={lon}"
               "&hourly=shortwave_radiation,temperature_2m,precipitation"
               "&past_days=2&forecast_days=2&timezone=auto")
        with urllib.request.urlopen(url, timeout=20) as r:
            j = json.loads(r.read().decode())
        h = j["hourly"]; off = j.get("utc_offset_seconds", 0)
        now_loc = dt.datetime.utcfromtimestamp(time.time() + off)
        today = now_loc.strftime("%Y-%m-%dT")
        idx = next((i for i, t in enumerate(h["time"]) if t.startswith(today)), 24)
        start = idx - 24; n = len(h["shortwave_radiation"])
        def grab(kk, fb):
            src = h[kk]
            return [float(src[i]) if 0 <= i < n and src[i] is not None else fb
                    for i in range(start, start + 72)]
        DAY = dict(ghi=grab("shortwave_radiation", 0.0), temp=grab("temperature_2m", 26.0),
                   ppt=grab("precipitation", 0.0), month=now_loc.month, dow=now_loc.weekday(),
                   date=now_loc.strftime("%Y-%m-%d"),
                   local_hour=now_loc.hour + now_loc.minute/60.0, ok=True)
    except Exception:
        DAY = dict(ghi=[max(0.0, math.sin(math.pi*((x % 24)-6)/12))**1.15*1050 for x in range(72)],
                   temp=[26+6*math.sin(math.pi*((x % 24)-9)/12) for x in range(72)],
                   ppt=[0.0]*72, month=dt.datetime.now().month, dow=dt.datetime.now().weekday(),
                   date="modeled", local_hour=dt.datetime.now().hour + dt.datetime.now().minute/60.0,
                   ok=False)
    _wx_cache[key] = (time.time(), DAY)
    return DAY

# ============================================================
# ENGINE  (verified 1:1 against the training environment)
# ============================================================
def day_val(arr, h):
    n = len(arr); i = max(0, min(n-1, int(h))); j = min(n-1, i+1); f = h - int(h)
    return arr[i]*(1-f) + arr[j]*f

def load_kw(h, tier):
    hh = h % 24; i = int(hh) % 24; j = (i+1) % 24; f = hh - int(hh)
    return tier["load"]*(LOAD_SHAPE[i]*(1-f)+LOAD_SHAPE[j]*f)

def window24(abs_k, DAY, tier):
    rows = []
    for n in range(24):
        a = abs_k - 24 + n
        idx = max(0, min(71, int(a)))
        rows.append([DAY["ghi"][idx], DAY["ppt"][idx], DAY["temp"][idx],
                     tier["load"]*LOAD_SHAPE[int(a) % 24], int(a) % 24, DAY["month"], DAY["dow"]])
    return np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0)

MC_PASSES = 8   # Monte-Carlo dropout ensemble size (uses the LSTM's trained dropout)

def forecast(abs_k, DAY, tier, M, bias=None):
    """Ensemble forecast: MC-dropout passes averaged (ensemble mean) with an
    uncertainty-derived confidence in [0,1]. Optional online bias correction
    (learned from production errors; no weight retraining)."""
    if M["have_lstm"]:
        import torch
        Xs = M["scaler_X"].transform(window24(abs_k, DAY, tier))
        x = torch.tensor(Xs, dtype=torch.float32).unsqueeze(0)
        outs = []
        with torch.no_grad():
            M["lstm"].train()          # enable dropout for stochastic passes
            for _ in range(MC_PASSES):
                outs.append(M["lstm"](x).squeeze(0).numpy())
            M["lstm"].eval()
        arr = np.clip(np.nan_to_num(np.stack(outs), nan=0.0), 0.0, 1.0)
        out = arr.mean(axis=0)                       # ensemble average
        std = float(arr.std(axis=0)[:6].mean())      # near-term spread
        conf = float(np.clip(1.0 - std/0.15, 0.0, 1.0))   # 0.15 norm-units ~ fully uncertain
        if bias:                                     # online production-error correction
            out[:, 0] = np.clip(out[:, 0] + bias.get("solar", 0.0), 0.0, 1.0)
            out[:, 1] = np.clip(out[:, 1] + bias.get("load", 0.0), 0.0, 1.0)
        phys = M["scaler_y"].inverse_transform(out)
        return out[:, 0], out[:, 1], phys, conf
    sol, ld = [], []
    for kk in range(FORECAST):
        hh = (abs_k + kk) % 24
        sol.append(min(1.0, day_val(DAY["ghi"], (abs_k + kk)) / 1000.0))
        ld.append(min(1.0, load_kw(hh, tier)/max(tier["load"]*2.2, 1)))
    sol, ld = np.array(sol), np.array(ld)
    phys = np.stack([sol*1000.0, ld*max(tier["load"]*2.2, 1)], axis=1)
    return sol, ld, phys, None

def build_obs(sol_n, ld_n, hod, month, base_act, soc, soh, tier):
    h2 = C["obs_hourly"]
    d2s = float(np.mean(sol_n[h2:])); d2l = float(np.mean(ld_n[h2:]))
    return np.asarray([soc, soh,
        math.sin(2*math.pi*hod/24), math.cos(2*math.pi*hod/24),
        math.sin(2*math.pi*month/12), math.cos(2*math.pi*month/12),
        *sol_n[:h2].tolist(), *ld_n[:h2].tolist(), d2s, d2l,
        tier["kwp"]/C["max_solar_kwp"], tier["kwh"]/C["max_bat_kwh"],
        tier["load"]/C["max_load_kw"], base_act], dtype=np.float64)

def ppo_action(obs, M):
    if not M["have_ppo"]:
        return 0.0
    obs = np.nan_to_num(np.asarray(obs, dtype=np.float64), nan=0.0)
    on = np.clip((obs - M["vn_mean"]) / np.sqrt(M["vn_var"] + M["vn_eps"]),
                 -M["vn_clip"], M["vn_clip"]).astype(np.float32)
    on = np.nan_to_num(on, nan=0.0, posinf=M["vn_clip"], neginf=-M["vn_clip"])
    a, _ = M["ppo"].predict(on, deterministic=True)
    return float(np.clip(np.ravel(a)[0], -1.0, 1.0))

def step(dt_h, abs_h, soc, soh, DAY, tier, M, maxP, bias=None, want_xai=False):
    if soc is None: soc = 0.5
    if soh is None: soh = 1.0
    hod = abs_h % 24
    ghi = day_val(DAY["ghi"], abs_h)
    solar = (ghi/1000.0)*tier["kwp"]*C["pv_derating"]
    load = load_kw(hod, tier)
    sol_n, ld_n, phys, conf = forecast(int(abs_h)+1, DAY, tier, M, bias=bias)
    base_act = float(np.clip((solar-load)/maxP, -1.0, 1.0))
    obs = build_obs(sol_n, ld_n, hod, DAY["month"], base_act, soc, soh, tier)
    a = ppo_action(obs, M)
    # confidence-aware residual: at low forecast confidence, trust the safe
    # load-following term more and the learned residual less
    trust = 1.0 if conf is None else (0.5 + 0.5*conf)
    act = float(np.clip(base_act + C["residual_scale"]*trust*a, -1.0, 1.0))
    drivers = []
    if want_xai and M["have_ppo"]:
        # local sensitivity (finite differences): what is pushing this decision
        probes = [("Battery SoC", 0, 0.05), ("Next hours solar", slice(6, 9), 0.05),
                  ("Next hours load", slice(30, 33), 0.05)]
        for name, idx, eps in probes:
            ob2 = obs.copy(); ob2[idx] = np.clip(ob2[idx] + eps, 0, 1)
            da = ppo_action(ob2, M) - a
            drivers.append(dict(name=name,
                                impact_kw=round(C["residual_scale"]*trust*da*maxP, 2)))
        drivers.sort(key=lambda d: -abs(d["impact_kw"]))
    power = act*maxP
    cap = tier["kwh"]*soh; stored = soc*cap
    if power >= 0:
        power = min(power, (C["soc_max"]-soc)*cap/C["eta"]/max(dt_h, 1e-3))
    else:
        power = max(power, -(soc-C["soc_min"])*cap*C["eta"]/max(dt_h, 1e-3))
    supply = solar+max(0.0, -power); demand = load+max(0.0, power)
    unmet = 0.0 if supply >= demand else min(demand-supply, load)
    ac = max(0.0, power); ad = max(0.0, -power)
    soc = float(np.clip((stored+(ac*C["eta"]-ad/C["eta"])*dt_h)/cap, C["soc_min"], C["soc_max"]))
    soh = max(0.80, soh-(ac+ad)*dt_h*2e-7)
    fc_solar = (phys[:3, 0]/1000.0)*tier["kwp"]*C["pv_derating"]
    return dict(soc=soc, soh=soh, solar=solar, load=load, power=power, unmet=unmet,
                ghi=ghi, base_act=base_act, residual=C["residual_scale"]*trust*a, act=act,
                conf=(None if conf is None else round(conf, 3)), drivers=drivers,
                fc_solar=[round(float(v), 1) for v in fc_solar],
                fc_load=[round(float(v), 1) for v in phys[:3, 1]])

def make_plan(abs_h0, soc0, soh0, DAY, tier, M, maxP):
    plan, soc, soh = [], float(soc0), float(soh0)
    for i in range(24):
        a = abs_h0 + i
        s = step(1.0, a, soc, soh, DAY, tier, M, maxP)
        soc, soh = s["soc"], s["soh"]
        plan.append(dict(clock=f"{int(a)%24:02d}:00",
                         solar=round(s["solar"], 1), load=round(s["load"], 1),
                         power=round(s["power"], 1), soc=round(soc*100, 1),
                         unmet=round(s["unmet"], 1),
                         action=("charge" if s["power"] > 0.5 else
                                 "discharge" if s["power"] < -0.5 else "hold")))
    return plan

# ============================================================
# CONTROLLER  (the run lives here — a background real-time thread)
# ============================================================
class Controller:
    def __init__(self, M):
        self.M = M
        self.lock = threading.Lock()
        self.thread = None
        self.stop_flag = False
        self.bias = {"solar": 0.0, "load": 0.0}      # online production-error correction
        self._prev_fc = None                          # (abs_hour_int, fc_solar_norm, fc_load_norm)
        self.state = {"status": "idle",
                      "models": {"have_lstm": M["have_lstm"], "have_ppo": M["have_ppo"],
                                 "msg": M["msg"]}}

    def hhmm(self, h): return f"{int(h)%24:02d}:{int((h%1)*60):02d}"

    def start(self, cfg):
        self.stop()
        tier = dict(kwp=float(cfg["kwp"]), kwh=float(cfg["kwh"]), load=float(cfg["load"]))
        maxP = min(0.5*tier["kwh"], 0.8*tier["kwp"])
        DAY = get_weather(cfg["lat"], cfg["lon"])
        abs0 = 24.0 + DAY["local_hour"]
        soc0 = float(cfg["soc"])/100.0
        plan = make_plan(abs0, soc0, 1.0, DAY, tier, self.M, maxP)
        self._prev_fc = None
        with self.lock:
            self.state = {
                "status": "running",
                "models": {"have_lstm": self.M["have_lstm"], "have_ppo": self.M["have_ppo"],
                           "msg": self.M["msg"]},
                "cfg": {"site_name": cfg.get("site_name", "Custom"), "lat": cfg["lat"],
                        "lon": cfg["lon"], **tier, "maxP": maxP, "soc0": soc0},
                "weather": {"date": DAY["date"], "ok": DAY["ok"], "ghi_today": DAY["ghi"][24:48]},
                "clock": {"abs": abs0, "start_abs": abs0, "hod": abs0 % 24, "elapsed": 0.0},
                "anchor_date": DAY["date"],
                "live": None, "plan": plan,
                "hist": {"solar": [None]*24, "load": [None]*24, "power": [None]*24, "soc": [None]*24},
                "kpi": {"served": 0.0, "loadE": 0.0, "lol": 0.0, "efc": 0.0},
                "today": {"date": DAY["date"], "served": 0.0, "loadE": 0.0, "lol": 0.0},
                "days": [],
                "log": [f"{self.hhmm(abs0)} · run started · SoC {cfg['soc']}% (operator) · "
                        f"{tier['kwp']:.0f} kWp / {tier['kwh']:.0f} kWh / {tier['load']:.1f} kW mean"],
                "soc": soc0, "soh": 1.0,
                "started_iso": dt.datetime.now().isoformat(timespec="seconds")}
        self.stop_flag = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return {"ok": True, "plan": plan}

    def stop(self):
        self.stop_flag = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=TICK_SEC*3)
        self.thread = None
        with self.lock:
            if self.state.get("status") == "running":
                self.state["status"] = "stopped"

    def _run(self):
        last = time.time()
        while not self.stop_flag:
            time.sleep(TICK_SEC)
            now = time.time()
            dt_h = min(max(now - last, 0.0)/3600.0, 1.0)
            last = now
            # read coords under a brief lock, fetch weather OUTSIDE the lock
            # (a cold cache blocks up to 20 s; /api/state must never wait on that)
            with self.lock:
                if self.state.get("status") != "running":
                    return
                _lat = self.state["cfg"]["lat"]; _lon = self.state["cfg"]["lon"]
            DAY = get_weather(_lat, _lon)
            with self.lock:
                S = self.state
                if S["status"] != "running":
                    return
                cfg = S["cfg"]; tier = {k: cfg[k] for k in ("kwp", "kwh", "load")}
                # midnight axis re-anchor
                if DAY["date"] not in (S["anchor_date"], "modeled"):
                    try:
                        dsh = (dt.date.fromisoformat(DAY["date"])
                               - dt.date.fromisoformat(S["anchor_date"])).days
                    except Exception:
                        dsh = 1
                    S["clock"]["abs"] -= 24*dsh; S["clock"]["start_abs"] -= 24*dsh
                    S["anchor_date"] = DAY["date"]
                S["weather"] = {"date": DAY["date"], "ok": DAY["ok"],
                                "ghi_today": DAY["ghi"][24:48]}
                S["clock"]["abs"] += dt_h
                abs_h = S["clock"]["abs"]
                elapsed = abs_h - S["clock"]["start_abs"]
                S["clock"]["hod"] = abs_h % 24
                S["clock"]["elapsed"] = elapsed
                d = step(dt_h, abs_h, S["soc"], S["soh"], DAY, tier, self.M, cfg["maxP"],
                         bias=self.bias, want_xai=True)
                S["soc"], S["soh"] = d["soc"], d["soh"]
                hod = int(abs_h) % 24
                S["hist"]["solar"][hod] = round(d["solar"], 2)
                S["hist"]["load"][hod] = round(d["load"], 2)
                S["hist"]["power"][hod] = round(d["power"], 2)
                S["hist"]["soc"][hod] = round(S["soc"]*100, 1)
                for K in (S["kpi"], S["today"]):
                    K["served"] += (d["load"]-d["unmet"])*dt_h
                    K["loadE"] += d["load"]*dt_h
                    if d["unmet"] > 0.5: K["lol"] += dt_h*60
                S["kpi"]["efc"] += abs(d["power"])*dt_h
                prev_hod = int(abs_h - dt_h) % 24
                if hod != prev_hod:
                    # ---- online learning from production data (no retraining) ----
                    # compare last hour's 1h-ahead forecast against what actually happened
                    if self._prev_fc is not None:
                        _, pfs, pfl = self._prev_fc
                        act_sol_n = min(1.0, day_val(DAY["ghi"], abs_h)/1000.0)
                        act_ld_n = min(1.0, d["load"]/max(tier["load"]*2.2, 1))
                        alpha = 0.2   # EMA
                        self.bias["solar"] = float(np.clip(
                            (1-alpha)*self.bias["solar"] + alpha*(act_sol_n - pfs), -0.10, 0.10))
                        self.bias["load"] = float(np.clip(
                            (1-alpha)*self.bias["load"] + alpha*(act_ld_n - pfl), -0.10, 0.10))
                    try:
                        fs_n = min(1.0, (d["fc_solar"][0]/max(tier["kwp"]*C["pv_derating"], 1e-6)))
                        fl_n = min(1.0, d["fc_load"][0]/max(tier["load"]*2.2, 1))
                        self._prev_fc = (int(abs_h), fs_n, fl_n)
                    except Exception:
                        self._prev_fc = None
                    # ---- production log + active-learning queue ----
                    try:
                        os.makedirs(ASSETS_DIR, exist_ok=True)
                        pl = os.path.join(ASSETS_DIR, "production_log.csv")
                        newf = not os.path.exists(pl)
                        with open(pl, "a") as f:
                            if newf:
                                f.write("timestamp,site,hour,ghi_wm2,solar_kw,load_kw,"
                                        "battery_kw,soc_pct,unmet_kw,confidence,"
                                        "bias_solar,bias_load\n")
                            f.write(",".join(str(x) for x in [
                                dt.datetime.now().isoformat(timespec="seconds"),
                                cfg["site_name"], hod, round(d["ghi"], 1),
                                round(d["solar"], 2), round(d["load"], 2),
                                round(d["power"], 2), round(S["soc"]*100, 1),
                                round(d["unmet"], 2),
                                d.get("conf") if d.get("conf") is not None else "",
                                round(self.bias["solar"], 4), round(self.bias["load"], 4)]) + "\n")
                        cf = d.get("conf")
                        if (cf is not None and cf < 0.6) or d["unmet"] > 0.5:
                            q = os.path.join(ASSETS_DIR, "active_learning_queue.csv")
                            newq = not os.path.exists(q)
                            with open(q, "a") as f:
                                if newq:
                                    f.write("timestamp,site,hour,reason,confidence,unmet_kw\n")
                                f.write(",".join(str(x) for x in [
                                    dt.datetime.now().isoformat(timespec="seconds"),
                                    cfg["site_name"], hod,
                                    ("low_confidence" if (cf is not None and cf < 0.6) else "load_shed"),
                                    cf if cf is not None else "", round(d["unmet"], 2)]) + "\n")
                    except Exception:
                        pass
                    if hod == 0:                              # midnight: archive the day
                        sp = S["today"]["served"]/S["today"]["loadE"]*100 if S["today"]["loadE"] > 0 else 100
                        S["days"].append({"date": S["today"]["date"], "served": round(sp, 1),
                                          "lol": round(S["today"]["lol"], 0)})
                        S["log"].append(f"00:00 · day closed · served {sp:.1f}% · LoL {S['today']['lol']:.0f} min")
                        S["today"] = {"date": DAY["date"], "served": 0.0, "loadE": 0.0, "lol": 0.0}
                    spd = S["today"]["served"]/S["today"]["loadE"]*100 if S["today"]["loadE"] > 0 else 100
                    S["log"].append(f"{self.hhmm(abs_h)} · SoC {S['soc']*100:.0f}% · served today {spd:.0f}%"
                                    + (f" · ⚠ shed {d['unmet']:.0f} kW" if d["unmet"] > 0.5 else ""))
                    del S["log"][:-40]
                srv = lambda K: K["served"]/K["loadE"]*100 if K["loadE"] > 0 else 100.0
                S["live"] = {**{k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()},
                             "served_day": round(srv(S["today"]), 1),
                             "served_run": round(srv(S["kpi"]), 1),
                             "lol_day": round(S["today"]["lol"], 1),
                             "efc": round(S["kpi"]["efc"]/(2*tier["kwh"]), 3),
                             "updated_iso": dt.datetime.now().isoformat(timespec="seconds")}
                if elapsed >= RUN_HOURS:
                    S["status"] = "done"
                    S["log"].append(f"{self.hhmm(abs_h)} · ✔ 24 h run complete · "
                                    f"served {srv(S['kpi']):.1f}% · LoL {S['kpi']['lol']:.0f} min · "
                                    f"{S['kpi']['efc']/(2*tier['kwh']):.2f} EFC")
                    return

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.state))

# ============================================================
# API
# ============================================================
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
import secrets

# ---- operator login ----
# Locally these defaults apply; in production set OPERATOR_USER / OPERATOR_PASS
# as environment variables (e.g. in the Render dashboard) so they are not in the code.
OPERATOR_USER = os.environ.get("OPERATOR_USER", "operator")
OPERATOR_PASS = os.environ.get("OPERATOR_PASS", "srep2026")
TOKENS = set()

class LoginCfg(BaseModel):
    username: str
    password: str

class StartCfg(BaseModel):
    site_name: str = "Custom"
    lat: float
    lon: float
    kwp: float
    kwh: float
    load: float
    soc: float          # percent, 10..95

app = FastAPI(title="SREP AI Battery EMS")
MODELS = load_models()
CTRL = Controller(MODELS)

# ---- landing slide images: served same-origin from assets/ ----
# Drop your own photos as assets/slide1.jpg .. slide3.jpg (they take priority).
# Missing ones are fetched once from Unsplash at boot (best effort).
ASSETS_DIR = os.path.join(HERE, "assets")
_SLIDE_SOURCES = {
    "slide1.jpg": "https://images.unsplash.com/photo-1509391366360-2e959784a276?q=80&w=1920&auto=format&fit=crop",
    "slide2.jpg": "https://images.unsplash.com/photo-1508514177221-188b1cf16e9d?q=80&w=1920&auto=format&fit=crop",
    "slide3.jpg": "https://images.unsplash.com/photo-1509389928833-fe62aef36deb?q=80&w=1920&auto=format&fit=crop",
    "knust_logo.png": "https://commons.wikimedia.org/wiki/Special:FilePath/Wikitech-knust-logo1.png?width=256",
}
def _fetch_slides():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    # a knust_logo.png placed next to server.py always takes priority
    _local = os.path.join(HERE, "knust_logo.png")
    _dst = os.path.join(ASSETS_DIR, "knust_logo.png")
    try:
        if os.path.isfile(_local) and (not os.path.isfile(_dst)
                or os.path.getsize(_local) != os.path.getsize(_dst)):
            import shutil
            shutil.copy(_local, _dst)
            print("  • using local knust_logo.png")
    except Exception as e:
        print("  • logo sync failed:", e)
    for name, url in _SLIDE_SOURCES.items():
        path = os.path.join(ASSETS_DIR, name)
        if os.path.exists(path):
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as r:
                data = r.read()
            if len(data) > 4000:
                with open(path, "wb") as f:
                    f.write(data)
                print(f"  • fetched {name} ({len(data)//1024} KB)")
        except Exception as e:
            print(f"  • could not fetch {name}: {e}")
threading.Thread(target=_fetch_slides, daemon=True).start()

@app.get("/assets/{name}")
def assets(name: str):
    safe = os.path.basename(name)
    path = os.path.join(ASSETS_DIR, safe)
    if os.path.isfile(path):
        return FileResponse(path, headers={"Cache-Control": "no-store"})
    raise HTTPException(status_code=404, detail="asset not found")

def _require(x_auth: str):
    if x_auth not in TOKENS:
        raise HTTPException(status_code=401, detail="login required")

@app.post("/api/login")
def api_login(cfg: LoginCfg):
    if cfg.username == OPERATOR_USER and cfg.password == OPERATOR_PASS:
        t = secrets.token_hex(16)
        TOKENS.add(t)
        return {"ok": True, "token": t}
    raise HTTPException(status_code=403, detail="invalid credentials")

@app.get("/")
def root():
    if os.path.isfile(TWIN_HTML):
        return FileResponse(TWIN_HTML)
    return HTMLResponse("<h3 style='font-family:monospace'>Backend running. "
                        "twin.html not found next to server.py.</h3>")

@app.post("/api/start")
def api_start(cfg: StartCfg, x_auth: str = Header(default="")):
    _require(x_auth)
    c = cfg.dict()
    c["soc"] = max(10.0, min(95.0, c["soc"]))
    return JSONResponse(CTRL.start(c))

@app.get("/api/production-log")
def production_log():
    """Public export of logged inference data (weather, dispatch, confidence).
    Contains no personal data."""
    p = os.path.join(ASSETS_DIR, "production_log.csv")
    if os.path.isfile(p):
        return FileResponse(p, media_type="text/csv",
                            headers={"Cache-Control": "no-store"})
    raise HTTPException(status_code=404, detail="no production data logged yet")

@app.get("/api/state")
def api_state():
    return JSONResponse(CTRL.snapshot())

@app.post("/api/stop")
def api_stop(x_auth: str = Header(default="")):
    _require(x_auth)
    CTRL.stop()
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    print("SREP AI Battery EMS · backend")
    print(f"  login: {OPERATOR_USER} / {OPERATOR_PASS}   (edit OPERATOR_USER/PASS at the top of server.py)")
    for m in MODELS["msg"]:
        print("  •", m)
    print(f"  models: LSTM={'✓' if MODELS['have_lstm'] else '✗'}  PPO={'✓' if MODELS['have_ppo'] else '✗'}")
    print("  open http://localhost:" + os.environ.get("PORT","8000"))
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
