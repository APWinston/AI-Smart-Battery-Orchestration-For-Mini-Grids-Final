#!/usr/bin/env python
# coding: utf-8
"""
app.py — SREP mini-grid digital twin (single Streamlit app)
===========================================================
3D environment + the REAL trained models running directly in Python.
No JSON: weights load from best_lstm_param.pth and the SB3 PPO .zip; all live
state lives in st.session_state. The Three.js scene is re-seeded each refresh
with the current real-model state; sun/particles are clock-derived so reseeds
are seamless.

Run:
    pip install -r requirements.txt
    streamlit run app.py

If the model artifacts aren't found, the app still runs in BASELINE mode
(load-following, no PPO) so you can see the 3D scene immediately.
"""
import os, time, math
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# CONFIG — EDIT PATHS
# ============================================================
MODELS_DIR  = "../models"
LSTM_PTH    = os.path.join(MODELS_DIR, "best_lstm_param.pth")
SCALER_X    = os.path.join(MODELS_DIR, "scaler_X_param.pkl")
SCALER_Y    = os.path.join(MODELS_DIR, "scaler_y_param.pkl")
# Phase-4 saves the deployment ("best") policy here; the final-run pair is
# ppo_srep_param.zip / vecnormalize_param.pkl in MODELS_DIR if you prefer that.
PPO_ZIP     = os.path.join(MODELS_DIR, "best", "best_model.zip")
VECNORM_PKL = os.path.join(MODELS_DIR, "best", "vecnormalize_param.pkl")

SITES = {
    "Akosombo (held-out)": (6.30, 0.05),
    "Accra (held-out)":    (5.56, -0.20),
    "Bolgatanga (held-out)":(10.79, -0.85),
    "Tamale (train)":      (9.40, -0.84),
    "Kumasi (train)":      (6.69, -1.62),
    "Axim (train)":        (4.87, -2.24),
}
REFRESH_MS = 4000   # engine tick; the twin iframe persists (no reload/blink)

# ============================================================
# CONSTANTS  (must match training env)
# ============================================================
C = dict(residual_scale=0.25, pv_derating=0.75, soc_min=0.10, soc_max=0.95, eta=0.95,
         max_solar_kwp=120.0, max_bat_kwh=400.0, max_load_kw=20.0, obs_hourly=24,
         vn_clip=10.0, vn_eps=1e-8)
TIERS = {50: dict(kwp=50, kwh=160, load=5.6),
         75: dict(kwp=75, kwh=237, load=8.4),
         120:dict(kwp=120,kwh=378, load=13.4)}
LOAD_SHAPE = [.45,.40,.38,.38,.42,.55,.85,1.05,.95,.82,.78,.80,
              .82,.80,.78,.82,.95,1.35,1.95,2.20,2.05,1.55,.95,.60]
FORECAST = 48

# ============================================================
# MODEL + ARTIFACTS  (cached once)
# ============================================================
def _build_lstm():
    import torch.nn as nn
    class MiniGridLSTM(nn.Module):
        def __init__(s, inp, hid, nl, fc, out):
            super().__init__(); s.hidden_size, s.num_layers = hid, nl
            import torch
            s.lstm = nn.LSTM(inp, hid, nl, batch_first=True, dropout=0.2)
            s.fc = nn.Linear(hid, fc*out); s.fc_out, s.fc_h = fc, out
        def forward(s, x):
            import torch
            h0 = torch.zeros(s.num_layers, x.size(0), s.hidden_size)
            c0 = torch.zeros(s.num_layers, x.size(0), s.hidden_size)
            o, _ = s.lstm(x, (h0, c0))
            return s.fc(o[:, -1, :]).view(-1, s.fc_out, s.fc_h)
    return MiniGridLSTM(7, 128, 2, FORECAST, 2)

@st.cache_resource(show_spinner="Loading models…")
def load_models():
    info = {"lstm": None, "scaler_X": None, "scaler_y": None, "ppo": None,
            "vn_mean": None, "vn_var": None, "vn_clip": C["vn_clip"], "vn_eps": C["vn_eps"],
            "have_lstm": False, "have_ppo": False, "msg": []}
    try:
        import torch, pickle
        if os.path.exists(LSTM_PTH) and os.path.exists(SCALER_X) and os.path.exists(SCALER_Y):
            m = _build_lstm()
            m.load_state_dict(torch.load(LSTM_PTH, map_location="cpu", weights_only=True)); m.eval()
            info["lstm"] = m
            info["scaler_X"] = pickle.load(open(SCALER_X, "rb"))
            info["scaler_y"] = pickle.load(open(SCALER_Y, "rb"))
            info["have_lstm"] = True
        else:
            info["msg"].append("LSTM/scalers not found — using weather-derived forecast.")
    except Exception as e:
        info["msg"].append(f"LSTM load failed: {e}")
    try:
        import pickle
        if os.path.exists(PPO_ZIP) and os.path.exists(VECNORM_PKL):
            from stable_baselines3 import PPO
            info["ppo"] = PPO.load(PPO_ZIP, device="cpu")
            vn = pickle.load(open(VECNORM_PKL, "rb"))
            info["vn_mean"] = np.asarray(vn.obs_rms.mean, dtype=np.float64)
            info["vn_var"] = np.asarray(vn.obs_rms.var, dtype=np.float64)
            info["vn_clip"] = float(getattr(vn, "clip_obs", C["vn_clip"]))
            info["vn_eps"] = float(getattr(vn, "epsilon", C["vn_eps"]))
            info["have_ppo"] = True
        else:
            info["msg"].append("PPO/VecNormalize not found — baseline load-follower only.")
    except Exception as e:
        info["msg"].append(f"PPO load failed: {e}")
    return info

@st.cache_data(ttl=900, show_spinner=False)
def get_weather(lat, lon):
    """72-hour window: index 0 = yesterday 00:00 local, 24 = today 00:00, 48 = tomorrow 00:00.
    Live 'absolute hour' = 24 + local_hour; a 24h run spans abs 24+h0 .. 48+h0 <= 72."""
    import json, urllib.request, datetime as dt
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}"
           "&hourly=shortwave_radiation,temperature_2m,precipitation"
           "&past_days=2&forecast_days=2&timezone=auto")
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            j = json.loads(r.read().decode())
        h = j["hourly"]; off = j.get("utc_offset_seconds", 0)
        now_loc = dt.datetime.utcfromtimestamp(time.time() + off)
        today = now_loc.strftime("%Y-%m-%dT")
        idx = next((i for i, t in enumerate(h["time"]) if t.startswith(today)), 24)
        start = idx - 24                       # yesterday 00:00
        n = len(h["shortwave_radiation"])
        def grab(key, fallback):
            src = h[key]
            out = []
            for i in range(start, start + 72):
                v = src[i] if 0 <= i < n else None
                out.append(float(v) if v is not None else fallback)
            return out
        return dict(ghi=grab("shortwave_radiation", 0.0),
                    temp=grab("temperature_2m", 26.0),
                    ppt=grab("precipitation", 0.0),
                    month=now_loc.month, dow=now_loc.weekday(),
                    date=now_loc.strftime("%Y-%m-%d"),
                    local_hour=now_loc.hour + now_loc.minute/60.0, ok=True)
    except Exception:
        t = list(range(72))
        ghi = [max(0.0, math.sin(math.pi*((x % 24)-6)/12))**1.15 * 1050 for x in t]
        return dict(ghi=ghi, temp=[26+6*math.sin(math.pi*((x % 24)-9)/12) for x in t],
                    ppt=[0.0]*72, month=6, dow=2, date="modeled",
                    local_hour=12.0, ok=False)

# ============================================================
# ENV  (ported 1:1 from the validated twin)
# ============================================================
def day_val(arr, h):
    """Linear interp over the 72h absolute-hour axis (clamped at the ends)."""
    n = len(arr)
    i = max(0, min(n-1, int(h))); j = min(n-1, i+1); f = h - int(h)
    return arr[i]*(1-f) + arr[j]*f

def load_kw(h, tier):
    hh = h % 24
    i = int(hh) % 24; j = (i+1) % 24; f = hh - int(hh)
    return tier["load"]*(LOAD_SHAPE[i]*(1-f)+LOAD_SHAPE[j]*f)

def window24(abs_k, DAY, tier):
    """LSTM lookback: the 24 REAL hours ending just before absolute hour abs_k."""
    rows = []
    for n in range(24):
        a = abs_k - 24 + n
        idx = max(0, min(71, int(a)))
        rows.append([DAY["ghi"][idx], DAY["ppt"][idx], DAY["temp"][idx],
                     tier["load"]*LOAD_SHAPE[int(a) % 24], int(a) % 24, DAY["month"], DAY["dow"]])
    return np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0)

def forecast(k, DAY, tier, M):
    if M["have_lstm"]:
        import torch
        Xs = M["scaler_X"].transform(window24(k, DAY, tier))
        x = torch.tensor(Xs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = M["lstm"](x).squeeze(0).numpy()
        out = np.clip(out, 0.0, 1.0)
        phys = M["scaler_y"].inverse_transform(out)
        return out[:, 0], out[:, 1], phys
    # fallback: weather-derived normalised forecast
    sol, ld = [], []
    for kk in range(FORECAST):
        hh = (k + kk) % 24
        sol.append(min(1.0, day_val(DAY["ghi"], hh)/1000.0))
        ld.append(min(1.0, load_kw(hh, tier)/max(tier["load"]*2.2, 1)))
    sol, ld = np.array(sol), np.array(ld)
    phys = np.stack([sol*1000.0, ld*max(tier["load"]*2.2, 1)], axis=1)
    return sol, ld, phys

def build_obs(sol_n, ld_n, hour, month, base_act, soc, soh, tier):
    h2 = C["obs_hourly"]
    d2s = float(np.mean(sol_n[h2:])); d2l = float(np.mean(ld_n[h2:]))
    return np.asarray([soc, soh,
        math.sin(2*math.pi*hour/24), math.cos(2*math.pi*hour/24),
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

def step(dt_h, abs_h, soc, soh, DAY, tier, M, maxP):
    if soc is None: soc = 0.5
    if soh is None: soh = 1.0
    if abs_h is None: abs_h = 36.0
    hod = abs_h % 24                                  # hour-of-day for obs/load
    ghi = day_val(DAY["ghi"], abs_h)
    solar = (ghi/1000.0)*tier["kwp"]*C["pv_derating"]
    load = load_kw(hod, tier)
    sol_n, ld_n, phys = forecast(int(abs_h)+1, DAY, tier, M)
    base_act = float(np.clip((solar-load)/maxP, -1.0, 1.0))
    a = ppo_action(build_obs(sol_n, ld_n, hod, DAY["month"], base_act, soc, soh, tier), M)
    act = float(np.clip(base_act + C["residual_scale"]*a, -1.0, 1.0))
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
                ghi=ghi, base_act=base_act, residual=C["residual_scale"]*a, act=act,
                fc_solar=[round(float(v),1) for v in fc_solar],
                fc_load=[round(float(v),1) for v in phys[:3,1]])

def make_plan(abs_h0, soc0, soh0, DAY, tier, M, maxP):
    """The operator's 24-hour dispatch plan: simulate the next 24 hours from the
    current state using forecast weather, recording the PPO decision each hour."""
    plan = []
    soc, soh = float(soc0), float(soh0)
    for i in range(24):
        a = abs_h0 + i
        st_ = step(1.0, a, soc, soh, DAY, tier, M, maxP)
        soc, soh = st_["soc"], st_["soh"]
        plan.append(dict(clock=f"{int(a)%24:02d}:00", abs_h=a,
                         solar=round(st_["solar"], 1), load=round(st_["load"], 1),
                         power=round(st_["power"], 1), soc=round(soc*100, 1),
                         unmet=round(st_["unmet"], 1),
                         action=("charge" if st_["power"] > 0.5 else
                                 "discharge" if st_["power"] < -0.5 else "hold")))
    return plan

# ============================================================
# PERSISTENT TWIN COMPONENT  (mounted once; state streams in — no reloads)
# Requires the twin_component/ folder next to app.py.
# ============================================================
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_TWIN_DIR = os.path.join(_APP_DIR, "twin_component")
# Self-heal a flattened download: index.html next to app.py -> twin_component/index.html
if not os.path.isfile(os.path.join(_TWIN_DIR, "index.html")):
    for cand in ("index.html", "twin_component.html", "index (1).html"):
        p = os.path.join(_APP_DIR, cand)
        if os.path.isfile(p):
            import shutil
            os.makedirs(_TWIN_DIR, exist_ok=True)
            shutil.copy(p, os.path.join(_TWIN_DIR, "index.html"))
            break
if os.path.isfile(os.path.join(_TWIN_DIR, "index.html")):
    _twin = components.declare_component("srep_twin", path=_TWIN_DIR)
else:
    _twin = None

# ============================================================
# STREAMLIT UI  —  OPERATOR MODE (single mode: 24 h real-time run)
# ============================================================
st.set_page_config(page_title="SREP mini-grid · operator", layout="wide", page_icon="⚡")

# ------------------------------------------------------------
# THEME  — real mini-grid photo backdrop + dark glass UI.
# Drop your own site photo as background.jpg / assets/background.jpg
# next to app.py to replace the stock image (recommended: your SREP site).
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _bg_url():
    import base64
    for p in ("background.jpg", "background.png", "assets/background.jpg", "assets/background.png"):
        if os.path.exists(p):
            ext = "png" if p.endswith("png") else "jpeg"
            return f"data:image/{ext};base64," + base64.b64encode(open(p, "rb").read()).decode()
    # Unsplash (free license, hotlink-served): utility-scale solar field
    return ("https://images.unsplash.com/photo-1509391366360-2e959784a276"
            "?q=80&w=1920&auto=format&fit=crop")

_THEME_CSS = """
<style>
[data-testid="stAppViewContainer"]{
  background:
    linear-gradient(160deg, rgba(4,9,14,.93) 0%, rgba(6,12,18,.88) 45%, rgba(3,8,12,.95) 100%),
    url('__BG__') center/cover fixed no-repeat;
}
[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.2rem; max-width:1250px;}
h1,h2,h3{color:#f2f8fc !important; text-shadow:0 2px 14px rgba(0,0,0,.75);}
[data-testid="stMarkdownContainer"] p{color:#e6f0f7;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p{
  color:#a9bcc9 !important; text-shadow:0 1px 8px rgba(0,0,0,.7);}
[data-testid="stPlotlyChart"]{
  background:rgba(10,17,25,.62); border:1px solid rgba(148,187,214,.14);
  border-radius:14px; padding:8px;}
[data-testid="stDataFrame"]{
  border:1px solid rgba(148,187,214,.18); border-radius:12px; overflow:hidden;}
[data-testid="stWidgetLabel"] p{color:#dbe9f2 !important; font-weight:500;}
[data-testid="stForm"]{
  background:rgba(10,17,25,.66); backdrop-filter:blur(12px);
  border:1px solid rgba(148,187,214,.16); border-radius:18px;
  padding:26px 30px 18px; box-shadow:0 18px 50px rgba(0,0,0,.45);
}
[data-testid="stForm"] h3{color:#7fd6b4; font-size:1.0rem; letter-spacing:.05em;
  text-transform:uppercase; border-bottom:1px solid rgba(148,187,214,.14);
  padding-bottom:6px; margin-bottom:10px;}
[data-testid="stFormSubmitButton"] button{
  background:linear-gradient(90deg,#0ea471,#10b981); color:#04130d; font-weight:700;
  border:0; border-radius:12px; padding:.7rem 1rem; font-size:1.02rem;
  box-shadow:0 8px 26px rgba(16,185,129,.35);}
[data-testid="stFormSubmitButton"] button:hover{filter:brightness(1.08);}
[data-testid="stMetric"]{
  background:rgba(10,17,25,.62); backdrop-filter:blur(8px);
  border:1px solid rgba(148,187,214,.14); border-radius:14px; padding:12px 16px;}
[data-testid="stMetricValue"]{color:#f2f8fc;}
[data-testid="stMetricLabel"] p{color:#93a8b6 !important;}
[data-testid="stExpander"]{
  background:rgba(10,17,25,.62); border:1px solid rgba(148,187,214,.14); border-radius:14px;}
div[data-testid="stButton"] button{
  background:rgba(15,24,33,.85); color:#dbe9f2; border:1px solid rgba(148,187,214,.25);
  border-radius:10px;}
.hero-chips{display:flex; gap:10px; flex-wrap:wrap; margin:2px 0 18px;}
.hero-chips span{background:rgba(16,185,129,.12); border:1px solid rgba(16,185,129,.35);
  color:#8ee8c4; border-radius:999px; padding:5px 14px; font-size:.82rem; font-weight:500;}
.hero-sub{color:#b7c9d6; font-size:1.04rem; max-width:760px; margin:-6px 0 14px;}
</style>
"""
st.markdown(_THEME_CSS.replace("__BG__", _bg_url()), unsafe_allow_html=True)

M = load_models()
ss = st.session_state
REFRESH_MS = 4000   # engine tick; the twin iframe persists (no reload/blink)

def hhmm(h): return f"{int(h)%24:02d}:{int((h%1)*60):02d}"

badge = ("🟢 Real LSTM + PPO" if (M["have_lstm"] and M["have_ppo"])
         else "🟡 " + ("PPO only" if M["have_ppo"] else "LSTM only" if M["have_lstm"] else "Baseline only"))

# ------------------------------------------------------------
# SETUP  (operator describes the plant, then Start)
# ------------------------------------------------------------
if not ss.get("configured"):
    st.markdown("# ⚡ SREP Mini-Grid — AI Battery EMS")
    st.markdown('<div class="hero-sub">Real-time energy management for solar-battery mini-grids '
                'on the Volta Lake. Describe your plant below — the LSTM forecaster and PPO dispatch '
                'agent will plan the next 24 hours and then operate live, one wall-clock hour at a time.</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="hero-chips"><span>{badge.split(" ",1)[1] if " " in badge else badge}</span>'
                '<span>LSTM 48 h forecaster</span><span>PPO residual dispatch</span>'
                '<span>Live Open-Meteo weather</span><span>1 h = 1 h real-time</span></div>',
                unsafe_allow_html=True)
    for msg in M["msg"]:
        st.caption("• " + msg)

    with st.form("setup"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Location")
            preset = st.selectbox("Site", list(SITES.keys()) + ["Custom coordinates…"])
            lat_in = st.number_input("Latitude", -90.0, 90.0, 6.30, 0.01, format="%.2f")
            lon_in = st.number_input("Longitude", -180.0, 180.0, 0.05, 0.01, format="%.2f")
        with c2:
            st.subheader("System size")
            kwp_in = st.number_input("Solar array (kWp)", 5.0, 120.0, 75.0, 1.0)
            kwh_in = st.number_input("Battery capacity (kWh)", 20.0, 400.0, 237.0, 1.0)
        with c3:
            st.subheader("Load & battery state")
            load_in = st.number_input("Mean community load (kW)", 1.0, 20.0, 8.4, 0.1,
                                      help="Daily-average demand; the standard SREP diurnal "
                                           "shape (evening peak ≈2.2× mean) is scaled to this.")
            soc_in = st.slider("Current battery state of charge (%)", 10, 95, 55,
                               help="Read this off your BMS right now.")
        started = st.form_submit_button("▶ Start 24-hour real-time run", use_container_width=True)

    if started:
        if preset != "Custom coordinates…":
            lat_in, lon_in = SITES[preset]
            ss.site_name = preset.split(" (")[0]
        else:
            ss.site_name = f"{lat_in:.2f}, {lon_in:.2f}"
        ss.lat, ss.lon = float(lat_in), float(lon_in)
        ss.tier = dict(kwp=float(kwp_in), kwh=float(kwh_in), load=float(load_in))
        ss.maxP = min(0.5*ss.tier["kwh"], 0.8*ss.tier["kwp"])
        with st.spinner("Fetching live weather and computing the 24-hour dispatch plan…"):
            DAY = get_weather(ss.lat, ss.lon)
            ss.day_date = DAY["date"]
            ss.abs_hour = 24.0 + DAY["local_hour"]          # 'now' on the 72h axis
            ss.start_abs = ss.abs_hour
            ss.soc = float(soc_in)/100.0
            ss.soh = 1.0
            ss.plan = make_plan(ss.abs_hour, ss.soc, ss.soh, DAY, ss.tier, M, ss.maxP)
        ss.hist = {"solar": [None]*24, "load": [None]*24, "power": [None]*24, "soc": [None]*24}
        ss.kpi = dict(served=0.0, loadE=0.0, lol=0.0, efc=0.0)          # run totals
        ss.today = dict(date=DAY["date"], served=0.0, loadE=0.0, lol=0.0)  # daily window
        ss.days_hist = []
        ss.log = [f"{hhmm(DAY['local_hour'])} · run started · SoC {soc_in}% (operator) · "
                  f"{ss.tier['kwp']:.0f} kWp / {ss.tier['kwh']:.0f} kWh / {ss.tier['load']:.1f} kW mean"]
        ss.last_real = time.time()
        ss.last_hour = int(ss.abs_hour)
        ss.last_dec = None
        ss.finished = False
        ss.configured = True
        st.rerun()
    st.stop()

# ------------------------------------------------------------
# LIVE RUN  (1 h = 1 h, always)
# ------------------------------------------------------------
lat, lon = ss.lat, ss.lon
tier = ss.tier; maxP = ss.maxP
DAY = get_weather(lat, lon)

# self-heal any stale/partial session
if ss.get("soc") is None: ss.soc = 0.5
if ss.get("soh") is None: ss.soh = 1.0
if ss.get("abs_hour") is None: ss.abs_hour = 24.0 + DAY["local_hour"]

# Re-anchor when the 72h weather window slides at midnight: after a fresh fetch,
# "today 00:00" moves to index 24 of the NEW axis, so shift our clock back to match.
if "anchor_date" not in ss: ss.anchor_date = DAY["date"]
if DAY["date"] not in (ss.anchor_date, "modeled"):
    import datetime as _dt
    try:
        dshift = (_dt.date.fromisoformat(DAY["date"]) - _dt.date.fromisoformat(ss.anchor_date)).days
    except Exception:
        dshift = 1
    ss.abs_hour -= 24*dshift; ss.start_abs -= 24*dshift; ss.last_hour -= 24*dshift
    ss.anchor_date = DAY["date"]

elapsed = ss.abs_hour - ss.start_abs

try:
    from streamlit_autorefresh import st_autorefresh
    if not ss.finished:
        st_autorefresh(interval=REFRESH_MS, key="tick")
except Exception:
    st.caption("Install streamlit-autorefresh for live ticking.")

if not ss.finished:
    now = time.time()
    dt_h = max(now - ss.last_real, 0.0)/3600.0; ss.last_real = now
    dt_h = min(dt_h, 1.0)                                   # guard long sleeps
    ss.abs_hour += dt_h
    elapsed = ss.abs_hour - ss.start_abs
    d = step(dt_h, ss.abs_hour, ss.soc, ss.soh, DAY, tier, M, maxP)
    ss.soc, ss.soh = d["soc"], d["soh"]
    ss.last_dec = d
    hod = int(ss.abs_hour) % 24
    ss.hist["solar"][hod] = round(d["solar"], 2); ss.hist["load"][hod] = round(d["load"], 2)
    ss.hist["power"][hod] = round(d["power"], 2); ss.hist["soc"][hod] = round(ss.soc*100, 1)
    # run + daily accumulators
    for K in (ss.kpi, ss.today):
        K["served"] = K.get("served", 0.0) + (d["load"]-d["unmet"])*dt_h
        K["loadE"] = K.get("loadE", 0.0) + d["load"]*dt_h
        if d["unmet"] > 0.5: K["lol"] = K.get("lol", 0.0) + dt_h*60
    ss.kpi["efc"] += abs(d["power"])*dt_h
    # midnight rollover -> archive the day
    if int(ss.abs_hour) % 24 == 0 and int(ss.abs_hour) != ss.last_hour and elapsed > 0.1:
        sp = ss.today["served"]/ss.today["loadE"]*100 if ss.today["loadE"] > 0 else 100
        ss.days_hist.append(dict(date=ss.today["date"], served=round(sp, 1),
                                 lol=round(ss.today["lol"], 0)))
        ss.log.append(f"00:00 · day closed · served {sp:.1f}% · LoL {ss.today['lol']:.0f} min")
        ss.today = dict(date=DAY["date"], served=0.0, loadE=0.0, lol=0.0)
    if int(ss.abs_hour) != ss.last_hour:
        ss.last_hour = int(ss.abs_hour)
        spd = ss.today["served"]/ss.today["loadE"]*100 if ss.today["loadE"] > 0 else 100
        ss.log.append(f"{hhmm(ss.abs_hour)} · SoC {ss.soc*100:.0f}% · served today {spd:.0f}%"
                      + (f" · ⚠ shed {d['unmet']:.0f} kW" if d["unmet"] > 0.5 else ""))
        del ss.log[:-40]
    if elapsed >= 24.0:
        ss.finished = True
        sp = ss.kpi["served"]/ss.kpi["loadE"]*100 if ss.kpi["loadE"] > 0 else 100
        ss.log.append(f"{hhmm(ss.abs_hour)} · ✔ 24 h run complete · served {sp:.1f}% · "
                      f"LoL {ss.kpi['lol']:.0f} min · {ss.kpi['efc']/(2*tier['kwh']):.2f} EFC")

d = ss.last_dec or step(0.0, ss.abs_hour, ss.soc, ss.soh, DAY, tier, M, maxP)
served_run = ss.kpi["served"]/ss.kpi["loadE"]*100 if ss.kpi["loadE"] > 0 else 100.0
served_day = ss.today["served"]/ss.today["loadE"]*100 if ss.today["loadE"] > 0 else 100.0
status_color = "#64748b" if ss.finished else "#10b981"

# ---- header ----
top1, top2 = st.columns([4, 1])
with top1:
    st.title(f"{ss.site_name} · {tier['kwp']:.0f} kWp / {tier['kwh']:.0f} kWh")
    st.caption(f"{badge} · weather {DAY['date']} {'(live)' if DAY['ok'] else '(modeled)'} · "
               f"clock {hhmm(ss.abs_hour)} · elapsed {min(elapsed,24):.2f}/24 h · real-time 1 h = 1 h")
with top2:
    if st.button("■ Stop & reconfigure", use_container_width=True):
        ss.configured = False
        st.rerun()
st.progress(min(elapsed/24.0, 1.0))
if ss.finished:
    st.success(f"24-hour run complete — energy served {served_run:.1f}%, "
               f"loss-of-load {ss.kpi['lol']:.0f} min, {ss.kpi['efc']/(2*tier['kwh']):.2f} EFC. "
               "Stop & reconfigure to start a new run.")

# ---- twin (3D + operator console), driven by real state ----
ghi_today = DAY["ghi"][24:48]
if _twin is None:
    st.error("twin_component/ folder not found next to app.py — the live twin cannot render. "
             "Keep app.py and the twin_component folder together.")
else:
    _twin(state=dict(
        S=dict(hour=ss.abs_hour % 24, soc=ss.soc, soh=ss.soh, solar=d["solar"], load=d["load"],
               power=d["power"], ghi=d["ghi"], base=d["base_act"], resid=d["residual"],
               act=d["act"], unmet=d["unmet"], kwp=tier["kwp"], kwh=tier["kwh"],
               loaddes=tier["load"], maxP=maxP, served=served_day, lol=ss.today["lol"],
               efc=ss.kpi["efc"]/(2*tier["kwh"]), ready=bool(M["have_lstm"] and M["have_ppo"]),
               running=not ss.finished),
        HIST=ss.hist, FC=dict(solar=d["fc_solar"], load=d["fc_load"]),
        LOG=ss.log[-40:], GHI24=ghi_today, month=DAY["month"], dow=DAY["dow"]),
        key="twin", default=None)

# ---- operator KPIs & the 24 h plan ----
import pandas as pd
k = st.columns(5)
k[0].metric("Energy served — today", f"{served_day:.1f}%")
k[1].metric("Energy served — run", f"{served_run:.1f}%")
k[2].metric("Loss-of-load — today", f"{ss.today['lol']:.0f} min")
k[3].metric("Battery SoC", f"{ss.soc*100:.0f}%")
k[4].metric("Throughput — run", f"{ss.kpi['efc']/(2*tier['kwh']):.2f} EFC")

pc1, pc2 = st.columns([2, 1])
with pc1:
    st.subheader("24-hour dispatch plan")
    st.caption("PPO schedule from run start — planned battery power per hour "
               "(+ charge / − discharge) with predicted SoC. Live decisions may deviate as real weather arrives.")
    dfp = pd.DataFrame(ss.plan)
    import plotly.graph_objects as go
    cols = ["#10b981" if p >= 0 else "#3b82f6" for p in dfp["power"]]
    figp = go.Figure()
    figp.add_trace(go.Bar(x=dfp["clock"], y=dfp["power"], marker_color=cols,
                          name="Battery kW", hovertemplate="%{x} · %{y:+.1f} kW<extra></extra>"))
    figp.add_trace(go.Scatter(x=dfp["clock"], y=dfp["soc"], yaxis="y2", name="SoC %",
                              line=dict(color="#a78bfa", width=2, dash="dot")))
    figp.update_layout(height=300, margin=dict(l=8, r=8, t=8, b=8),
                       paper_bgcolor="rgba(10,17,25,.62)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#c9dae6", family="IBM Plex Mono, monospace", size=11),
                       xaxis=dict(gridcolor="rgba(148,187,214,.10)"),
                       yaxis=dict(title="kW", gridcolor="rgba(148,187,214,.10)", zerolinecolor="rgba(148,187,214,.25)"),
                       yaxis2=dict(title="SoC %", overlaying="y", side="right", range=[0, 100], showgrid=False),
                       legend=dict(orientation="h", y=1.06, x=0, font=dict(size=11)),
                       bargap=0.25)
    st.plotly_chart(figp, use_container_width=True, config={"displayModeBar": False})
with pc2:
    st.subheader("Plan detail")
    st.dataframe(dfp[["clock", "action", "power", "soc", "solar", "load"]],
                 height=260, use_container_width=True, hide_index=True)
    if dfp["unmet"].max() > 0.5:
        bad = dfp[dfp["unmet"] > 0.5]["clock"].tolist()
        st.warning("Plan predicts load shedding at: " + ", ".join(bad))

if ss.days_hist:
    st.subheader("Daily performance")
    st.dataframe(pd.DataFrame(ss.days_hist), hide_index=True, use_container_width=True)

st.caption("Operator mode · real LSTM + PPO in Python · weather from Open-Meteo (72 h window: "
           "yesterday for the LSTM lookback, today + tomorrow for the run) · keep this tab open.")
