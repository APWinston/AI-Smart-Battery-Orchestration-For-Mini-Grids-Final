# SREP Mini Grid AI Battery EMS

Research prototype from KNUST. An LSTM forecaster and a PPO reinforcement
learning agent plan and operate a solar battery mini grid in real time,
one wall clock hour at a time, using live Open Meteo weather.

This platform is an academic research experiment and is not a deployed
utility system.

## Run locally
    pip install -r requirements.txt
    python server.py
Open http://localhost:8000 and sign in (default operator / srep2026).

## Layout
    server.py            FastAPI backend, real models, run controller
    twin.html            front end: landing, login, setup, 3D twin, console
    requirements.txt     Python dependencies (CPU torch)
    knust_logo.png       brand crest (served at /assets/knust_logo.png)
    best_lstm_param.pth  trained LSTM weights
    scaler_X_param.pkl / scaler_y_param.pkl
    best/best_model.zip  trained PPO policy
    best/vecnormalize_param.pkl

## Deploy (Render)
Build command: pip install -r requirements.txt
Start command: python server.py
Environment variables: OPERATOR_USER, OPERATOR_PASS
