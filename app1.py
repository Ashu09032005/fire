# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# %% Load and clean data
df = pd.read_csv("fire.csv")
df.drop(['version', 'instrument', 'satellite'], axis=1, inplace=True)

df['acq_datetime'] = pd.to_datetime(
    df['acq_date'].astype(str) + df['acq_time'].astype(str).str.zfill(4),
    format='%Y-%m-%d%H%M'
)
df.drop(['acq_date', 'acq_time'], axis=1, inplace=True)

# Map categorical columns
confidence_map = {'low': 0, 'nominal': 1, 'high': 2}
daynight_map = {'D': 0, 'N': 1}
df['confidence'] = df['confidence'].map(confidence_map)
df['daynight'] = df['daynight'].map(daynight_map)

# %% Feature columns
feature_cols = ['latitude', 'longitude', 'brightness', 'scan', 'track',
                'confidence', 'bright_t31', 'frp', 'daynight']

# %% Function: prepare dataset and train model for each time delta
def train_model_for_horizon(hours_ahead):
    print(f"\n🔁 Training model for +{hours_ahead} hour(s)...")

    # Create future-shifted df
    future_df = df[['acq_datetime', 'latitude', 'longitude']].copy()
    future_df['acq_datetime'] = future_df['acq_datetime'] - pd.Timedelta(hours=hours_ahead)
    future_df.rename(columns={
        'latitude': f'lat_future_{hours_ahead}h',
        'longitude': f'lon_future_{hours_ahead}h'
    }, inplace=True)

    # Align data
    valid_times = set(df['acq_datetime']).intersection(set(future_df['acq_datetime']))
    df_filt = df[df['acq_datetime'].isin(valid_times)].copy()
    future_filt = future_df[future_df['acq_datetime'].isin(valid_times)].copy()

    df_filt = df_filt.drop_duplicates(subset='acq_datetime', keep='first')
    future_filt = future_filt.drop_duplicates(subset='acq_datetime', keep='first')

    df_filt.sort_values(by='acq_datetime', inplace=True)
    future_filt.sort_values(by='acq_datetime', inplace=True)

    df_filt.reset_index(drop=True, inplace=True)
    future_filt.reset_index(drop=True, inplace=True)

    assert df_filt.shape[0] == future_filt.shape[0], f" Mismatch at +{hours_ahead}h"

    # Prepare X and y
    X = df_filt[feature_cols]
    y = future_filt[[f'lat_future_{hours_ahead}h', f'lon_future_{hours_ahead}h']]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE (+{hours_ahead}h): {rmse:.4f}")

    errors = np.abs(y_test - y_pred)
    within_thresh = (errors.iloc[:, 0] <= 0.5) & (errors.iloc[:, 1] <= 0.5)
    acc = np.mean(within_thresh) * 100
    print(f"Accuracy within ±0.5°: {acc:.2f}%")

    return model

# %% Train models for multiple time horizons
horizons = [0.5, 1, 2, 6]
models = {}

for h in horizons:
    models[h] = train_model_for_horizon(h)

# %% General Prediction Function
def predict_future_location_multi(model, horizon_hours, latitude, longitude,
                                   brightness, scan, track, confidence, bright_t31, frp, daynight):
    confidence_map = {'low': 0, 'nominal': 1, 'high': 2}
    daynight_map = {'day': 0, 'night': 1}

    confidence_val = confidence_map[confidence.lower()] if isinstance(confidence, str) else confidence
    daynight_val = daynight_map[daynight.lower()]

    input_df = pd.DataFrame([{
        'latitude': latitude,
        'longitude': longitude,
        'brightness': brightness,
        'scan': scan,
        'track': track,
        'confidence': confidence_val,
        'bright_t31': bright_t31,
        'frp': frp,
        'daynight': daynight_val
    }])

    pred = model.predict(input_df)
    lat_pred, lon_pred = pred[0]
    print(f"Predicted Location in {horizon_hours}h:\nLatitude: {lat_pred:.4f}, Longitude: {lon_pred:.4f}")
    return lat_pred, lon_pred

# %% Example: Predict for +0.5h, +1h, +2h, +6h
for h in horizons:
    predict_future_location_multi(
        model=models[h],
        horizon_hours=h,
        latitude=12.34,
        longitude=77.56,
        brightness=330.5,
        scan=1.2,
        track=1.1,
        confidence='high',
        bright_t31=310.2,
        frp=25.6,
        daynight='Day'
    )
