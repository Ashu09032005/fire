import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
import folium
from folium.plugins import MeasureControl
import os

# =============================
# 1. Load & preprocess data
# =============================
def load_and_prepare_data(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')

    df['Fire/No Fire'] = df['Fire/No Fire'].apply(lambda x: 1 if x == 'Fire' else 0)
    return df

# =============================
# 2. Train/test split & scale
# =============================
def split_and_scale(df, spread_cols):
    X = df.drop(['Fire/No Fire', 'Date'] + spread_cols, axis=1)
    y = df['Fire/No Fire']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

# =============================
# 3. Train classifiers
# =============================
def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
        "Random Forest": RandomForestClassifier(class_weight="balanced"),
        "Adaboost": AdaBoostClassifier(),
        "XGB": XGBClassifier()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        print(f"\nModel: {name}")
        print("Training performance:")
        print_metrics(y_train, y_pred_train)
        print("Testing performance:")
        print_metrics(y_test, y_pred_test)
    return models

# =============================
# 4. Print metrics
# =============================
def print_metrics(y_true, y_pred):
    print(f"  Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"  F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"  ROC AUC: {roc_auc_score(y_true, y_pred):.4f}")
    print("-" * 40)

# =============================
# 5. Train spread models
# =============================
def train_spread_models(df_fire_only, spread_cols, X_columns, scaler):
    fire_features = df_fire_only[X_columns]
    spread_models = {}
    for col in spread_cols:
        reg = RandomForestRegressor()
        reg.fit(scaler.transform(fire_features), df_fire_only[col])
        spread_models[col] = reg
    return spread_models

# =============================
# 6. Predict and visualize
# =============================
def predict_fire_and_spread(best_model, scaler, spread_models, X_columns, input_data):
    input_model = input_data[X_columns]
    input_scaled = scaler.transform(input_model)
    prediction = best_model.predict(input_model)[0]

    lat = input_data['Latitude'].values[0]
    lon = input_data['Longitude'].values[0]

    spread_cols = ['Spread (0.5 hr)', 'Spread (1 hr)', 'Spread (2 hr)']
    spread_results = {}
    map_generated = False

    if prediction == 1:
        for col in spread_cols:
            dist = spread_models[col].predict(input_scaled)[0]
            spread_results[col] = dist

        fire_map = folium.Map(location=[lat, lon], zoom_start=12, control_scale=True)
        folium.Marker([lat, lon], popup="🔥 Fire Origin", icon=folium.Icon(color='red')).add_to(fire_map)

        for col, dist in spread_results.items():
            folium.Circle(
                radius=dist * 1000,
                location=[lat, lon],
                popup=f"{col}: {dist:.2f} km",
                color='crimson',
                fill=True,
                fill_color='orange',
                fill_opacity=0.3
            ).add_to(fire_map)

        fire_map.add_child(MeasureControl())

        # Save the map in 'static' folder for Flask to serve
        os.makedirs("static", exist_ok=True)
        fire_map.save("static/fire_spread_map.html")
        map_generated = True

    return prediction, spread_results, map_generated

# =============================
# 7. Loader for Flask
# =============================
def load_models():
    file_path = "fire.xlsx"
    spread_cols = ['Spread (0.5 hr)', 'Spread (1 hr)', 'Spread (2 hr)']

    df = load_and_prepare_data(file_path)
    df_fire_only = df[df['Fire/No Fire'] == 1]

    X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = split_and_scale(df, spread_cols)
    models = train_models(X_train, X_test, y_train, y_test)

    best_model = models["Random Forest"]
    spread_models = train_spread_models(df_fire_only, spread_cols, X.columns, scaler)

    return best_model, scaler, spread_models, X.columns
