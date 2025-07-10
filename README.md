# 🔥 Kalahandi Forest Fire Prediction and Spread Estimation

This project uses meteorological and geographical data sourced from the [Orissa University of Agriculture and Technology (OUAT)](https://ouat.ac.in/) to predict the likelihood of a forest fire in the Kalahandi region and estimate the spread of fire over time if one occurs.

---

## 🛰️ About the Project

📢 **This project was developed as part of the ISRO-organized national-level**  
🇮🇳 **Bhartiya Antariksh Hackathon**, where participants were challenged with real-world space and environmental problems.  
The forest fire prediction and spread of forest fire was one of the official **problem statements** provided in the hackathon.

---

## 📌 Why This Project Is Important

Forest fires pose severe threats to life, biodiversity, and resources. Early fire detection and spread prediction are crucial for:
- **Prevention & Planning:** Helps authorities act before the fire spreads.
- **Resource Allocation:** Enables better deployment of firefighting teams.
- **Decision Support:** Supports local policies and environmental monitoring efforts.

---

## 🚀 Features

- 🔍 **Predicts Fire Likelihood**: Based on real-time input like temperature, rainfall, wind speed/direction, humidity, etc.
- 🌐 **Estimates Fire Spread**: If fire is predicted, the model estimates how far it may spread in:
  - 0.5 hour
  - 1 hour
  - 2 hours
- 📍 **Interactive Fire Spread Map**: Visualizes origin and predicted spread radius using **Folium maps**.

---

## 📁 Dataset

- **Source**: [OUAT - https://ouat.ac.in](https://ouat.ac.in/)
- **Region Focus**: Kalahandi district, Odisha, India
- **Size**: ~390MB `.xlsx` dataset
- **Fields Used**:
  - Rainfall (mm)
  - T-MAX (°C), T-MIN (°C)
  - Cloud Cover
  - Relative Humidity (Max/Min)
  - Wind Speed & Direction
  - Latitude & Longitude
  - Fire/No Fire (label)
  - Spread distances (km) after 0.5, 1, and 2 hours (for fire cases)

---

## 🧠 Model Architecture

### 🔎 Fire Detection (Classification)
- **Input**: Weather & location features
- **Output**: `Fire (1)` or `No Fire (0)`
- **Models Tried**:
  - Decision Tree (with class weighting)
  - Random Forest ✅ *(Selected for best generalization)*
  - AdaBoost
  - XGBoost

### 🔁 Spread Prediction (Regression)
- **Input**: Same features for fire-labeled data only
- **Output**: Distance of spread in km at 0.5h, 1h, 2h
- **Model Used**: Random Forest Regressor (for each time point)

---

## 🧪 Performance

| Metric        | Training | Testing |
|---------------|----------|---------|
| Accuracy      | ~99.4%   | ~97.6%  |
| Precision     | 0.85–1.0 | 0.0–1.0 |
| Recall        | 1.0      | 0.5     |
| ROC AUC       | ~0.99    | ~0.75   |

> 🔔 **Note**: The dataset is highly imbalanced (far more "No Fire" cases). We used `class_weight='balanced'` and evaluated multiple models to mitigate this.

---

## 🌍 Fire Spread Visualization

If a fire is predicted, a **Folium map** is generated showing:
- The **origin point** of fire (based on user input latitude/longitude).
- **Spread radius** circles for:
  - 0.5 hour 🔸
  - 1 hour 🔶
  - 2 hour 🔴

The map is saved as `static/fire_spread_map.html` and viewable via the web app.

---

## 🌐 Web Interface (Flask)

Users can:
1. Enter conditions (via form)
2. Submit for prediction
3. View results and interactive fire map

### 🏗️ Tech Stack

- **Backend**: Python, Flask
- **Data Science**: Pandas, Scikit-learn, XGBoost
- **Visualization**: Folium
- **Deployment-ready**: Lightweight and modular design

---

## ⚠️ Challenges

- **Imbalanced Data**: Used balanced classifiers and appropriate metrics like F1/Recall.
- **Generalization Risk**: Since data is from a specific region (Kalahandi), predictions on other regions may require retraining.
- **Large Dataset**: Handled using Git LFS for proper version control on GitHub.

---

## 📦 Setup

```bash
git clone https://github.com/your-username/fire-prediction.git
cd fire-prediction
pip install -r requirements.txt
python app.py
