from flask import Flask, render_template, request
import pandas as pd
from models import load_models, predict_fire_and_spread
##request grabs user inputs from forms.
app = Flask(__name__)
import os

# Load models and preprocessor
best_model, scaler, spread_models, X_columns = load_models()

@app.route("/", methods=["GET", "POST"])
def index():
    ## when form is submitted
    if request.method == "POST":
        # Get input from form
        input_data = pd.DataFrame([{
            'Rainfall (mm)': float(request.form['rainfall']),
            'T-MAX (°C)': float(request.form['tmax']),
            'T-MIN (°C)': float(request.form['tmin']),
            'Cloud Cover': float(request.form['cloud']),
            'Rh Max (%)': float(request.form['rhmax']),
            'Rh Min (%)': float(request.form['rhmin']),
            'Wind speed (kmph)': float(request.form['wind_speed']),
            'Wind Direction (deg)': float(request.form['wind_dir']),
            'Latitude': float(request.form['latitude']),
            'Longitude': float(request.form['longitude'])
        }])

        prediction, spread, map_generated = predict_fire_and_spread(
            best_model, scaler, spread_models, X_columns, input_data
        )

        return render_template("result.html", prediction=prediction, spread=spread, map_url="/static/fire_spread_map.html" if map_generated else None)

    return render_template("index.html")



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

