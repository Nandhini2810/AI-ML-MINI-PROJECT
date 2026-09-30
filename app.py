from flask import Flask, request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# ============================================================
# 1. LOAD DATABASE / DATASET
# ============================================================

DATASET = "crop_yield_dataset_1000_rows.csv"

df = pd.read_csv(DATASET)

df = df.drop_duplicates()
df = df.dropna()

print("======================================")
print("CROP YIELD PREDICTION SYSTEM")
print("======================================")

print("Dataset loaded successfully!")
print("Rows:", len(df))

# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

X = df.drop("crop_yield", axis=1)
y = df["crop_yield"]

numeric_features = [
    "rainfall",
    "temperature",
    "soil_ph",
    "soil_moisture",
    "fertilizer",
    "pesticide"
]

categorical_features = [
    "crop_type"
]

# ============================================================
# 3. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# ============================================================
# 4. TRAIN MODEL AUTOMATICALLY
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

print("Training model...")

model.fit(X_train, y_train)

print("Model trained successfully!")

# ============================================================
# 5. GET CROP NAMES FROM DATABASE
# ============================================================

crop_names = sorted(
    df["crop_type"].astype(str).unique()
)

print("Available crops:", crop_names)


# ============================================================
# 6. HOME PAGE
# ============================================================

@app.route("/")
def home():

    crop_options = ""

    for crop in crop_names:
        crop_options += f'<option value="{crop}">{crop}</option>'

    return f"""
<!DOCTYPE html>

<html>

<head>

<title>Crop Yield Prediction</title>

<style>

body {{
    margin: 0;
    padding: 0;
    font-family: Arial, sans-serif;
    background: #eef7ee;
}}

.container {{
    width: 500px;
    margin: 40px auto;
    background: white;
    padding: 35px;
    border-radius: 15px;
    box-shadow: 0 0 15px #999;
}}

h1 {{
    text-align: center;
    color: green;
}}

.subtitle {{
    text-align: center;
    color: #666;
    margin-bottom: 25px;
}}

label {{
    display: block;
    margin-top: 12px;
    font-weight: bold;
}}

input, select {{
    width: 100%;
    padding: 11px;
    margin-top: 5px;
    border: 1px solid #ccc;
    border-radius: 6px;
    box-sizing: border-box;
}}

button {{
    width: 100%;
    padding: 13px;
    margin-top: 25px;
    background: green;
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 17px;
    cursor: pointer;
}}

button:hover {{
    background: darkgreen;
}}

</style>

</head>

<body>

<div class="container">

<h1>🌾 Crop Yield Prediction</h1>

<div class="subtitle">
Enter the agricultural details
</div>

<form action="/predict" method="POST">

<label>Crop Type</label>

<select name="crop_type" required>

<option value="">Select Crop</option>

{crop_options}

</select>


<label>Rainfall (mm)</label>

<input
type="number"
step="any"
name="rainfall"
placeholder="Example: 850"
required
>


<label>Temperature (°C)</label>

<input
type="number"
step="any"
name="temperature"
placeholder="Example: 28"
required
>


<label>Soil pH</label>

<input
type="number"
step="any"
name="soil_ph"
placeholder="Example: 6.5"
required
>


<label>Soil Moisture</label>

<input
type="number"
step="any"
name="soil_moisture"
placeholder="Example: 40"
required
>


<label>Fertilizer</label>

<input
type="number"
step="any"
name="fertilizer"
placeholder="Example: 100"
required
>


<label>Pesticide</label>

<input
type="number"
step="any"
name="pesticide"
placeholder="Example: 20"
required
>


<button type="submit">
🌱 Predict Crop Yield
</button>

</form>

</div>

</body>

</html>
"""


# ============================================================
# 7. PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    crop_type = request.form["crop_type"]

    rainfall = float(request.form["rainfall"])

    temperature = float(request.form["temperature"])

    soil_ph = float(request.form["soil_ph"])

    soil_moisture = float(request.form["soil_moisture"])

    fertilizer = float(request.form["fertilizer"])

    pesticide = float(request.form["pesticide"])


    # Create input data

    input_data = pd.DataFrame({

        "rainfall": [rainfall],

        "temperature": [temperature],

        "soil_ph": [soil_ph],

        "soil_moisture": [soil_moisture],

        "fertilizer": [fertilizer],

        "pesticide": [pesticide],

        "crop_type": [crop_type]

    })


    # Prediction

    prediction = model.predict(input_data)[0]


    # Result page

    return f"""
<!DOCTYPE html>

<html>

<head>

<title>Crop Yield Result</title>

<style>

body {{
    margin: 0;
    padding: 0;
    font-family: Arial, sans-serif;
    background: #eef7ee;
}}

.result-box {{
    width: 500px;
    margin: 60px auto;
    background: white;
    padding: 40px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 0 15px #999;
}}

h1 {{
    color: green;
}}

.result {{
    font-size: 30px;
    color: darkgreen;
    font-weight: bold;
    margin: 25px;
}}

.details {{
    text-align: left;
    background: #f1f8f1;
    padding: 20px;
    border-radius: 10px;
}}

a {{
    display: inline-block;
    margin-top: 25px;
    padding: 12px 25px;
    background: green;
    color: white;
    text-decoration: none;
    border-radius: 8px;
}}

</style>

</head>

<body>

<div class="result-box">

<h1>🌾 Crop Yield Result</h1>

<div class="result">

{prediction:.2f} tons/ha

</div>

<div class="details">

<p><b>Crop:</b> {crop_type}</p>

<p><b>Rainfall:</b> {rainfall} mm</p>

<p><b>Temperature:</b> {temperature} °C</p>

<p><b>Soil pH:</b> {soil_ph}</p>

<p><b>Soil Moisture:</b> {soil_moisture}</p>

<p><b>Fertilizer:</b> {fertilizer}</p>

<p><b>Pesticide:</b> {pesticide}</p>

</div>

<a href="/">
Back to Prediction
</a>

</div>

</body>

</html>
"""


# ============================================================
# 8. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(debug=True)
