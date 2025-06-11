import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

# File paths
DATA_FILE = "data.csv"
COPY_FILE = "data_copy.csv"
MODEL_FILE = "model.joblib"

# Column names
COLUMNS = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
           'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# Load reference dataset
data = pd.read_csv(DATA_FILE)

# Train and save model
def train_and_save_model(df):
    df_clean = df.dropna(subset=['MEDV'])  # only use rows with MEDV
    X = df_clean[COLUMNS]
    y = df_clean['MEDV']
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor())
    ])
    pipeline.fit(X, y)
    dump(pipeline, MODEL_FILE)
    return pipeline

# Load model or train if not exists
if os.path.exists(MODEL_FILE):
    model = load(MODEL_FILE)
else:
    model = train_and_save_model(data)

# Ensure data_copy.csv exists
if not os.path.exists(COPY_FILE):
    data.to_csv(COPY_FILE, index=False)

# UI
st.title("🏠 Real Estate Price Prediction")

st.markdown("""
Enter the property details below. Please follow the format guidelines:

- **CRIM**: Crime in % (e.g., `2` for 2%)
- **ZN**: Residential zone in % (e.g., `25`)
- **INDUS**: Industry area in % (e.g., `15`)
- **CHAS**: 0 = No river bound, 1 = River bound
- **NOX**: Nitric oxide concentration in % (e.g., `0.5`)
- **RM**: Number of rooms (e.g., `3`, `6.5`)
- **AGE**: Age of house (e.g., `50`)
- **DIS**: Distance to jobs in km (e.g., `4.2`)
- **RAD**: Highway access in km (e.g., `3`)
- **TAX**: Tax in % per $1000 (e.g., `18`)
- **PTRATIO**: Pupil-teacher ratio (e.g., `15.5`)
- **B**: Black population index (as-is)
- **LSTAT**: Lower status population % (e.g., `5.3`)
""")

user_input = {}
empty_count = 0

# Input section
for col in COLUMNS:
    val = st.text_input(f"{col}:", key=col)
    if val.strip() == '':
        empty_count += 1
    user_input[col] = val.strip()

if st.button("Predict"):
    if empty_count > 2:
        st.error("❌ Please fill at least 11 out of 13 fields.")
    else:
        try:
            processed_input = {}
            for col in COLUMNS:
                val = user_input[col]
                if val == '':
                    processed_input[col] = np.nan
                else:
                    num = float(val)
                    if col in ['CRIM', 'ZN', 'INDUS', 'NOX', 'LSTAT', 'TAX']:
                        processed_input[col] = num / 100  # convert % to decimal
                    elif col == 'CHAS':
                        processed_input[col] = int(num)
                    else:
                        processed_input[col] = num

            input_df = pd.DataFrame([processed_input])

            # Fill missing for prediction only
            input_filled = input_df.fillna(data[COLUMNS].mean())
            prediction = model.predict(input_filled)[0]
            st.success(f"💵 Predicted house price: ${prediction * 1000:,.0f}")

            # Save prediction (1 decimal place)
            input_df['MEDV'] = round(prediction, 1)

            # Append to copy file
            existing = pd.read_csv(COPY_FILE)
            new_df = pd.concat([existing, input_df], ignore_index=True)
            new_df.to_csv(COPY_FILE, index=False)

            # Retrain logic
            orig = pd.read_csv(DATA_FILE)
            new_entries = len(new_df) - len(orig)
            if new_entries >= 10:
                valid_rows = new_df.dropna(subset=['MEDV'])
                retrained_model = train_and_save_model(valid_rows)
                valid_rows.to_csv(DATA_FILE, index=False)
                # st.info("🔁 Model retrained with latest data.")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

