from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI app
app = FastAPI(title="Diabetes Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
best_pipe = joblib.load("best_pipe.pkl")


@app.get("/")
def root():
    return {"message": "Hello, World!"}

@app.get("/predict")
def root():
    return {"message": "Predicting"}

best_pipe = joblib.load("best_pipe.pkl")


# Define the input schema (must match your model’s features)
class UserFeatures(BaseModel):
    Age: int
    Gender: int
    Height_cm: int
    Weight_kg: int
    Waist_cm: int
    Hip_cm: int
    Systolic_BP: int
    Diastolic_BP: int
    # Add all other features your pipeline expects

@app.post("/predict")
def predict(data: UserFeatures):
    
    # Convert input to 2D array for sklearn
    features = pd.DataFrame([{
    "Age": data.Age,
    "Gender": data.Gender,
    "Height_cm": data.Height_cm,
    "Weight_kg": data.Weight_kg,
    "Waist_cm": data.Waist_cm,
    "Hip_cm": data.Hip_cm,
    "Systolic_BP": data.Systolic_BP,
    "Diastolic_BP": data.Diastolic_BP,
}])

    # Run the pipeline
    prediction = best_pipe.predict(features)[0]
    probability = (
        best_pipe.predict_proba(features)[0][1]
        if hasattr(best_pipe, "predict_proba")
        else None
    )

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 2) if probability is not None else None,
    }