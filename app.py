from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import traceback
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import traceback

# Initialize the FastAPI app
app = FastAPI(title="Diabetes Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your models safely
try:
    clinical_pipe = joblib.load("final_pipe/clinical_pipe.pkl")
    lifestyle_pipe = joblib.load("final_pipe/lifestyle_pipe.pkl")
    print("✅ Models loaded successfully.")
except Exception as e:
    print("❌ Error loading models:", e)
    traceback.print_exc()
    clinical_pipe = None
    lifestyle_pipe = None
    best_pipe_lifestyle = None


@app.get("/")
def root():
    return {"message": "Diabetes Prediction API is running."}


# Define input schema
class UserFeatures(BaseModel):
    username: str
    age: int
    gender: int
    height: float
    weight: float
    waist: float
    hip: float
    systolic: int
    diastolic: int
    hba1c: float
    fbs: float
    cholesterol: float
    hdl: float
    fruits: int
    vegetables: int
    fried: int
    sweets: int
    fastfood: int
    processed: int
    softdrink: int
    weight_concern: int
    doesExercise: int
    exercise_times: int
    exercise_duration: int
    sitting: int
    main_activity: int
    mode_of_transpo: int
    fh_father: int
    fh_mother: int
    fh_sister: int
    fh_brother: int
    fh_extended: int
    sleep_hours: int
    sleep_cigarette: int
    sleep_alcohol: int


@app.post("/predict")
def predict(data: UserFeatures):
    prob_clinical = 0
    prob_lifestyle = 0
    featuresClinical = None
    featuresClinicalwithoutGL = None
    print("Received data for prediction:", data.model_dump())
    
    try: 
     
        featuresClinical = pd.DataFrame([{
            "Age": data.age,
            "Gender": data.gender,
            "Height_cm": data.height,
            "Weight_kg": data.weight,
            "Waist_cm": data.waist,
            "Hip_cm": data.hip,
            "Systolic_BP": data.systolic,
            "Diastolic_BP": data.diastolic,
            "HbA1c": data.hba1c,
            "FBS": data.fbs,
            "HDL": data.hdl,
            "Cholesterol_Total": data.cholesterol
        }])

        
        if (data.doesExercise == 2):
            featuresLifestyle = pd.DataFrame([{
            "alcohol_ord": data.sleep_alcohol,
            "fruit_freq_ord": data.fruits,
            "veg_freq_ord": data.vegetables,
            "sweets_freq_ord": data.sweets,
            "fastfood_freq_ord": data.fastfood,
            "processed_freq_ord": data.processed,
            "sweetdrink_freq_ord": data.softdrink,
            "fried_food_freq_ord": data.fried,
            "lose_weight_ord": data.weight_concern,
            "exercise_yes_ord": data.doesExercise,
            "exercise_freq_ord": data.exercise_times,
            "exercise_duration_ord": data.exercise_duration,
            "sedentary_hours_ord": data.sitting,
            "activity_level_ord": data.main_activity,
            "transpo_ord": data.mode_of_transpo,
            "sleep_ord": data.sleep_hours,
            "smoking_ord": data.sleep_cigarette,
            "father_diab_ord": data.fh_father,
            "mother_diab_ord": data.fh_mother,
            "sister_diab_ord": data.fh_sister,
            "brother_diab_ord": data.fh_brother,
            "extended_diab_ord": data.fh_extended,
        }])
            
        if (data.doesExercise == 1):
            featuresLifestyle = pd.DataFrame([{
            "alcohol_ord": data.sleep_alcohol,
            "fruit_freq_ord": data.fruits,
            "veg_freq_ord": data.vegetables,
            "sweets_freq_ord": data.sweets,
            "fastfood_freq_ord": data.fastfood,
            "processed_freq_ord": data.processed,
            "sweetdrink_freq_ord": data.softdrink,
            "fried_food_freq_ord": data.fried,
            "lose_weight_ord": data.weight_concern,
            "exercise_yes_ord": data.doesExercise,
            "exercise_freq_ord": data.exercise_times,
            "exercise_duration_ord": data.exercise_duration,
            "sedentary_hours_ord": data.sitting,
            "activity_level_ord": data.main_activity,
            "transpo_ord": data.mode_of_transpo,
            "sleep_ord": data.sleep_hours,
            "smoking_ord": data.sleep_cigarette,
            "father_diab_ord": data.fh_father,
            "mother_diab_ord": data.fh_mother,
            "sister_diab_ord": data.fh_sister,
            "brother_diab_ord": data.fh_brother,
            "extended_diab_ord": data.fh_extended,
        }])

        
        
        prob_clinical = clinical_pipe.predict_proba(featuresClinical)[0][1]
        prob_lifestyle = lifestyle_pipe.predict_proba(featuresLifestyle)[0][1]

        prob_combined = (prob_clinical * 0.6) + (prob_lifestyle * 0.4)
        prob_percent = round(prob_combined * 100, 2)

        print(f"Prediction successful: Clinical Prob={prob_clinical * 100}, Lifestyle Prob={prob_lifestyle*100}, Combined Prob={prob_combined}, Percent={prob_percent}% ")
        return {
            #"prediction": int(prediction),
            #"probability": round(float(prob_combined) * 100, 2),
            "clinical":prob_clinical, "lifestyle": prob_lifestyle, "combined": prob_combined, "percent": prob_percent
        }

    except Exception as e:
        print("❌ Error in /predict:", e)
        traceback.print_exc()
        print("Received data:",data.model_dump())
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc(), "data": data.model_dump()},
        )