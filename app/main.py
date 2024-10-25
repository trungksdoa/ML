from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.koi_health_predictor import load_trained_model, predict_koi_health, incorporate_feedback
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Load the model at startup
model_path = '/app/models/koi_health_model'

if not os.path.exists(model_path):
    raise HTTPException(status_code=500, detail="Model files not found. Please ensure the model is saved before running the API.")

model, tokenizer = load_trained_model(model_path)

class KoiData(BaseModel):
    age_months: int
    length_cm: float
    weight_g: int
    water_temp: float
    ph: float
    ammonia: float
    nitrite: float
    activity_level: str


@app.post("/predict_koi_health")
async def predict_health(koi_data: KoiData):
    result = predict_koi_health(model, tokenizer, koi_data.dict())
    return result

@app.get("/")
async def root():
    return {"message": "Koi Health Prediction API"}

class FeedbackData(BaseModel):
    age_months: int
    length_cm: float
    weight_g: int
    water_temp: float
    ph: float
    ammonia: float
    nitrite: float
    activity_level: str
    true_health_status: str

@app.post("/feedback")
async def provide_feedback(feedback_data: FeedbackData):
    result = incorporate_feedback(model, tokenizer, feedback_data.dict(), model_path)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9001)
