from fastapi import FastAPI
from pydantic import BaseModel

from ml_service.model import AccidentModel

app = FastAPI()

model = AccidentModel()

class FeaturePayload(BaseModel):

    acc_delta: float
    gyro_delta: float
    vibration_intensity: float
    impact_duration: float
    airbag_deployed: int
    wheel_speed_drop_pct: float
    thermal_c: float
    latitude: float
    longitude: float
    initial_speed: float

@app.get("/")
def root():
    return {"service":"VANET ML"}

@app.post("/predict")
def predict(payload:FeaturePayload):

    result = model.predict(payload.dict())

    return result
