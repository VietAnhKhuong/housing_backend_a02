import os
import joblib
import pandas as pd
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="VN Housing Price Prediction API")

# Cấu hình CORS để Vercel có thể gọi sang Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hoặc điền domain Vercel của bạn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "house_price_pipeline.joblib"
model_pipeline = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

@app.post("/predict")
async def predict(
    area: float = Form(...),
    bedrooms: float = Form(...),
    floors: float = Form(...),
    district: str = Form(...),
    housing_type: str = Form(...),
    legal_status: str = Form(...)
):
    input_data = pd.DataFrame([{
        "Diện tích": area,
        "Số phòng ngủ": bedrooms,
        "Số tầng": floors,
        "Quận": district,
        "Loại hình nhà ở": housing_type,
        "Giấy tờ pháp lý": legal_status
    }])

    if model_pipeline is not None:
        predicted_price_per_m2 = float(model_pipeline.predict(input_data)[0])
    else:
        predicted_price_per_m2 = 85.5

    total_price = predicted_price_per_m2 * area

    return JSONResponse(content={
        "status": "success",
        "prediction": {
            "predicted_price_per_m2": round(predicted_price_per_m2, 2),
            "estimated_total_price_billion": round(total_price / 1000, 2)
        }
    })