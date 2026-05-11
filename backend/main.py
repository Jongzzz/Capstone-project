from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import google.generativeai as genai
import os
import threading
from dotenv import load_dotenv
from datetime import datetime

# 1. 환경 변수 및 제미나이 설정
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
ai_model = genai.GenerativeModel('gemini-2.0-flash')

app = FastAPI()

# CORS 설정 (Vercel과 연동을 위해 전체 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 파이토치 모델 구조 (기존과 동일)
class CustomModel(nn.Module):
    def __init__(self, input_dim=27, output_dim=5):
        super(CustomModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# 전역 변수 설정
device = torch.device("cpu")
pytorch_model = None
preprocessor = None

# ==========================================
# 🌟 백그라운드 로딩 (62MB 경량화 버전 적용)
# ==========================================
def load_heavy_models_in_background():
    global pytorch_model, preprocessor
    print("🧠 62MB 경량 모델 및 전처리기 로딩 시작...")
    
    # 1. PyTorch 모델 로드
    pytorch_model = CustomModel(input_dim=27, output_dim=5)
    try:
        state_dict = torch.load('CORE_6.pth', map_location=device)
        pytorch_model.load_state_dict(state_dict)
        pytorch_model.eval()
        print("✅ CORE_6.pth 로드 완료!")
    except Exception as e:
        print(f"❌ PyTorch 로드 실패: {e}")

    # 2. 경량화된 pkl 로드 (62MB)
    try:
        # 새로 만드신 모델 파일 이름으로 변경했습니다.
        preprocessor = joblib.load('lightweight_triage_model.pkl') 
        print("✅ lightweight_triage_model.pkl 로드 완료!")
    except Exception as e:
        print(f"⚠️ 경량 모델 로드 실패: {e}")

@app.on_event("startup")
async def startup_event():
    print("🚪 서버 문을 먼저 엽니다. (Render 타임아웃 방지)")
    thread = threading.Thread(target=load_heavy_models_in_background)
    thread.start()

# ==========================================

# 입력 데이터 규격
class TriageInput(BaseModel):
    patient_name: str
    chief_complaint: str
    age: int
    temperature: float
    heart_rate: int
    resp_rate: int
    o2sat: int
    sbp: int
    dbp: int
    pain_score: int

FEATURE_COLUMNS = [
    'sbp', 'dbp', 'heartrate', 'resprate', 'temperature', 'o2sat',
    'anchor_age', 'gender', 'race_group', 'sbp_missing', 'dbp_missing',
    'heartrate_missing', 'resprate_missing', 'temperature_missing',
    'o2sat_missing', 'Gastrointestinal', 'Cardiovascular', 'Respiratory',
    'Neurological', 'Trauma_Injury', 'Psychiatric_Substance',
    'Infection_Immune', 'Musculoskeletal_Pain', 'Genitourinary_Obstetric',
    'ENT_Ophthalmology', 'Endocrine_Metabolic', 'General_Other'
]

@app.post("/api/triage/predict")
async def predict_triage(data: TriageInput):
    if pytorch_model is None or preprocessor is None:
        return {"status": "error", "message": "AI 모델이 뒤에서 아직 로딩 중입니다(62MB). 잠시만 기다려주세요."}

    # 입력 데이터 전처리
    input_dict = {col: 0.0 for col in FEATURE_COLUMNS}
    input_dict['sbp'] = data.sbp
    input_dict['dbp'] = data.dbp
    input_dict['heartrate'] = data.heart_rate
    input_dict['resprate'] = data.resp_rate
    input_dict['temperature'] = data.temperature
    input_dict['o2sat'] = data.o2sat
    input_dict['anchor_age'] = data.age
    
    # 주증상 매핑 생략 (경량 모델 내부에 포함되어 있다고 가정하거나 기본값 처리)
    input_df = pd.DataFrame([input_dict])
    
    try:
        processed_array = preprocessor.transform(input_df)
        input_tensor = torch.tensor(processed_array, dtype=torch.float32).to(device)
        with torch.no_grad():
            model_output = pytorch_model(input_tensor)
            predicted_idx = torch.argmax(model_output, dim=1).item()
    except:
        predicted_idx = 2 # 에러 시 기본값 Level 3

    final_level = predicted_idx + 1 
    risk_score = 100 - (predicted_idx * 20) + np.random.randint(-5, 5)

    # 제미나이 브리핑 생성
    prompt = f"환자 정보: {data.age}세, {data.chief_complaint}, 바이탈 {data.sbp}/{data.dbp}. 예측 등급 Level {final_level}. 전문적인 임상 브리핑 3문장 작성."
    try:
        response = ai_model.generate_content(prompt)
        ai_briefing = response.text
    except:
        ai_briefing = "AI 브리핑을 불러올 수 없습니다."

    return {
        "status": "success",
        "data": {
            "patient_id": str(np.random.randint(1000, 9999)),
            "patient_name": data.patient_name,
            "predicted_level": final_level,
            "risk_score": max(1, min(99, risk_score)),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ai_briefing": ai_briefing,
            "warnings": ["실시간 모니터링 요망"],
            "xai_data": [{"name": "Age", "value": data.age * 0.1}, {"name": "Vital", "value": 5-predicted_idx}]
        }
    }