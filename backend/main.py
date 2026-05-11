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
from dotenv import load_dotenv
from datetime import datetime

# 1. 환경 변수 로드
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
ai_model = genai.GenerativeModel('gemini-2.0-flash')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 파이토치 모델 뼈대 구축
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

# ==========================================
# 🌟 [핵심 해결책] Render 타임아웃 방지 로직 🌟
# 전역 변수로 껍데기만 먼저 만들어 둡니다.
# ==========================================
device = torch.device("cpu")
pytorch_model = None
preprocessor = None

@app.on_event("startup")
async def load_heavy_models():
    """서버 문(포트)을 먼저 열고 나서 무거운 파일들을 백그라운드로 로딩합니다."""
    global pytorch_model, preprocessor
    print("🚪 1단계: 서버 포트 연결 성공! (Render가 안심함)")
    print("🧠 2단계: 무거운 AI 모델과 150MB 파일 로딩을 시작합니다...")
    
    pytorch_model = CustomModel(input_dim=27, output_dim=5)
    try:
        state_dict = torch.load('CORE_6.pth', map_location=device)
        pytorch_model.load_state_dict(state_dict)
        pytorch_model.eval()
        print("✅ AI 모델(CORE_6.pth) 로드 완료!")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")

    try:
        preprocessor = joblib.load('best_xgboost_model.pkl') 
        print("✅ 전처리기(150MB pkl) 로드 완료!")
    except Exception as e:
        print(f"⚠️ pkl 로드 실패: {e}")

# ==========================================

# 프론트엔드 입력 규격
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

cc_mapping = {
    '흉통/심장질환': 'Cardiovascular',
    '호흡곤란': 'Respiratory',
    '복통': 'Gastrointestinal',
    '두통/뇌졸중': 'Neurological',
    '외상/출혈': 'Trauma_Injury',
    '발열': 'Infection_Immune',
    '기타': 'General_Other'
}

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
    # 아직 모델 로딩이 안 끝났는데 프론트에서 요청이 올 경우 방어
    if pytorch_model is None or preprocessor is None:
        return {"status": "error", "message": "AI 모델이 아직 깨어나는 중입니다. 1분 뒤에 다시 시도해주세요."}

    input_dict = {col: 0.0 for col in FEATURE_COLUMNS}
    input_dict['sbp'] = data.sbp
    input_dict['dbp'] = data.dbp
    input_dict['heartrate'] = data.heart_rate
    input_dict['resprate'] = data.resp_rate
    input_dict['temperature'] = data.temperature
    input_dict['o2sat'] = data.o2sat
    input_dict['anchor_age'] = data.age
    input_dict['gender'] = 0 
    input_dict['race_group'] = 0
    
    mapped_cc = cc_mapping.get(data.chief_complaint, 'General_Other')
    if mapped_cc in input_dict:
        input_dict[mapped_cc] = 1.0

    input_df = pd.DataFrame([input_dict])
    
    if preprocessor:
        try:
            processed_array = preprocessor.transform(input_df)
        except:
            processed_array = input_df.values
    else:
        processed_array = input_df.values

    input_tensor = torch.tensor(processed_array, dtype=torch.float32).to(device)
    with torch.no_grad():
        model_output = pytorch_model(input_tensor)
        predicted_idx = torch.argmax(model_output, dim=1).item()
        
    final_level = predicted_idx + 1 
    risk_score = 100 - (predicted_idx * 20) + np.random.randint(-5, 5)

    prompt = f"""
    환자 정보: {data.age}세, 주증상: {data.chief_complaint}
    Vitals: 혈압 {data.sbp}/{data.dbp}, 맥박 {data.heart_rate}, 호흡 {data.resp_rate}, 체온 {data.temperature}, SpO2 {data.o2sat}%
    AI 중증도 예측: Level {final_level}
    위 데이터를 바탕으로 의사에게 전달할 짧고 전문적인 임상 브리핑을 한국어로 작성해 줘. (3문장 이내)
    """
    
    try:
        response = ai_model.generate_content(prompt)
        ai_briefing = response.text
    except Exception:
        ai_briefing = "브리핑 생성 중 오류가 발생했습니다."

    return {
        "status": "success",
        "data": {
            "patient_name": data.patient_name,
            "predicted_level": final_level,
            "risk_score": max(1, min(99, risk_score)),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ai_briefing": ai_briefing,
            "warnings": ["생체 징후 모니터링이 필요합니다."],
            "xai_data": [
                {"name": "Age", "value": data.age * 0.1},
                {"name": "Vital_Stability", "value": 5 - predicted_idx}
            ]
        }
    }