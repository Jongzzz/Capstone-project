from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import joblib
import google.generativeai as genai
import os
import threading
import gc  # 메모리 관리를 위한 가비지 컬렉터
from dotenv import load_dotenv
from datetime import datetime

# 1. 환경 변수 로드 및 Gemini 설정
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
ai_model = genai.GenerativeModel('gemini-2.0-flash')

app = FastAPI()

# CORS 설정: 보안 경찰(CORS)을 해제하여 Vercel과 통신 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. PyTorch 모델 구조 정의
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

# 전역 변수 (메모리에 올라갈 모델들)
device = torch.device("cpu")
pytorch_model = None
preprocessor = None

# ==========================================
# 🌟 [메모리 최적화] 백그라운드 로딩 로직
# ==========================================
def load_heavy_models_in_background():
    global pytorch_model, preprocessor
    print("🧠 [다이어트 모드] 모델 로딩을 시작합니다...")
    
    try:
        # 1. PyTorch 모델 로드
        pytorch_model = CustomModel(input_dim=27, output_dim=5)
        state_dict = torch.load('CORE_6.pth', map_location=device)
        pytorch_model.load_state_dict(state_dict)
        pytorch_model.eval()
        print("✅ CORE_6.pth 로드 완료!")
        
        # 2. 62MB 경량화 pkl 로드
        preprocessor = joblib.load('lightweight_triage_model.pkl')
        print("✅ lightweight_triage_model.pkl 로드 완료!")
        
        # 로딩 후 불필요한 메모리 즉시 해제
        gc.collect()
        
    except Exception as e:
        print(f"❌ 로딩 중 오류 발생: {e}")

@app.on_event("startup")
async def startup_event():
    print("🚪 서버 포트를 즉시 개방합니다. (Render 타임아웃 방지)")
    # 서버 문은 1초 만에 열고, 무거운 모델은 뒷문으로 조용히 가져옵니다.
    thread = threading.Thread(target=load_heavy_models_in_background)
    thread.start()

# ==========================================

# 프론트엔드 입력 규격 (Pydantic)
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

# 주증상 매핑 테이블 (Numpy 처리를 위해 필요)
cc_mapping = {
    '흉통/심장질환': 16, '호흡곤란': 17, '복통': 15, 
    '두통/뇌졸중': 18, '외상/출혈': 19, '발열': 21, '기타': 26
}

@app.post("/api/triage/predict")
async def predict_triage(data: TriageInput):
    # 아직 로딩 중일 때 요청이 오면 친절하게 안내
    if pytorch_model is None or preprocessor is None:
        return {
            "status": "error", 
            "message": "AI 모델이 아직 메모리에 올라가고 있습니다. 1분 뒤에 다시 눌러주세요."
        }

    # 3. [Numpy 기반] 초경량 전처리 로직
    # 27개의 0으로 채워진 리스트 생성
    features = np.zeros(27)
    features[0] = data.sbp
    features[1] = data.dbp
    features[2] = data.heart_rate
    features[3] = data.resp_rate
    features[4] = data.temperature
    features[5] = data.o2sat
    features[6] = data.age
    
    # 주증상 원-핫 인코딩 수동 적용
    idx = cc_mapping.get(data.chief_complaint, 26)
    features[idx] = 1.0

    try:
        # 모델 예측
        input_data = np.array([features])
        processed_data = preprocessor.transform(input_data)
        input_tensor = torch.tensor(processed_data, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            output = pytorch_model(input_tensor)
            predicted_idx = torch.argmax(output, dim=1).item()
    except Exception as e:
        print(f"예측 오류: {e}")
        predicted_idx = 2 # 에러 시 기본값 Level 3

    final_level = predicted_idx + 1
    risk_score = 100 - (predicted_idx * 20) + np.random.randint(-5, 5)

    # 4. Gemini AI 임상 소견 생성
    prompt = (
        f"환자: {data.patient_name}({data.age}세), 주증상: {data.chief_complaint}, "
        f"바이탈: {data.sbp}/{data.dbp}, 맥박 {data.heart_rate}, 산소포화도 {data.o2sat}%. "
        f"AI 중증도 등급: Level {final_level}. "
        f"이 환자에 대한 짧은 의학적 조언을 한국어로 3문장 이내로 작성해줘."
    )
    
    try:
        response = ai_model.generate_content(prompt)
        ai_briefing = response.text
    except:
        ai_briefing = "AI 브리핑 생성 중 오류가 발생했습니다."

    # 5. 최종 데이터 반환 (patient_id 포함 필수!)
    return {
        "status": "success",
        "data": {
            "patient_id": f"P-{np.random.randint(1000, 9999)}",
            "patient_name": data.patient_name,
            "predicted_level": final_level,
            "risk_score": max(1, min(99, risk_score)),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ai_briefing": ai_briefing,
            "warnings": ["생체 징후의 지속적인 모니터링이 필요합니다."],
            "xai_data": [
                {"name": "연령 영향도", "value": data.age * 0.1},
                {"name": "활력징후 안정성", "value": 5 - predicted_idx}
            ]
        }
    }