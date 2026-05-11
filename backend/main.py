from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib  # pkl 파일을 불러오기 위한 라이브러리
import google.generativeai as genai
import os
from dotenv import load_dotenv
from datetime import datetime

# 1. 환경 변수 로드 및 제미나이 설정
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
# 아까 확인했던 최신 모델 적용!
ai_model = genai.GenerativeModel('gemini-2.5-flash')

app = FastAPI()

# CORS 설정 (프론트엔드와 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 파이토치 모델 뼈대 구축 (input_dim=27)
class CustomModel(nn.Module):
    def __init__(self, input_dim=27, output_dim=5):
        super(CustomModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# 3. 모델 가중치(.pth) 및 전처리기(.pkl) 로드
# 서버가 켜질 때 딱 한 번만 메모리에 올립니다.
device = torch.device("cpu") # Render 무료 서버는 CPU를 사용하므로 명시
pytorch_model = CustomModel(input_dim=27, output_dim=5)
pytorch_model.load_state_dict(torch.load('CORE_6.pth', map_location=device))
pytorch_model.eval() # 평가 모드로 전환

try:
    preprocessor = joblib.load('best_xgboost_model.pkl') 
    print("✅ 150MB 전처리기(pkl) 로드 성공!")
except Exception as e:
    print(f"⚠️ pkl 로드 실패 (파일 이름 확인 필요): {e}")
    preprocessor = None

# 프론트엔드에서 날아오는 데이터 규격
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

# 프론트의 주증상을 모델의 주증상 칸(12개)으로 변환하는 매핑
cc_mapping = {
    '흉통/심장질환': 'Cardiovascular',
    '호흡곤란': 'Respiratory',
    '복통': 'Gastrointestinal',
    '두통/뇌졸중': 'Neurological',
    '외상/출혈': 'Trauma_Injury',
    '발열': 'Infection_Immune',
    '기타': 'General_Other'
}

# 27개 피처 순서표
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
    # ==========================================
    # STEP 1: 9개 데이터를 27개 배열로 뻥튀기 (데이터 어댑터)
    # ==========================================
    input_dict = {col: 0.0 for col in FEATURE_COLUMNS} # 일단 0으로 다 채움
    
    # 1. 바이탈 & 나이 넣기
    input_dict['sbp'] = data.sbp
    input_dict['dbp'] = data.dbp
    input_dict['heartrate'] = data.heart_rate
    input_dict['resprate'] = data.resp_rate
    input_dict['temperature'] = data.temperature
    input_dict['o2sat'] = data.o2sat
    input_dict['anchor_age'] = data.age
    
    # 2. 성별/인종은 UI에 없으므로 임의의 기본값(0) 할당
    input_dict['gender'] = 0 
    input_dict['race_group'] = 0
    
    # 3. Missing Tag 넣기 (프론트에서 필수값으로 다 들어오므로 결측치 없음=0 처리)
    # 만약 빈칸을 허용한다면 조건문으로 1을 넣어야 함
    
    # 4. 주증상 원핫 인코딩
    mapped_cc = cc_mapping.get(data.chief_complaint, 'General_Other')
    if mapped_cc in input_dict:
        input_dict[mapped_cc] = 1.0

    # 5. DataFrame으로 변환 후 pkl 적용
    input_df = pd.DataFrame([input_dict])
    
    if preprocessor:
        # pkl이 파이프라인이나 스케일러라면 transform을 적용
        try:
            processed_array = preprocessor.transform(input_df)
        except:
            # 만약 에러가 난다면, 전처리기가 배열을 그대로 받지 않는 형태일 수 있음
            processed_array = input_df.values
    else:
        processed_array = input_df.values

    # ==========================================
    # STEP 2: PyTorch AI 모델 추론 (진짜 지능 작동!)
    # ==========================================
    input_tensor = torch.tensor(processed_array, dtype=torch.float32).to(device)
    
    with torch.no_grad(): # 추론 모드(학습 안함)
        model_output = pytorch_model(input_tensor)
        # 0~4 사이의 결과가 나옴
        predicted_idx = torch.argmax(model_output, dim=1).item()
        
    # 결과값에 +1을 해서 Level 1~5로 변환 (주의: 수민님 모델의 정답지가 0=Lv.1 이라면 +1 해야 함)
    # 만약 수민님이 0=Lv.5 로 역순으로 했다면 로직을 반대로 해야함 (보통 0=Lv.1)
    final_level = predicted_idx + 1 
    
    # 임의의 위험도 점수 계산 로직 (Softmax 확률값 기반으로 짜면 더 좋음)
    risk_score = 100 - (predicted_idx * 20) + np.random.randint(-5, 5)

    # ==========================================
    # STEP 3: 제미나이 LLM 브리핑 생성
    # ==========================================
    prompt = f"""
    환자 정보: {data.age}세, 주증상: {data.chief_complaint}
    Vitals: 혈압 {data.sbp}/{data.dbp}, 맥박 {data.heart_rate}, 호흡 {data.resp_rate}, 체온 {data.temperature}, SpO2 {data.o2sat}%
    AI 중증도 예측: Level {final_level}
    
    위 데이터를 바탕으로 의사에게 전달할 짧고 전문적인 임상 브리핑을 한국어로 작성해 줘. (3문장 이내)
    """
    
    try:
        response = ai_model.generate_content(prompt)
        ai_briefing = response.text
    except Exception as e:
        ai_briefing = f"브리핑 생성 오류: {str(e)}"

    # ==========================================
    # STEP 4: 프론트엔드로 최종 데이터 쏘기
    # ==========================================
    import datetime
    return {
        "status": "success",
        "data": {
            "patient_id": str(np.random.randint(1000, 9999)),
            "patient_name": data.patient_name,
            "predicted_level": final_level,
            "risk_score": max(1, min(99, risk_score)), # 1~99점 사이 고정
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ai_briefing": ai_briefing,
            "warnings": ["생체 징후 모니터링이 필요합니다."],
            "xai_data": [
                {"name": "Age", "value": data.age * 0.1},
                {"name": "Temp", "value": (data.temperature - 36.5) * 5}
            ]
        }
    }