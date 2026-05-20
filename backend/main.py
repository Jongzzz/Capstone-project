import os
import io
import base64
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from google import genai as google_genai
from dotenv import load_dotenv

load_dotenv()
google_genai_client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(root_path="/")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TriageNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TriageNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 5)
        )
    def forward(self, x):
        return self.network(x)

print("🧠 [Ensemble Mode] XGBoost와 PyTorch 동시 로딩 중...")
xgb_model = None
pytorch_model = None

try:
    xgb_model = joblib.load('best_xgboost_model.pkl')
    print("✅ XGBoost 로딩 성공!")

    state_dict = torch.load('CORE_6.pth', map_location=torch.device('cpu'))
    input_size = state_dict['network.0.weight'].shape[1]
    hidden_size = state_dict['network.0.weight'].shape[0]
    pytorch_model = TriageNet(input_size=input_size, hidden_size=hidden_size)
    pytorch_model.load_state_dict(state_dict)
    pytorch_model.eval()
    print("✅ PyTorch 로딩 성공!")
except Exception as e:
    print(f"❌ 모델 로딩 실패: {e}")

CAT_FEATURES = [
    'gender', 'race_group', 'sbp_missing', 'dbp_missing', 'heartrate_missing',
    'resprate_missing', 'temperature_missing', 'o2sat_missing', 'Gastrointestinal',
    'Cardiovascular', 'Respiratory', 'Neurological', 'Trauma_Injury',
    'Psychiatric_Substance', 'Infection_Immune', 'Musculoskeletal_Pain',
    'Genitourinary_Obstetric', 'ENT_Ophthalmology', 'Endocrine_Metabolic', 'General_Other'
]

def apply_clinical_rules(data, model_pred):
    if data.spo2 < 90:           return 1
    if data.systolic < 80:       return 1
    if data.systolic > 220:      return 1
    if data.diastolic > 130:     return 1
    if data.heart_rate > 150:    return 1
    if data.heart_rate < 40:     return 1
    if data.temperature > 41.0:  return 1
    if data.temperature < 32.0:  return 1
    if data.resp_rate > 35:      return 1
    if data.resp_rate < 8:       return 1

    if data.spo2 < 94:           return min(model_pred, 2)
    if data.systolic < 90:       return min(model_pred, 2)
    if data.systolic > 180:      return min(model_pred, 2)
    if data.heart_rate > 130:    return min(model_pred, 2)
    if data.heart_rate < 50:     return min(model_pred, 2)
    if data.temperature > 39.5:  return min(model_pred, 2)
    if data.temperature < 34.0:  return min(model_pred, 2)
    if data.resp_rate > 28:      return min(model_pred, 2)

    if (data.spo2 >= 97 and 110 <= data.systolic <= 140 and 60 <= data.diastolic <= 90 and
        60 <= data.heart_rate <= 100 and 36.0 <= data.temperature <= 37.5 and 12 <= data.resp_rate <= 20):
        return max(model_pred, 4)
    return model_pred

class PatientData(BaseModel):
    age: float
    gender: int
    arrival_mode: int
    pain: int
    systolic: float
    diastolic: float
    temperature: float
    heart_rate: float
    resp_rate: float
    spo2: float
    chief_complaint: str

@app.post("/api/triage/predict")
async def predict(data: PatientData):
    global xgb_model, pytorch_model
    if xgb_model is None or pytorch_model is None:
        raise HTTPException(status_code=500, detail="서버 모델 준비 안됨")

    try:
        cc = data.chief_complaint
        gastro = 1 if "복통" in cc or "위장" in cc else 0
        cardio = 1 if "흉통" in cc or "심장" in cc else 0
        resp   = 1 if "호흡" in cc else 0
        neuro  = 1 if "두통" in cc or "뇌졸중" in cc else 0
        trauma = 1 if "외상" in cc or "출혈" in cc else 0
        infect = 1 if "발열" in cc or "감염" in cc else 0
        other  = 1 if not any([gastro, cardio, resp, neuro, trauma, infect]) else 0
        mapped_gender = 1 if data.gender == 1 else 0

        xgb_mapped = {
            'sbp': float(data.systolic), 'dbp': float(data.diastolic), 'heartrate': float(data.heart_rate),
            'resprate': float(data.resp_rate), 'temperature': float(data.temperature), 'o2sat': float(data.spo2),
            'anchor_age': float(data.age), 'gender': mapped_gender, 'race_group': 0,
            'sbp_missing': 0, 'dbp_missing': 0, 'heartrate_missing': 0, 'resprate_missing': 0,
            'temperature_missing': 0, 'o2sat_missing': 0, 'Gastrointestinal': gastro,
            'Cardiovascular': cardio, 'Respiratory': resp, 'Neurological': neuro,
            'Trauma_Injury': trauma, 'Psychiatric_Substance': 0, 'Infection_Immune': infect,
            'Musculoskeletal_Pain': 0, 'Genitourinary_Obstetric': 0, 'ENT_Ophthalmology': 0,
            'Endocrine_Metabolic': 0, 'General_Other': other
        }

        xgb_df = pd.DataFrame([xgb_mapped])
        for col in CAT_FEATURES:
            xgb_df[col] = xgb_df[col].astype(float).astype(int).astype('category')

        xgb_result = int(xgb_model.predict(xgb_df)[0])
        xgb_pred = max(1, min(5, xgb_result + 1))
        final_prediction = apply_clinical_rules(data, xgb_pred)

        # ==========================================
        # 📊 SHAP 시각화: 1D 평탄화 무적 로직 (차원 오류 완벽 차단)
        # ==========================================
        shap_base64 = ""
        try:
            # 1. 5차원 확률(proba)을 버리고, 가장 직관적인 단일 예측값(predict)으로 래핑
            def predict_wrapper(data_matrix):
                temp_df = pd.DataFrame(data_matrix, columns=xgb_df.columns)
                for col in CAT_FEATURES:
                    temp_df[col] = temp_df[col].astype(float).astype(int).astype('category')
                return xgb_model.predict(temp_df)

            # 2. 기준점(Baseline) 설정: 건강한 정상 수치
            baseline_mapped = xgb_mapped.copy()
            baseline_mapped.update({
                'sbp': 120.0, 'dbp': 80.0, 'heartrate': 75.0, 'resprate': 16.0,
                'temperature': 36.5, 'o2sat': 98.0, 'anchor_age': 40.0,
                'Gastrointestinal': 0, 'Cardiovascular': 0, 'Respiratory': 0,
                'Neurological': 0, 'Trauma_Injury': 0, 'Psychiatric_Substance': 0,
                'Infection_Immune': 0, 'Musculoskeletal_Pain': 0, 'Genitourinary_Obstetric': 0,
                'ENT_Ophthalmology': 0, 'Endocrine_Metabolic': 0, 'General_Other': 1
            })
            baseline_df = pd.DataFrame([baseline_mapped])
            for col in CAT_FEATURES:
                baseline_df[col] = baseline_df[col].astype(float).astype(int).astype('category')

            # 3. KernelExplainer 구동 (이제 1차원 데이터만 나옴)
            explainer = shap.KernelExplainer(predict_wrapper, baseline_df)
            shap_values_raw = explainer.shap_values(xgb_df)
            
            # 4. 차원(Dimension) 강제 평탄화 (어떤 변수 배열이 오든 무조건 1차원으로 분쇄)
            sv_array = np.array(shap_values_raw)
            if len(sv_array.shape) == 3: 
                sv = sv_array[0, :, 0]
            elif len(sv_array.shape) == 2:
                sv = sv_array[0]
            else:
                sv = sv_array.flatten()

            bv = explainer.expected_value
            if isinstance(bv, (list, np.ndarray)):
                bv = float(bv[0])
            else:
                bv = float(bv)

            # 폭포수 차트 조립
            exp = shap.Explanation(
                values=sv,
                base_values=bv,
                data=xgb_df.iloc[0].values.astype(float),
                feature_names=list(xgb_df.columns)
            )
            
            plt.figure(figsize=(7, 4))
            shap.plots.waterfall(exp, max_display=6, show=False)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight', dpi=100)
            plt.close()
            buf.seek(0)
            shap_base64 = base64.b64encode(buf.read()).decode("utf-8")
            print("✅ SHAP 이미지 1차원 파싱 성공!")

        except Exception as e:
            print(f"SHAP 완벽 우회 생성 실패: {e}")
            plt.close()

        # ==========================================
        # 🏆 최적화 프롬프트 + 제미나이 2.5 이중 우회
        # ==========================================
        opinion_text = ""
        try:
            response = google_genai_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"""
                환자: {data.dict()}
                ESI {final_prediction}단계 판정.
                진단을 내려선 안된다고 하고 의심되는 질병들을 출력해달라 의학 용어로 1~2문장으로만 작성.
                다른 말 일절 금지.
                """
            )
            opinion_text = response.text.strip()
        except Exception as e1:
            try:
                response = google_genai_client.models.generate_content(
                    model='gemini-1.5-pro',
                    contents=f"""
                    환자: {data.dict()}
                    ESI {final_prediction}단계 판정.
                    진단을 내려선 안된다고 하고 의심되는 질병들을 출력해달라 의학 용어로 1~2문장으로만 작성.
                    다른 말 일절 금지.
                    """
                )
                opinion_text = response.text.strip()
            except Exception as e2:
                opinion_text = "[서버 트래픽 과부하로 텍스트 소견이 지연되었습니다.]"

        return {
            "ktas_level": final_prediction,
            "opinion": opinion_text,
            "shap_image": shap_base64,
            "status": "success"
        }

    except Exception as e:
        print(f"Predict Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "C.O.R.E Backend is Running!"}