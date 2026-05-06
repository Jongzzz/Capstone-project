from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import pytz
import random
import os
from dotenv import load_dotenv
import google.generativeai as genai

# ==========================================
# 🌟 1. 구글 Gemini API 세팅 및 디버깅
# ==========================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 서버 실행 시 터미널에서 API 키 로드 여부 즉각 확인
print("="*50)
if GEMINI_API_KEY:
    # 보안을 위해 키의 앞부분만 출력하여 확인
    print(f"✅ API 키 로드 성공: {GEMINI_API_KEY[:10]}...") 
else:
    print("❌ API 키 로드 실패: .env 파일의 위치와 오타를 확인해 주세요.")
print("="*50)

# API 키가 있으면 Gemini 모델 활성화
ai_model = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    ai_model = genai.GenerativeModel('gemini-2.5-flash')

app = FastAPI(title="CDSS Triage API", description="룰베이스 경고 + Gemini AI 비서 브리핑 통합 백엔드")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class PatientInput(BaseModel):
    patient_name: str
    chief_complaint: str
    age: int
    temperature: float
    heart_rate: int
    resp_rate: int
    o2sat: float
    sbp: int
    dbp: int
    pain_score: int

@app.post("/api/triage/predict")
async def predict_triage(data: PatientInput):
    KST = pytz.timezone('Asia/Seoul')
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
    
    warnings = []
    pred_level = 4
    risk_score = 30.0 + random.uniform(0, 5)
    shap_vitals = {'SpO2': 0.0, 'SBP': 0.0, 'Pain': 0.0, 'Age': 0.0, 'Temp': 0.0}

    # ==========================================
    # 🧠 기존 룰베이스 로직 (유지)
    # ==========================================
    if data.o2sat < 90.0:
        risk_score = 95.5
        pred_level = 1
        shap_vitals['SpO2'] += 25.0
        warnings.append(f"[호흡기계] SpO2 {data.o2sat}%로 심각한 저산소증 소견을 보입니다. 즉각적인 하이플로우 산소 투여 및 기도 확보를 권고합니다.")
    elif data.o2sat < 94.0:
        risk_score = max(risk_score, 82.0)
        pred_level = min(pred_level, 2)
        shap_vitals['SpO2'] += 12.0
        warnings.append(f"[호흡기계] SpO2 {data.o2sat}%로 경도 저산소증이 관찰됩니다. 지속적인 산소 포화도 모니터링이 필요합니다.")

    if data.sbp <= 90.0:
        risk_score = max(risk_score, 92.5)
        pred_level = min(pred_level, 1)
        shap_vitals['SBP'] += 20.0
        warnings.append(f"[순환기계] 수축기 혈압 {data.sbp}mmHg로 쇼크(저혈압) 위험이 있습니다. 수액 소생술 및 승압제 준비를 권장합니다.")
    elif data.age >= 80 and data.sbp >= 150:
        shap_vitals['SBP'] += 8.0
        warnings.append(f"[순환기계] 80세 이상 고령 환자로, 수축기 혈압 {data.sbp}mmHg 기준 노인성 고혈압 통제 가이드라인 적용 대상입니다.")
    elif data.sbp >= 140 or data.dbp >= 90:
        shap_vitals['SBP'] += 5.0
        warnings.append(f"[순환기계] 고혈압 위험 수준의 혈압(SBP {data.sbp} / DBP {data.dbp})이 관찰됩니다.")

    if data.pain_score >= 8:
        risk_score = max(risk_score, 68.0)
        pred_level = min(pred_level, 3)
        shap_vitals['Pain'] += 15.0
        warnings.append(f"[통증평가] NRS {data.pain_score}점의 극심한 통증을 호소하고 있습니다. 신속한 진통제 투여를 고려하십시오.")

    if data.temperature >= 39.0:
        shap_vitals['Temp'] += 10.0
        warnings.append(f"[감염내과] 체온 {data.temperature}℃의 고열 소견이 있습니다. 패혈증 배제를 위한 혈액 배양 검사를 권장합니다.")

    if data.age >= 65:
        shap_vitals['Age'] += 5.0 

    xai_data = [{'name': k, 'value': round(v, 1)} for k, v in shap_vitals.items() if v != 0]
    xai_data = sorted(xai_data, key=lambda x: abs(x['value']), reverse=True)

    if len(warnings) == 0:
        warnings.append("[종합소견] 현재 입력된 활력징후 상 특이 소견이나 즉각적인 처치를 요하는 위험 징후가 관찰되지 않습니다.")

    # ==========================================
    # 🤖 2. 신규: Gemini AI 비서 프롬프트 및 호출 (에러 추적 강화)
    # ==========================================
    ai_briefing = ""
    if ai_model:
        try:
            # AI에게 던져줄 프롬프트 조립
            prompt = f"""
            너는 응급의학과 전문의야. 다음 환자의 바이탈과 AI 예측 등급을 보고, 간호사나 다른 의사에게 인계할 '핵심 임상 브리핑'을 3문장 이내로 작성해줘.
            말투는 전문가답고 간결한 '-입니다', '-바랍니다', '-요망됨' 체를 사용해.

            [환자 데이터]
            - 나이: {data.age}세
            - 주호소: {data.chief_complaint}
            - 체온: {data.temperature}℃ / 심박수: {data.heart_rate}회/분 / 호흡수: {data.resp_rate}회/분
            - 산소포화도: {data.o2sat}% / 혈압: {data.sbp}/{data.dbp}mmHg
            - 예측 중증도: ESI {pred_level}등급
            """
            
            # API 호출
            response = ai_model.generate_content(prompt)
            
            # 정상적으로 텍스트를 받아온 경우
            if response and response.text:
                ai_briefing = response.text.strip()
            else:
                ai_briefing = "AI가 빈 응답을 반환했습니다."
                print("🚨 제미나이 호출 경고: 빈 응답 반환")
                
        except Exception as e:
            # 프론트엔드 화면과 터미널 양쪽 모두에 에러 내용 출력
            error_msg = str(e)
            print(f"\n🚨 제미나이 API 호출 실패!\n원인: {error_msg}\n")
            ai_briefing = f"AI 브리핑 생성 오류: {error_msg}"
    else:
        ai_briefing = "서버에 API 키가 등록되지 않아 AI 브리핑 기능이 비활성화되었습니다."

    # ==========================================
    # 🚀 3. 최종 JSON 응답 반환
    # ==========================================
    return {
        "status": "success",
        "data": {
            "patient_id": random.randint(1000, 9999),
            "patient_name": data.patient_name,
            "predicted_level": pred_level,
            "risk_score": round(risk_score, 1),
            "warnings": warnings,            
            "ai_briefing": ai_briefing,      
            "timestamp": now,
            "xai_data": xai_data
        }
    }

@app.get("/")
async def root():
    return {"message": "CDSS Backend is running with Gemini AI!"}