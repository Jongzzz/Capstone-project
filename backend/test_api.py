import google.generativeai as genai
import os
from dotenv import load_dotenv

# .env 파일에서 GEMINI_API_KEY를 가져옵니다
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)

print("🚀 종환님의 API 키로 쓸 수 있는 제미나이 모델 목록:")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print("-", m.name)
except Exception as e:
    print("에러 발생:", e)