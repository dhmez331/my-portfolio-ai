import os
import google.generativeai as genai
from dotenv import load_dotenv

# تحميل المفتاح من ملف .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("الموديلات المتاحة لمفتاحك للرد والمحادثة هي:")
print("-" * 40)

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print("حدث خطأ في الاتصال بجوجل:", e)