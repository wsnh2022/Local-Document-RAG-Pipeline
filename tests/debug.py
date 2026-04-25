import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("OPENROUTER_MODEL")

print("API Key (first 10 chars):", api_key[:10] if api_key else "NOT SET")
print("Model:", model)

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    json={
        "model": model,
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 50,
    },
    timeout=30
)

print("Status:", response.status_code)
print("Body:", response.text)