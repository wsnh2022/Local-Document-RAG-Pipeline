import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 3

def call_openrouter(messages: list[dict]) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in .env")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "lancedb_chunking_project",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.2,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=30
            )
            if response.status_code == 401:
                raise ValueError("Invalid OpenRouter API key — check .env")
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            print(f"  Timeout on attempt {attempt}/{MAX_RETRIES}")
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenRouter API error: {e}")

    raise RuntimeError("All retry attempts failed")