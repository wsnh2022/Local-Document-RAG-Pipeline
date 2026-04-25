from llm.openrouter_client import call_openrouter

reply = call_openrouter([
    {"role": "user", "content": "Say hello in one sentence."}
])
print("OpenRouter reply:", reply)