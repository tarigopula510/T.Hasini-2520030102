
import os
from dotenv import load_dotenv

from llm_client import (
    OllamaProvider,
    OpenAIProvider,
    LLMClient
)

load_dotenv()

print("===== PROVIDER AGNOSTIC LLM CLIENT =====")
print("1. Ollama (Local)")
print("2. OpenAI (Cloud)")

choice = input("Choose provider (1/2): ").strip()

if choice == "1":
    provider = OllamaProvider("qwen2.5:1.5b")

elif choice == "2":
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("OpenAI API key not found in .env")
        raise SystemExit(1)

    provider = OpenAIProvider(api_key)

else:
    print("Invalid choice")
    raise SystemExit(1)

client = LLMClient(provider)

prompt = input("\nEnter your question: ")

try:
    answer = client.generate(prompt)
    print("\n===== LLM RESPONSE =====")
    print(answer)

except Exception as e:
    print("Error:", e)
