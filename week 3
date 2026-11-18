from llm_client import OllamaProvider, LLMClient
from mongo_logger import MongoLogger

print("================================")
print("       WEEK 3: LLM MONITORING")
print("================================")

logger = None

try:
    logger = MongoLogger()

    provider = OllamaProvider("llama3.2")
    client = LLMClient(provider, logger)

    prompt = input("\nEnter your question: ")

    print("\nGenerating AI response...\n")

    answer = client.generate(prompt)

    print("========== AI RESPONSE ==========")
    print(answer)
    print("\n========== COMPLETED ==========")

except Exception as error:
    print("\nApplication error:", error)

finally:
    if logger is not None:
        logger.close()
