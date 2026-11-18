import requests
import time
from cost_calculator import calculate_cost

class OllamaProvider:
    def __init__(self, model="llama3.2"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt):
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()
        data = response.json()

        return {
            "response": data["response"],
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0)
        }


class LLMClient:
    def __init__(self, provider, logger):
        self.provider = provider
        self.logger = logger

    def generate(self, prompt):
        start_time = time.perf_counter()

        try:
            result = self.provider.generate(prompt)

            latency = round(
                (time.perf_counter() - start_time) * 1000, 2
            )

            input_tokens = result["input_tokens"]
            output_tokens = result["output_tokens"]

            log_data = {
                "provider": type(self.provider).__name__,
                "model": self.provider.model,
                "prompt": prompt,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "latency_ms": latency,
                "cost": calculate_cost(
                    type(self.provider).__name__,
                    self.provider.model,
                    input_tokens,
                    output_tokens
                ),
                "attempts": 1,
                "status": "success",
                "error": None
            }

            try:
                self.logger.log_call(log_data)
            except Exception as error:
                print("MongoDB logging error:", error)

            return result["response"]

        except Exception as error:
            raise error
