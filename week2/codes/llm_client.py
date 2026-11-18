
from abc import ABC, abstractmethod
import requests

class LLMProvider(ABC):

    @abstractmethod
    def generate(self, prompt):
        pass


class OllamaProvider(LLMProvider):

    def __init__(self, model="qwen2.5:1.5b"):
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
        return response.json()["response"]


class OpenAIProvider(LLMProvider):

    def __init__(self, api_key):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt):
        response = self.client.responses.create(
            model="gpt-5.4",
            input=prompt
        )
        return response.output_text


class LLMClient:

    def __init__(self, provider):
        self.provider = provider

    def generate(self, prompt):
        return self.provider.generate(prompt)
