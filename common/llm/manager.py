import google.generativeai as genai
from typing_extensions import Self

from common.config.setup import Config


class GeminiApi:
    """TODO: Replace this with ChatGPT"""
    _instance = None

    def __new__(cls, *args, **kwargs) -> Self:
        if cls._instance is None:
            cls._instance = super(GeminiApi, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, model_name: str = "gemini-1.5-flash") -> None:
        if self.__initialized:
            return

        self.__initialized = True
        genai.configure(api_key=Config().GEMINI_API_KEY)
        self.model = genai.GenerativeModel(model_name=model_name)
        self.chat_client = self.model.start_chat(history=[])

    def chat(self, prompt: str) -> str:
        response = self._instance.chat_client.send_message(prompt)
        return response.text

    @property
    def instance(self):
        return self._instance


class AzureOpenAI:
    _instance = None