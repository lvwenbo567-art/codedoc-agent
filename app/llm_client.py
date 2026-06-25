from typing import List


class LLMClient:
    """
    大模型调用客户端。

    后续用于兼容 OpenAI-compatible API、vLLM、本地模型服务等。
    """

    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key

    def summarize_text(self, text: str) -> str:
        """
        总结文本内容。

        Day 2 暂时返回模拟结果，后续再接真实大模型 API。
        """
        if not text.strip():
            return "文本为空，无法总结。"

        return f"这是一个模拟摘要，原文长度为 {len(text)} 个字符。"