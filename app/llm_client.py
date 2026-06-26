class LLMClient:
    """
    大模型调用客户端。

    当前 Day 3 是模拟版本。
    后续会扩展为支持 OpenAI-compatible API、vLLM、本地模型服务等。
    """

    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key

    def summarize_text(self, text: str, max_chars: int = 3000) -> str:
        """
        总结文本内容。

        Day 3 暂时返回模拟摘要，不真正调用大模型。
        """
        if not text or not text.strip():
            return "文本为空，无法总结。"

        clipped_text = text[:max_chars]

        return (
            f"这是一个模拟摘要，原文长度为 {len(text)} 个字符，"
            f"实际用于摘要的文本长度为 {len(clipped_text)} 个字符。"
        )