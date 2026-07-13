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
    
def generate_chat_response(
    prompt: str,
    model_name: str = "mock-chat-model",
) -> str:
    """
    模拟 Chat 大模型生成回答。

    当前用于打通 RAG 问答链路。
    后续替换为真实 OpenAI-compatible、Ollama 或 vLLM API。
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt 不能为空")

    source_marker = "[Source 1]"

    if source_marker not in prompt:
        return (
            "当前没有检索到足够的项目内容，"
            "暂时无法根据项目资料回答该问题。"
        )

    source_content = prompt.split(source_marker, maxsplit=1)[1]

    if "[Source 2]" in source_content:
        source_content = source_content.split(
            "[Source 2]",
            maxsplit=1,
        )[0]

    source_content = source_content.strip()

    return (
        f"根据当前检索结果，最相关的信息如下：\n\n"
        f"{source_content[:600]}\n\n"
        f"[Source 1]\n\n"
        f"当前回答由 {model_name} 生成。"
    )