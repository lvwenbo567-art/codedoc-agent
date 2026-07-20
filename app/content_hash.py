import hashlib


def normalize_content(content: str) -> str:
    """
    统一换行符，避免 Windows 和 Linux 换行差异
    导致同一份内容产生不同哈希。
    """
    if not isinstance(content, str):
        raise TypeError("content 必须是字符串")

    return (
        content
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def compute_content_hash(content: str) -> str:
    """
    计算文本内容的 SHA-256 哈希。
    """
    normalized_content = normalize_content(content)

    return hashlib.sha256(
        normalized_content.encode("utf-8")
    ).hexdigest()