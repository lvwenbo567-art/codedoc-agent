from context_engineering.token_counter import ApproximateTokenCounter, CharacterTokenCounter


def test_token_counters_cover_chinese_and_ascii() -> None:
    counter = ApproximateTokenCounter()
    assert counter.count_text("") == 0
    assert counter.count_text("你好") == 2
    assert counter.count_text("abcd") == 1
    assert counter.count_text("你abcd") == 2
    assert CharacterTokenCounter().count_text("abc") == 3
