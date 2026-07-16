# 测试说明

这个文件不是 pytest 代码，而是用于测试文档检索效果的说明文档。

## Calculator 测试点

- `Calculator.add(1, 2)` 应该返回 `3`。
- `multiply(3, 4)` 应该返回 `12`。

## API 测试点

- `health_check()` 应该返回 `status=ok`。
- `create_document_api()` 应该返回 document_id、content 和 length。
- `list_chunks_api()` 应该返回指定文档的 chunks。

## 检索测试点

如果问题是“如何查询 chunks”，正确内容应该命中 `api.py` 中的 `list_chunks_api()` 或 `database.py` 中的 `list_chunks()`。
