# 测试项目使用说明

这个目录用于模拟一个小型 Python 后端项目，方便测试 CodeDoc Research Agent 的扫描、切块、索引和问答能力。

## 启动方式

在项目根目录执行：

```bash
python main.py
```

启动流程会创建 `Calculator`，调用 `add()` 计算加法，然后调用 `multiply()` 计算乘法。

## API 说明

`api.py` 中包含三个主要接口函数：

- `health_check()`：返回服务健康状态。
- `create_document_api()`：创建文档并保存内容。
- `list_chunks_api()`：根据 document_id 查询 chunks。

## 数据库说明

`database.py` 中模拟了三张表：

- `projects`
- `documents`
- `chunks`

`init_database()` 会返回数据库路径和表列表。

## 检索说明

`search.py` 提供了一个简单关键词检索流程：

1. `keyword_score()` 计算 query 和文本的关键词匹配次数。
2. `search_documents()` 对文档进行打分。
3. 根据 score 从高到低排序。
4. 返回 Top-K 文档。
