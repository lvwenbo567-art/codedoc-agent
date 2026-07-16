# Day24 Embedding 检索对比记录

本文件用于记录同一份 `outputs/chunks.json` 在不同 Embedding 索引下的检索效果。

## 索引文件

```text
Mock 索引：outputs/vector_index_mock.json
真实索引：outputs/vector_index_real.json
```

## 对比方法

1. 使用同一份 `outputs/chunks.json`。
2. 分别构建 Mock 索引和真实 Embedding 索引。
3. 对同一组 query 分别执行 `/vector_search`。
4. 记录 Top-3 结果，观察正确 chunk 是否出现、排名是否提升。

## 对比表

| Query | Mock Top-1 | 真实模型 Top-1 | 正确结果是否出现 | 备注 |
| --- | --- | --- | --- | --- |
| 这个项目如何启动？ | 待填写 | 待填写 | 待填写 | 观察中文问题能否命中 README 或启动说明 |
| scan_project 函数在哪里定义？ | 待填写 | 待填写 | 待填写 | 观察函数名检索能力 |
| 项目如何查询 chunks？ | 待填写 | 待填写 | 待填写 | 观察 API/数据库查询相关 chunk |
| EmbeddingClient 的作用是什么？ | 待填写 | 待填写 | 待填写 | 观察类职责说明是否命中 |
| 项目使用了哪些数据库表？ | 待填写 | 待填写 | 待填写 | 观察数据库 schema 相关信息 |

## 初步观察

```text
待填写：
- 真实模型是否更能处理中文自然语言问题？
- Mock 是否更依赖关键词字面匹配？
- 代码符号检索是否仍需要关键词、BM25 或 rerank 辅助？
- 错误结果主要来自切块、Embedding，还是 query 表达？
```
