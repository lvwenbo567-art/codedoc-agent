# Day27 Rerank 效果对比记录

## 目标

Day27 用 Rerank 精排改进 Day26 的 Hybrid Search 排序。

核心链路：

```text
Hybrid Search 召回 Top-K 候选
→ Reranker 对 query-document 文本对重新打分
→ 按 rerank_score 重排
→ 返回最终 Top-K
```

## 对比表

| Query | Hybrid 正确块排名 | Rerank 后排名 | 是否改善 | 备注 |
| --- | --- | --- | --- | --- |
| 项目如何启动 |  |  |  |  |
| EmbeddingClient 做什么 |  |  |  |  |
| `/index` 如何构建索引 |  |  |  |  |
| SQLite 有哪些表 |  |  |  |  |
| 如何处理模型超时 |  |  |  |  |
| 如何查询 chunks |  |  |  |  |
| Hybrid Search 是什么 |  |  |  |  |
| content_hash 有什么用 |  |  |  |  |
| batch_size 如何影响建库 |  |  |  |  |
| RAG 如何构建 prompt |  |  |  |  |

## 观察重点

- 正确 chunk 是否从较低排名被提升到 Top-3。
- 原本 Top-1 正确结果是否被错误降级。
- `retrieval_rank` 和 `rank` 是否体现精排前后的变化。
- `candidate_top_k` 太小时，正确 chunk 是否根本没有进入候选集。
- 真实 CrossEncoder 的延迟是否可以接受。

## 验收标准

- `/rerank_search` 能返回 `success=true`。
- 返回结果包含 `retrieval_rank` 和 `rerank_score`。
- `rank` 是精排后的新排名。
- mock rerank 单元测试稳定通过。
- 真实模型可通过 `app/rerank_probe.py` 单独验证。
