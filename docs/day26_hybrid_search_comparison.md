# Day26 Hybrid Search 对比记录

## 目标

Day26 的目标是把原来的单一路径检索升级为混合检索：

- 关键词检索：适合命中函数名、类名、文件名、精确术语。
- 向量检索：适合理解语义相近但字面不完全一致的问题。
- 混合检索：把两个通道的结果融合，提升召回稳定性。

## 默认权重

```text
keyword_weight = 0.4
vector_weight = 0.6
```

如果问题里包含非常明确的函数名、接口名、错误码，可以适当提高关键词权重。
如果问题更偏自然语言描述，可以适当提高向量权重。

## 对比建议

同一个问题建议分别测试：

1. `POST /search`
2. `POST /vector_search`
3. `POST /hybrid_search`

观察：

- Top-1 是否命中正确文件；
- Top-K 是否包含应该出现的 chunk；
- 分数排序是否符合直觉；
- `matched_by` 是否同时包含 `keyword` 和 `vector`。

## 验收标准

- `/hybrid_search` 能返回 `success=true`。
- 返回结果包含 `final_score`。
- 同一个 `chunk_id` 不重复出现。
- 能看到 `keyword_score`、`vector_score`、`matched_by`，方便解释每个结果为什么被召回。
