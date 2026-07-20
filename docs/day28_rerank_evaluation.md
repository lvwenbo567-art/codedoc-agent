# Day28 Rerank 稳定性与评估报告

## 今日目标

Day28 不新增复杂大功能，重点补齐 Rerank 工程稳定性和效果评估：

- Rerank 服务异常时降级为 Hybrid Search。
- 记录 `rerank_duration_ms`。
- 返回 `rerank_applied`、`degraded`、`degrade_reason`。
- 实现 Hit@K 和 MRR。
- 建立 10 条小型 Rerank 评测集。
- 对比 Hybrid Search 与 Hybrid + Rerank。

## 稳定性策略

只捕获 `RerankServiceError` 并降级：

```text
Hybrid Search candidates
→ Rerank
→ 如果 RerankServiceError
→ 使用 Hybrid 原始排序作为 fallback
```

不会吞掉以下错误：

- query 为空；
- final_top_k 非法；
- 候选缺少 content；
- Rerank 返回分数数量与候选数量不一致；
- 参数或数据结构错误。

这些错误应该暴露出来，方便开发阶段修 bug。

## 返回字段说明

```text
rerank_applied：是否真正完成 Rerank
degraded：是否发生降级
degrade_reason：降级原因
rerank_duration_ms：Rerank 阶段耗时
retrieval_mode：hybrid_rerank 或 hybrid_fallback
```

## 指标说明

```text
Hit@K：Top-K 中是否命中任意正确 chunk
MRR：第一个正确 chunk 排名的倒数
```

例子：

```text
正确 chunk 排第 1：RR = 1
正确 chunk 排第 2：RR = 0.5
正确 chunk 排第 4：RR = 0.25
完全未命中：RR = 0
```

## 实验表模板

| 实验 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均延迟 |
| --- | --- | --- | --- | --- | --- |
| Hybrid |  |  |  |  |  |
| Rerank 5→3 |  |  |  |  |  |
| Rerank 10→5 |  |  |  |  |  |
| Rerank 20→5 |  |  |  |  |  |

## 重要结论边界

Rerank 只能重新排序第一阶段已经召回的候选。

如果正确 chunk 没有进入 Hybrid Search 候选集，Rerank 无法凭空找回它。因此评估时要同时观察：

- candidate_top_k 是否足够；
- 正确 chunk 是否进入候选；
- rerank 是否把正确 chunk 提升；
- rerank 是否错误降低了原本正确的结果。
