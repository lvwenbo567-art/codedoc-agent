# CodeDoc 评测与回归说明

本项目将检索、Agent 行为和 Bad Case 回归分开评估。这样可以区分“没有找到证据”“Agent 选错工具”和“模型没有根据证据回答”三类问题。

## 1. 检索 Benchmark

### 数据集与标注

数据集：`data/evaluation/retrieval_experiment_cases.jsonl`

- 样例数：100；
- 范围：函数定位、类/方法理解、API、配置、数据库、启动流程、README/使用文档、模块概览；
- 每条样例包含人工标注的 `expected_chunk_ids`；
- 固定 `Top-K = 5`，所有策略在相同 chunks、相同相关 chunk 标注上比较。

指标含义：

- **Hit@5**：Top 5 中是否至少出现一个相关 chunk；
- **Recall@5**：Top 5 召回的相关 chunk 占全部相关 chunk 的比例；
- **MRR**：第一个相关结果的排名质量；
- **NDCG@5**：兼顾多个相关结果位置的排序质量；
- **Average/P95 Latency**：端到端检索耗时。

### 对比结果

报告：`outputs/experiments/retrieval_experiment_report_bge_m3_100.json`

| 策略 | Hit@5 | Recall@5 | MRR | NDCG@5 | Avg latency(ms) | P95(ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.9800 | 0.8825 | 0.8123 | 0.7855 | 1.00 | 1.48 |
| Dense Vector | 0.9500 | 0.8475 | 0.7857 | 0.7551 | 2455.94 | 2503.58 |
| **Hybrid** | **0.9900** | **0.8992** | **0.8382** | **0.8123** | 2402.45 | 2458.98 |
| Hybrid + Rerank | 0.9400 | 0.8433 | 0.6715 | 0.6775 | 6029.10 | 5675.90 |
| Multi-Query + Rerank | 0.9400 | 0.8450 | 0.6693 | 0.6762 | 10533.33 | 10901.98 |

### 结论与边界

在当前小型 `test_project` 上，Hybrid 在召回与排序指标上表现最好；BM25 延迟最低；本次 Reranker 和 Multi-Query 组合未提升指标且增加延迟。该结论只适用于当前数据、模型与参数组合，不应泛化为“Reranker 无效”。

真实模型运行示例：

```powershell
python scripts/run_retrieval_experiments.py `
  --dataset data/evaluation/retrieval_experiment_cases.jsonl `
  --chunks-path outputs/test_project_chunks.json `
  --index-path outputs/test_project_vector_index_bge_m3.json `
  --output outputs/experiments/retrieval_experiment_report_bge_m3_100.json `
  --embedding-provider ollama `
  --embedding-model bge-m3 `
  --embedding-base-url http://localhost:11434 `
  --rerank-provider sentence_transformers `
  --rerank-model D:/models/bge-reranker-v2-m3 `
  --rerank-local-files-only
```

## 2. Chunk 结构保留实验

数据集：`data/evaluation/chunk_experiment_cases.jsonl`。该实验校验 Python 函数/类/方法的 AST 切分，以及代码和文档 chunk 是否保留 `source_path`、`chunk_type`、`symbol_name`、`qualified_name` 和行号等定位元数据。

运行：

```powershell
python scripts/run_chunk_experiments.py
```

报告：`outputs/experiments/chunk_experiment_report_latest.json`。

## 3. Agent 任务 Benchmark

数据集：`data/evaluation/codedoc_agent_eval.jsonl`

- 样例数：50；
- 覆盖：符号定位、源码范围读取、项目结构、文档问答、代码流程、测试执行与安全边界；
- 每条样例定义预期工具、预期首工具、回答关键术语、禁止工具和可接受停止原因；
- 使用真实支持 Tool Calling 的模型运行，**新增数据集后必须重新执行，不能沿用旧 10 条报告作为结论**。

指标：

- 任务成功率、完成率；
- 工具精确率、工具召回率、工具 F1、精确工具链匹配率；
- 首工具准确率、禁止工具安全率；
- 答案关键术语覆盖率；
- 平均与 P95 延迟。

运行命令见项目根目录 [README](../README.md#评测与实验)。输出报告内保留每个 case 的工具调用、答案、执行步骤和错误信息，可作为 Trace 分析输入。

### 当前真实基线

报告：`outputs/experiments/agent_eval_report_qwen35_50_round3_after_key_badcase_fix.json`

模型为本地 `qwen3.5:4b`，使用真实 Tool Calling、`bge-m3` 索引和 50 条任务集：

| 任务成功率 | 精确工具链匹配率 | 首工具准确率 | 工具 F1 | 禁止工具安全率 | 答案关键词覆盖率 | 平均延迟 | P95 延迟 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 86.00% | 66.00% | 88.00% | 81.33% | 100.00% | 97.00% | 20.75 s | 66.26 s |

该结果表明工具选择与安全拒答总体稳定，但复杂流程与文档结合代码场景仍可能产生额外调用或触及模型步数限制。报告保留每条任务的 Trace、工具调用和失败原因，用于后续回归排查。

## 4. Bad Case 与回归

项目提供两类 Bad Case：

- `data/evaluation/agent_bad_case_regression.jsonl`：人工维护的高价值回归集，涵盖显式源码读取、文档证据、函数定位、跨函数流程和项目外问题拒答；
- 从任意 Agent 报告中用 `scripts/export_bad_cases.py` 自动导出的当次失败样本。

推荐闭环：

```text
真实 Agent Eval → 查看报告中的 Trace / failure_reasons
                → 分类（检索、排序、路由、工具、生成）
                → 修复并执行人工回归集 + 本次自动导出的失败集
```

自动导出只用于本次回归，不覆盖人工维护的回归集。命令见根目录 README。

最近一次人工回归报告：`outputs/experiments/agent_bad_case_regression_report_qwen35_round8.json`。8 条高价值回归任务全部通过，首工具准确率、禁止工具安全率与答案关键词覆盖率均为 100%，工具 F1 为 95.83%。回归集覆盖显式源码读取、文档+入口代码、符号调用链、结构问答和安全拒答；它用于防止已修复问题再次出现，不替代全量 Agent Benchmark。
