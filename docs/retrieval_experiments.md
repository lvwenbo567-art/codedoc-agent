# CodeDoc 实验说明

本目录记录 CodeDoc Research Agent 当前已经落地的三类实验：

1. 检索策略消融实验；
2. Chunk 结构保留实验；
3. Agent 工具调用实验。

这些实验的目标不是“把指标硬做高”，而是让项目具备可复现、可分析、可面试讲清楚的评测闭环。

## 一、检索策略消融实验

默认评测集：

```text
data/evaluation/retrieval_experiment_cases.jsonl
```

当前包含 25 条 case，覆盖：

- 函数定位；
- 类和方法理解；
- API 查询；
- 数据库函数；
- 启动流程；
- README / 使用文档；
- 测试说明文档；
- 模块级概览。

对比方法：

- `bm25`
- `vector`
- `hybrid`
- `hybrid_rerank`
- `multi_query_rerank`

真实模型命令：

```powershell
python scripts\run_retrieval_experiments.py `
  --dataset data/evaluation/retrieval_experiment_cases.jsonl `
  --chunks-path outputs/test_project_chunks.json `
  --index-path outputs/test_project_vector_index_bge_m3.json `
  --output outputs/experiments/retrieval_experiment_report_bge_m3_25.json `
  --embedding-provider ollama `
  --embedding-model bge-m3 `
  --embedding-base-url http://localhost:11434 `
  --mock-dimension 1024 `
  --rerank-provider sentence_transformers `
  --rerank-model D:/models/bge-reranker-v2-m3 `
  --query-rewrite-provider mock
```

本次真实实验结果：

| 方法 | Hit@K | Recall@K | MRR | NDCG@K | 平均延迟(ms) | P95延迟(ms) |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.88 | 0.6933 | 0.7267 | 0.6395 | 1.40 | 1.52 |
| Vector | 1.00 | 0.8400 | 0.9500 | 0.8286 | 2565.90 | 2732.92 |
| Hybrid | 1.00 | 0.8533 | 0.9200 | 0.8225 | 2672.40 | 2725.58 |
| Hybrid + Rerank | 0.96 | 0.8067 | 0.8300 | 0.7383 | 6324.81 | 6202.12 |
| Multi-Query + Rerank | 0.96 | 0.8167 | 0.8300 | 0.7450 | 11270.22 | 11905.38 |

当前结论：

- 在 `test_project` 这种小型代码仓库上，`Vector` 和 `Hybrid` 表现最好；
- `BM25` 延迟极低，但语义召回不如真实向量；
- `Rerank` 在当前小评测集上没有带来提升，反而增加延迟；
- 这个结论说明项目已经具备“用数据分析检索策略”的能力，而不是只凭感觉调参。

## 二、Chunk 结构保留实验

默认评测集：

```text
data/evaluation/chunk_experiment_cases.jsonl
```

运行命令：

```powershell
python scripts\run_chunk_experiments.py
```

输出报告：

```text
outputs/experiments/chunk_experiment_report.json
```

本次结果：

```text
case_count = 14
chunk_count = 26
structure_preservation_rate = 1.0
average_candidate_count = 3.1429
```

这个实验验证的是：

- Python 函数是否被切成函数级 chunk；
- Python 类是否被切成类级 chunk；
- Python 方法是否带有 `qualified_name`；
- README / usage 文档是否被切成 document chunk；
- Chunk 结果是否保留了 `source_path`、`chunk_type`、`code_unit_type`、`symbol_name`、`qualified_name` 等元数据。

## 三、Agent 工具调用实验

默认评测集：

```text
data/evaluation/codedoc_agent_eval.jsonl
```

真实模型命令：

```powershell
$env:LANGCHAIN_CHAT_PROVIDER="openai_compatible"
$env:LANGCHAIN_CHAT_MODEL="qwen3.5:4b"
$env:LANGCHAIN_CHAT_BASE_URL="http://localhost:11434/v1"
$env:LANGCHAIN_CHAT_API_KEY="EMPTY"
$env:LANGCHAIN_CHAT_TIMEOUT_SECONDS="120"
$env:LANGCHAIN_CHAT_MAX_TOKENS="800"
$env:LANGCHAIN_CHAT_MAX_RETRIES="0"
$env:LANGCHAIN_CHAT_TEMPERATURE="0.1"
$env:LANGCHAIN_STRUCTURED_OUTPUT_METHOD="function_calling"

python scripts\run_agent_experiments.py `
  --dataset data/evaluation/codedoc_agent_eval.jsonl `
  --output outputs/experiments/agent_tool_call_experiment_report_qwen35_real.json `
  --project-root . `
  --chunks-path outputs/test_project_chunks.json `
  --index-path outputs/test_project_vector_index_bge_m3.json `
  --embedding-provider ollama `
  --embedding-model bge-m3 `
  --embedding-base-url http://localhost:11434 `
  --embedding-timeout-seconds 120 `
  --mock-dimension 1024 `
  --rerank-provider sentence_transformers `
  --rerank-model D:/models/bge-reranker-v2-m3 `
  --rerank-local-files-only `
  --recursion-limit 40
```

本次真实实验结果：

```text
total_cases = 10
passed_cases = 4
failed_cases = 6
task_success_rate = 0.4
tool_exact_match_rate = 0.3
average_tool_precision = 0.5667
average_tool_recall = 1.0
average_tool_f1 = 0.6667
first_tool_accuracy = 1.0
forbidden_tool_safety_rate = 0.9
completion_rate = 0.7
average_latency_ms = 35671.41
p95_latency_ms = 85958.13
```

当前结论：

- Agent 能够正确命中预期工具，`tool_recall = 1.0`；
- 第一次工具选择稳定，`first_tool_accuracy = 1.0`；
- 但工具精确率和任务成功率还有优化空间，说明模型会额外调用工具，部分回答关键术语覆盖不足；
- 这部分更适合作为“Agent 评测与问题发现闭环”来讲，而不是直接吹高指标。

## 简历表述建议

当前最稳的写法：

```text
构建覆盖代码定位、文档问答、配置查询和工具调用的 JSONL 评测集，对 BM25、向量检索、Hybrid Search、Rerank、Multi-Query 和 Agent 工具调用链路进行消融评估，统计 Recall@K、MRR、NDCG@K、工具选择准确率、任务成功率和 P95 延迟，并基于失败样本分析检索策略与 Agent 工具路由的优化方向。
```

不建议写：

```text
Rerank 显著提升检索效果。
```

因为当前真实实验结果不支持这个结论。
