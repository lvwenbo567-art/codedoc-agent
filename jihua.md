对，下面按**正常项目学习路线**讲，不单独强调异步。你当前计划就是围绕一个主项目：

```text
CodeDoc Research Agent
基于 Agentic RAG 的开源项目研究助手
```

目标是把它做成一个能写进简历、能演示、能面试讲清楚的大模型应用项目。

---

# 总体路线

当前计划按 6 周左右推进：

```text
第 1 周：Python 工程化 + RAG 数据准备层
第 2 周：FastAPI + SQLite 后端服务层
第 3 周：手写向量 RAG + /ask 问答闭环
第 4 周：RAG 增强 + Function Calling
第 5 周：LangChain + LangGraph Agentic RAG
第 6 周：MCP + 模型服务 + Docker + 项目包装
```

你现在大概在：

```text
第 2 周中段
正在做 FastAPI /eval 接口 + 统一 API 响应格式
```

---

# 第 1 周：Python 工程化 + RAG 数据准备层

这一周你已经基本完成。

## 主要干什么

先不急着接大模型，而是把 RAG 最底层的数据链路写出来。

完成内容包括：

```text
读取项目文件
解析 Python 代码结构
构建 chunks
保存 chunks.json
关键词检索
检索评估
```

对应模块：

```text
file_loader.py
code_parser.py
chunker.py
chunk_storage.py
retriever.py
search_service.py
retrieval_eval.py
eval_cli.py
config.py
logger.py
document_schema.py
```

## 学到了什么

这一周学的是 **RAG 的底层数据准备能力**。

你学到：

```text
文件怎么读取
代码结构怎么解析
chunk 是什么
chunk_size 和 overlap 有什么用
metadata 为什么重要
chunk_id / source_path / chunk_type 为什么要保存
关键词检索怎么做
HitRate / Recall / MRR 怎么评估检索效果
Python 项目为什么要拆模块、写测试、写日志
```

这一周结束后，你的项目已经能做到：

```text
输入项目目录
  ↓
读取文件
  ↓
切成 chunks
  ↓
保存 chunks
  ↓
检索 chunks
  ↓
评估检索效果
```

这是 RAG 项目的地基。

---

# 第 2 周：FastAPI + SQLite 后端服务层

你现在就在这一周。

## 主要干什么

把前面写的命令行工具，升级成真正的后端 API 服务。

你已经做或正在做：

```text
GET  /health
GET  /version
GET  /config
POST /scan
POST /search
POST /eval
统一 API 响应格式
统一异常处理
```

后面这一周还要补：

```text
SQLite 数据库存储
projects 表
files 表
chunks 表
GET /projects
GET /chunks
GET /chunks/{chunk_id}
```

## 学到了什么

这一周学的是 **大模型应用后端工程化**。

你要掌握：

```text
FastAPI 怎么写接口
GET 和 POST 怎么区分
Pydantic 请求体有什么用
API 层和 service 层为什么要分开
CLI 和 API 为什么要复用同一个 service
HTTPException 怎么处理
400 / 404 / 422 / 500 分别什么时候用
为什么要统一 success / data / error 响应格式
SQLite 在 RAG 系统里存什么
后端接口怎么写 pytest 测试
```

这一周结束后，你的项目会从：

```text
本地命令行工具
```

升级成：

```text
有 API、有测试、有异常处理、有数据库的 RAG 后端服务
```

这是秋招项目里非常重要的一步，因为它体现你的后端工程能力。

---

# 第 3 周：手写向量 RAG + `/ask` 问答闭环

这一周开始进入真正的大模型 RAG 问答。

## 主要干什么

从关键词检索升级到向量检索，并做出问答接口。

要做的模块：

```text
embedding_client.py
vector_store.py
index_service.py
prompt_builder.py
citation_builder.py
rag_service.py
```

要做的接口：

```text
POST /index
POST /vector_search
POST /ask
```

核心流程是：

```text
chunks
  ↓
生成 embedding
  ↓
建立向量索引
  ↓
用户问题生成 embedding
  ↓
向量相似度检索
  ↓
拼接 prompt
  ↓
调用 LLM
  ↓
生成答案
  ↓
返回引用来源
```

## 学到了什么

这一周学的是 **RAG 核心原理**。

你要掌握：

```text
Embedding 是什么
query embedding 和 document embedding 有什么区别
cosine similarity 是什么
向量检索怎么实现
向量库为什么要保存 metadata
RAG prompt 怎么拼
citation 引用溯源怎么做
/ask 接口背后的完整链路是什么
Mock LLM 后续怎么替换成真实模型
```

这一周结束后，你项目就有了基础 RAG 问答能力。

也就是用户可以问：

```text
这个项目的 main 函数在哪里？
这个项目怎么启动？
这个模块是干什么的？
```

系统可以检索相关 chunks，并生成回答。

---

# 第 4 周：RAG 增强 + Function Calling

这一周分两部分。

---

## 第一部分：RAG 增强

## 主要干什么

让检索效果更好。

要做：

```text
Hybrid Search
Rerank
Query Rewrite
检索评估对比
```

## 学到了什么

你要掌握真实面试里经常问的问题：

```text
为什么只靠向量检索不够？
关键词检索和向量检索有什么区别？
Hybrid Search 怎么做？
Rerank 解决什么问题？
Rerank 放在哪一层？
Query Rewrite 为什么能提升召回？
检索效果差怎么排查？
怎么用 Recall / MRR 对比优化前后效果？
```

这一部分会让你的项目从“能检索”升级到“会优化检索”。

---

## 第二部分：Function Calling

这是你已经决定正式加入项目的核心模块。

## 主要干什么

把 CodeDoc 里的能力封装成工具，让模型可以选择工具调用。

要做的模块：

```text
tools.py
tool_schema.py
tool_registry.py
function_calling_service.py
```

要做的接口：

```text
GET  /tools
POST /tool-call
```

工具包括：

```text
scan_project
search_chunks
get_chunk_detail
evaluate_retrieval
ask_rag
parse_python_structure
```

流程是：

```text
用户自然语言问题
  ↓
模型判断该调用哪个工具
  ↓
生成结构化参数
  ↓
后端执行工具
  ↓
返回工具结果
  ↓
模型组织最终回答
```

## 学到了什么

这一周你要掌握：

```text
Function Calling 是什么
工具函数和普通 service 函数有什么区别
tool schema 为什么重要
tool registry 是什么
模型怎么选择工具
工具参数为什么要结构化
工具调用失败怎么办
Function Calling 和普通 RAG 有什么区别
Function Calling 和 Agent 有什么关系
```

这一周结束后，你的项目就开始具备 Agent 雏形。

---

# 第 5 周：LangChain + LangGraph Agentic RAG

这一周做两个重要框架：LangChain 和 LangGraph。

---

## 第一部分：LangChain RAG

## 主要干什么

用 LangChain 重构一版 RAG，但不删除手写版本。

要做：

```text
langchain_loader.py
langchain_retriever.py
langchain_prompt.py
langchain_rag.py
POST /langchain/ask
```

## 学到了什么

你要掌握 LangChain 的核心组件：

```text
Document
DocumentLoader
TextSplitter
Retriever
PromptTemplate
Runnable
LCEL
Chain
```

重点是能讲清楚它和你手写模块的对应关系：

```text
DocumentLoader ≈ file_loader.py
TextSplitter ≈ chunker.py
Retriever ≈ retriever.py / vector_store.py
PromptTemplate ≈ prompt_builder.py
Runnable ≈ rag_service.py 的流程编排
```

这部分面试表达很重要：

```text
我不是一开始就直接套 LangChain，而是先手写 RAG 底层链路，再用 LangChain 重构。这样既理解原理，也会使用主流框架。
```

---

## 第二部分：LangGraph Agentic RAG

## 主要干什么

把普通 RAG 升级成多步骤 Agent。

要做：

```text
agent_state.py
agent_graph.py
nodes/
tools/
POST /agent/ask
```

Agent 节点大概包括：

```text
classify_question
rewrite_query
retrieve_code
retrieve_docs
select_tool
call_tool
generate_answer
check_citation
```

也就是不同问题走不同流程：

```text
代码问题 → 优先检索 .py chunks
文档问题 → 优先检索 README / md chunks
复杂问题 → 多路检索后综合回答
```

## 学到了什么

这一周你要掌握：

```text
LangGraph 和 LangChain 有什么区别
State 是什么
Node 是什么
Edge 是什么
Conditional Edge 怎么用
为什么复杂 RAG 需要多节点工作流
代码问题和文档问题为什么走不同检索路径
Agent 状态怎么维护
工具调用失败怎么兜底
Agentic RAG 和普通 RAG 有什么区别
```

这一周结束后，你的项目就从普通 RAG 升级成 Agentic RAG。

---

# 第 6 周：MCP + 模型服务 + Docker + 项目包装

这一周做高级扩展和最终可投递包装。

---

## 第一部分：MCP Server

MCP 是你项目里的高级加分模块。

## 主要干什么

把 CodeDoc 的能力暴露成 MCP Server。

暴露三类能力：

```text
Tools
Resources
Prompts
```

Tools：

```text
scan_project
search_chunks
get_chunk_detail
evaluate_retrieval
ask_rag
```

Resources：

```text
codedoc://projects
codedoc://chunks/{chunk_id}
codedoc://files/{file_path}
```

Prompts：

```text
project_summary_prompt
code_explain_prompt
rag_answer_prompt
```

## 学到了什么

你要掌握：

```text
MCP 是什么
MCP Server 是什么
tools / resources / prompts 分别是什么
MCP 和 Function Calling 有什么区别
为什么 MCP 适合把 CodeDoc 暴露给外部 AI 客户端
```

面试时可以这样讲：

```text
Function Calling 是系统内部让模型调用工具；
MCP 是把这些工具、资源和 Prompt 标准化暴露给外部 AI 客户端。
```

---

## 第二部分：模型服务

## 主要干什么

把 Mock LLM 替换成真实模型服务。

要学习和接入：

```text
OpenAI-compatible API
Ollama
vLLM
流式输出
```

要做的接口：

```text
POST /ask/stream
POST /agent/ask/stream
```

## 学到了什么

你要掌握：

```text
OpenAI-compatible API 是什么
Ollama 适合什么场景
vLLM 适合什么场景
本地模型服务怎么接入后端
为什么大模型应用需要流式输出
模型响应慢怎么排查
模型服务失败怎么兜底
```

---

## 第三部分：Docker + 项目包装

## 主要干什么

把项目变成可展示、可部署、可投递的版本。

要做：

```text
Dockerfile
docker-compose
requirements.txt
.env.example
README 完善
架构图
API 文档
演示流程
简历项目描述
面试问答
```

## 学到了什么

你要掌握：

```text
Dockerfile 怎么写
docker-compose 解决什么问题
.env 为什么不能提交真实密钥
README 怎么写才像真实项目
项目架构图怎么讲
API 文档怎么展示
项目演示怎么准备
简历项目怎么包装
面试怎么讲项目亮点和难点
```

这一周结束后，你的项目就进入秋招可展示状态。

---

# 算法线

算法每天 2 道，不重复。

整体路线：

```text
第 1 周：哈希表
第 2 周：双指针 + 滑动窗口
第 3 周：栈和队列
第 4 周：二叉树基础
第 5 周：回溯 + 动态规划入门
第 6 周：面试高频综合复习
```

你目前已经做过或安排过：

```text
哈希表
双指针
滑动窗口入门
```

当前已计入的题：

```text
1 两数之和
217 存在重复元素
242 有效的字母异位词
383 赎金信
349 两个数组的交集
350 两个数组的交集 II
219 存在重复元素 II
205 同构字符串
290 单词规律
202 快乐数
283 移动零
26 删除有序数组中的重复项
27 移除元素
344 反转字符串
977 有序数组的平方
125 验证回文串
167 两数之和 II
11 盛最多水的容器
345 反转字符串中的元音字母
392 判断子序列
3 无重复字符的最长子串
209 长度最小的子数组
438 找到字符串中所有字母异位词
567 字符串的排列
```

后面继续：

```text
20 有效的括号
232 用栈实现队列
225 用队列实现栈
150 逆波兰表达式求值
104 二叉树最大深度
226 翻转二叉树
101 对称二叉树
102 二叉树层序遍历
```

---

# 八股线

八股以后按真实面试问题来，不出太空泛的问题。

每周重点：

```text
第 1 周：RAG 基础
第 2 周：FastAPI 后端工程
第 3 周：Embedding / 向量检索 / RAG 问答
第 4 周：Rerank / Query Rewrite / Function Calling
第 5 周：LangChain / LangGraph / Agent
第 6 周：MCP / 模型服务 / Docker / 项目包装
```

后面的问题会偏这种：

```text
RAG 检索效果差怎么排查？
Embedding 模型怎么选？
为什么要 Rerank？
Rerank 放在哪一层？
FastAPI 接口超时怎么处理？
Function Calling 的 tool schema 怎么设计？
工具调用失败怎么办？
LangGraph 比普通 Chain 好在哪里？
MCP 和 Function Calling 有什么区别？
Ollama 和 vLLM 分别适合什么场景？
Docker 部署后模型连不上怎么排查？
```

这些都是更接近真实大模型应用开发面试的问题。

---

# 最终你会学到什么

完整学完后，你会形成一套比较完整的大模型应用开发能力：

```text
Python 工程化
FastAPI 后端开发
SQLite 数据存储
RAG 数据处理
关键词检索
向量检索
Embedding
RAG 评估
Hybrid Search
Rerank
Query Rewrite
Prompt 构造
引用溯源
Function Calling
LangChain
LangGraph
MCP
OpenAI-compatible API
Ollama / vLLM
流式输出
Docker 部署
项目包装和面试表达
```

---

# 最终项目形态

最终 CodeDoc Research Agent 是：

```text
一个面向代码项目理解的 Agentic RAG 系统
```

它能做到：

```text
扫描项目
解析代码
构建 chunks
保存元数据
关键词检索
向量检索
检索评估
生成答案
返回引用
Function Calling 工具调用
LangGraph 多步骤 Agent
MCP Server 暴露工具和资源
FastAPI 服务化
Docker 部署
```

---

# 一句话总结

你当前的计划就是：

```text
先手写 RAG 底层链路，
再用 FastAPI 和 SQLite 做成后端服务，
然后接 Embedding 和向量检索形成 /ask 问答闭环，
再加入 Hybrid Search、Rerank、Query Rewrite 和 Function Calling，
之后用 LangChain 和 LangGraph 升级为 Agentic RAG，
最后补 MCP、模型服务、Docker 和项目包装。
```

最终目标是：

```text
能投大模型应用开发、RAG 工程、AI Agent 应用开发、Python AI 后端这些方向。
```
