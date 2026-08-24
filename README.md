# SecOps-Copilot — 云原生安全运维 AI 助手

> 探真科技 AI 工程师面试项目

## 项目简介

基于 RAG（检索增强生成）+ Function Calling 的安全运维 Agent，
用于 K8s 安全告警研判、CVE 漏洞查询、安全知识库检索。

**架构**：BGE-small-zh Embedding → FAISS 向量检索 → LLM Agent 多轮工具调用 → 结构化输出

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 首次运行（构建向量库）
```bash
python agent_main.py
```
首次运行会自动：
- 加载 BGE-small-zh 嵌入模型
- 将 `knowledge_chunks.json` 的 12 条知识块编码为向量
- 写入 `faiss/index_faiss.index`（持久化，后续启动直接加载）

### 3. 运行 Agent
```bash
python agent_main.py "帮我分析告警ALERT-0815"
python agent_main.py "CVE-2024-21762 的修复方案是什么"
python agent_main.py "K8s 容器逃逸的防御措施有哪些"
```

### 4. 启动 API 服务
```bash
python api.py
```
然后用 curl 测试：
```bash
# 健康检查
curl http://localhost:8000/api/health

# Agent 对话
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"帮我分析告警ALERT-0815，给出风险研判以及处置建议。"}'

# 查告警
curl http://localhost:8000/api/v1/alerts/ALERT-0923

# 查 CVE
curl http://localhost:8000/api/v1/cve/CVE-2024-21762

# 知识库检索（调试用）
curl "http://localhost:8000/api/v1/knowledge/search?query=容器逃逸防御"
```

### 5. 运行评估
```bash
python eval.py
```
输出检索召回率和幻觉抑制率。

## 文件结构

```
Agnes/
├── agent_main.py          # Agent 主程序（CLI 入口）
├── api.py                 # FastAPI 服务（HTTP 接口）
├── eval.py                # 评估脚本（召回率 + 幻觉测试）
├── process_knowledge.py   # 知识文档切片工具
├── requirements.txt       # Python 依赖
├── alerts.json            # 12 条安全告警数据
├── cve_db.json            # 8 条 CVE 漏洞数据
├── knowledge_chunks.json  # 12 条知识库切片
├── hallucination_test_case.json  # 8 个幻觉测试用例
├── docs/                  # 安全知识文档
│   ├── cis_benchmarks.md
│   ├── cloud_native_security.md
│   ├── incident_response.md
│   └── k8s_attack_patterns.md
├── faiss/                 # FAISS 向量库（运行时生成）
│   ├── index_faiss.index
│   ├── embeddings.npy
│   └── chunk_texts.json
└── logs/                  # 运行日志
    └── agent.log
```

## 核心设计

### RAG 检索
- **Embedding**: BAAI/bge-small-zh（中文优化）
- **向量库**: FAISS IndexFlatIP（内积 = 余弦相似度）
- **阈值过滤**: 相似度 < 0.75 的结果不返回，防止噪声干扰 LLM
- **持久化**: 首次构建后保存为 .index 文件，后续启动直接加载

### Agent 循环
- **最大轮次**: 8 轮
- **防死循环**: 检测重复工具调用，超过 2 次相同参数自动跳过
- **异常处理**: 工具执行失败返回错误信息，LLM 可重试
- **结构化日志**: 每次调用记录时间、工具名、参数、返回值、耗时

### 防幻觉机制
1. **检索兜底**: 召回分数低于阈值时，工具返回"未查询到"而非空字符串
2. **Prompt 约束**: 系统提示明确禁止编造
3. **评估验证**: `eval.py` 包含 8 个幻觉测试用例

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | agnes-2.0-flash（OpenAI 兼容 API） |
| Embedding | BAAI/bge-small-zh |
| 向量检索 | FAISS (IndexFlatIP) |
| Agent 框架 | 手写 Function Calling 循环 |
| API 框架 | FastAPI + Uvicorn |
| 评估 | 自定义召回率 + 幻觉测试 |

## 面试亮点

1. **手写 Agent 循环** — 不依赖 LangChain，展示对 Function Calling 原理的理解
2. **FAISS 持久化** — 向量库落地磁盘，支持生产环境快速启动
3. **防幻觉双重机制** — 检索阈值 + Prompt 约束，有评估数据支撑
4. **真实数据驱动** — 12 条告警 + 8 条 CVE，不再是 mock 数据
5. **工程化完整** — API 接口 + 结构化日志 + 自动化评估
