# SecOps-Copilot — 云原生安全运维多 Agent 系统

> 探真科技 AI 工程师面试项目

## 项目简介

基于 **多 Agent 协作 + 异步并发** 的安全运维 AI 系统，用于 K8s 安全告警研判、CVE 漏洞分析、安全知识库检索。

**架构**：Coordinator 智能路由 → 专业 Agent 异步并行 → Report Agent 汇总输出

```
用户查询
  │
  ▼
Coordinator（异步路由）
  ├── AlertAnalystAgent    ─┐
  ├── CVEResearcherAgent   ─┼── asyncio.gather（并行）
  └── KnowledgeAgent       ─┘
  │
  ▼
ReportAgent（串行，依赖前面结果）
  │
  ▼
结构化安全运营报告
```

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 设置环境变量
```bash
export AGNES_API_KEY=your_key_here
```

### 运行方式

**1. CLI 直接调用**
```bash
python agent_main.py "帮我分析告警ALERT-0815"
python agent_main.py "CVE-2024-21762漏洞详情"
python agent_main.py "K8s容器逃逸防御措施有哪些"
```

**2. FastAPI 服务**
```bash
python api.py
# 然后测试:
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"帮我分析告警ALERT-0815"}'
```

**3. 数据流水线模拟**
```bash
# 单条处理
python data_pipeline.py --mode alert --alert-id ALERT-0815
python data_pipeline.py --mode cve --cve-id CVE-2024-21762

# 批量处理
python data_pipeline.py --mode batch --batch-alerts ALERT-0815 ALERT-0923
```

**4. 运行评估**
```bash
python eval.py
```

## 系统架构

### Agent 协作架构

| Agent | 职责 | 并发方式 |
|-------|------|---------|
| Coordinator | 路由分发 + 结果编排 | 同步入口 |
| AlertAnalystAgent | 告警研判（等级/影响/攻击路径） | 并行 |
| CVEResearcherAgent | CVE 研究（漏洞分析/修复方案） | 并行 |
| KnowledgeAgent | RAG 知识检索（FAISS + BGE） | 并行 |
| ReportAgent | 结构化报告生成 | 串行（最后） |

### 关键技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| LLM | agnes-2.0-flash（OpenAI 兼容 API） | 云端 API，非本地部署 |
| Embedding | BAAI/bge-small-zh | 中文优化 |
| 向量检索 | FAISS IndexFlatIP | 余弦相似度，阈值 0.75 |
| 并发模型 | asyncio + asyncio.gather | 独立 Agent 并行执行 |
| API | FastAPI + Uvicorn | 5 个 REST 接口 |
| 日志 | logging + 结构化输出 | 每次调用记录耗时/参数/结果 |

### 异步并发效果

| 查询 | 串行耗时 | 异步并发耗时 | 提升 |
|------|---------|-------------|------|
| ALERT-0815 分析 | ~47s | ~32s | 32% |
| CVE-2024-21762 | ~27s | ~22s | 19% |
| 知识检索 | ~24s | ~24s | — |

> AlertAnalyst + KnowledgeAgent 并行执行，ReportAgent 等待两者完成后汇总。

## 评估数据

```
RAG 检索召回率:   7/7 = 100%
路由准确率:       4/4 = 100%
端到端可用率:     4/4 = 100%
幻觉抑制率:       7/8 = 88%
综合评分:         92/100
```

## 文件结构

```
Agnes/
├── agent_main.py          # 多 Agent 核心（asyncio 并发）
├── api.py                 # FastAPI 服务（5 个接口）
├── data_pipeline.py       # 数据流水线模拟（新增）
├── eval.py                # 评估脚本
├── process_knowledge.py   # 知识文档切片
├── requirements.txt
├── alerts.json            # 12 条安全告警
├── cve_db.json            # 8 条 CVE 漏洞
├── knowledge_chunks.json  # 12 条知识块
├── hallucination_test_case.json  # 8 个幻觉测试
├── docs/                  # 安全知识文档（4份）
├── faiss/                 # FAISS 向量库（运行时生成）
└── logs/                  # 结构化运行日志
```

## 面试话术

> "这是一个面向云原生安全运营的多 Agent 协作系统。核心架构是 Coordinator 路由 + 4 个专业 Agent 异步并发 + Report Agent 汇总。
>
> 异步并发将 AlertAnalyst 和 KnowledgeAgent 并行执行，相比串行节省了约 30% 的响应时间。
>
> 我还设计了一个数据流水线模块，模拟 SOC 平台接收告警/CVE 数据并自动研判的流程，支持单条和批量处理。
>
> 评估数据显示：RAG 召回率 100%，路由准确率 100%，端到端可用率 100%，幻觉抑制率 88%。
>
> 技术上全部手写，没有依赖 LangChain/LlamaIndex，展示了对 Agent 原理和异步编程的理解。"

## 隐私说明

API Key 通过环境变量 `AGNES_API_KEY` 配置，不硬编码在代码中。`.env` 文件已被 `.gitignore` 排除。
