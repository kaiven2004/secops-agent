"""
SecOps-Copilot API — 多 Agent 协作版 FastAPI 服务
启动: python api.py
"""
import os, sys, json, time
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import re

# 导入多 Agent 核心
from agent_main import (
    load_infra, Coordinator, AgentContext, AgentResult,
    AlertAnalystAgent, CVEResearcherAgent, KnowledgeAgent, ReportAgent,
    FAISS_INDEX_PATH, ALERTS_PATH, CVE_DB_PATH,
)

BASE_DIR  = Path(__file__).parent
CACHE_DIR = Path(r"D:\RAG1\huggingface_cache")
os.environ["HF_HOME"]            = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 全局状态
_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] 正在初始化多 Agent 基础设施...")
    index, texts, tokenizer, model, alerts, cve_db, client = load_infra()
    _state.update(dict(
        coordinator=Coordinator(client, alerts, cve_db, index, texts, tokenizer, model),
        alerts=alerts, cve=cve_db, chunks=len(texts),
    ))
    print(f"[API] 就绪 | 告警:{len(alerts)}条 CVE:{len(cve_db)}条 知识块:{len(texts)}条")
    yield
    print("[API] 服务已关闭")


app = FastAPI(
    title="SecOps-Copilot Multi-Agent API",
    description="探真科技云原生安全运维 AI 助手 — 多 Agent 协作服务",
    version="2.0.0",
    lifespan=lifespan,
)


class AgentQuery(BaseModel):
    query: str
    # 可选：指定要调用的 Agent
    agents: list[str] = None   # ["AlertAnalyst", "Knowledge", "Report"] 或 None（自动路由）


class AgentInfo(BaseModel):
    name: str
    description: str


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0",
            "alerts":    len(_state.get("alerts", {})),
            "cve":       len(_state.get("cve", {})),
            "chunks":    _state.get("chunks", 0)}


@app.get("/api/v1/agents")
async def list_agents():
    """列出可用的 Agent 及其职责。"""
    agents = [
        {"name": "AlertAnalystAgent",  "description": "告警研判：分析告警等级、影响范围、攻击路径"},
        {"name": "CVEResearcherAgent", "description": "CVE 研究：漏洞分析、影响版本、修复方案"},
        {"name": "KnowledgeAgent",     "description": "知识检索：RAG 检索安全知识库，返回相关文档"},
        {"name": "ReportAgent",        "description": "报告生成：汇总各 Agent 结果，输出结构化安全报告"},
        {"name": "Coordinator",        "description": "协调器：智能路由查询到对应 Agent，编排执行流程"},
    ]
    return {"agents": agents, "total": len(agents)}


@app.post("/api/v1/agent/query", response_model=dict)
async def agent_query(body: AgentQuery):
    """提交安全问答，多 Agent 协作后返回最终报告。"""
    if not _state.get("coordinator"):
        raise HTTPException(423, "服务未初始化")
    t0 = time.time()
    report = _state["coordinator"].run(body.query)
    return {
        "query":      body.query,
        "answer":     report,
        "latency_ms": round((time.time() - t0) * 1000),
        "version":    "2.0.0-multi-agent",
    }


@app.get("/api/v1/alerts/{alert_id}", response_model=dict)
async def get_alert(alert_id: str):
    alert = _state.get("alerts", {}).get(alert_id)
    if not alert:
        raise HTTPException(404, f"告警 {alert_id} 不存在")
    return alert


@app.get("/api/v1/cve/{cve_id}", response_model=dict)
async def get_cve(cve_id: str):
    cve = _state.get("cve", {}).get(cve_id.upper())
    if not cve:
        raise HTTPException(404, f"CVE {cve_id} 不存在")
    return cve


@app.get("/api/v1/agent/stages", response_model=dict)
async def agent_stages(query: str):
    """调试接口：查看查询会被路由到哪些 Agent，以及每个 Agent 的独立输出。"""
    if not _state.get("coordinator"):
        raise HTTPException(423, "服务未初始化")

    coord = _state["coordinator"]
    plan = coord._classify_query(query)

    # 准备上下文
    alert_id  = re.search(r'ALERT-(\d+)', query)
    cve_id    = re.search(r'CVE-\d{4}-\d{4,}', query, re.IGNORECASE)
    ctx = AgentContext(query=query)
    if alert_id:
        ctx.alert_data = coord.alerts.get(f"ALERT-{alert_id.group(1)}", {})
    if cve_id:
        ctx.cve_data = coord.cve_db.get(cve_id.group(0).upper(), {})

    stages = []
    for agent_name in plan:
        t0 = time.time()
        try:
            if agent_name == "AlertAnalystAgent":
                r = coord.alert_agent.run(ctx)
            elif agent_name == "CVEResearcherAgent":
                r = coord.cve_agent.run(ctx)
            elif agent_name == "KnowledgeAgent":
                r = coord.kb_agent.run(ctx, coord.index, coord.tokenizer, coord.model, coord.texts)
            else:
                continue  # ReportAgent 最后统一生成
            stages.append({
                "agent": agent_name,
                "success": r.success,
                "latency_ms": round((time.time() - t0) * 1000),
                "output_preview": r.content[:300] if r.content else "",
            })
        except Exception as e:
            stages.append({"agent": agent_name, "success": False, "error": str(e)})

    return {"query": query, "routing_plan": plan, "stages": stages}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
