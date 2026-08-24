"""
SecOps-Copilot API v2 — 多 Agent 异步并发版
启动: python api.py
"""
import os, sys, json, time
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from agent_main import (
    load_infra, Coordinator, AgentContext, AgentResult,
    FAISS_INDEX_PATH, ALERTS_PATH, CVE_DB_PATH,
)

BASE_DIR  = Path(__file__).parent
CACHE_DIR = Path(r"D:\RAG1\huggingface_cache")
os.environ["HF_HOME"]            = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] 正在初始化多 Agent 基础设施（异步并发版）...")
    index, texts, tokenizer, model, alerts, cve_db, client = load_infra()
    _state.update(dict(
        coordinator=Coordinator(client, alerts, cve_db, index, texts, tokenizer, model),
        alerts=alerts, cve=cve_db, chunks=len(texts),
    ))
    print(f"[API] 就绪 | 告警:{len(alerts)}条 CVE:{len(cve_db)}条 知识块:{len(texts)}条")
    yield
    print("[API] 服务已关闭")


app = FastAPI(
    title="SecOps-Copilot Multi-Agent API v2",
    description="探真科技云原生安全运维 AI 助手 — 异步并发多 Agent 协作服务",
    version="2.0.0",
    lifespan=lifespan,
)


class AgentQuery(BaseModel):
    query: str


class PipelineRequest(BaseModel):
    mode: str = "alert"       # alert | cve
    alert_id: str = None
    cve_id: str = None


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0-async-multi-agent",
            "alerts": len(_state.get("alerts", {})),
            "cve":    len(_state.get("cve", {})),
            "chunks": _state.get("chunks", 0)}


@app.get("/api/v1/agents")
async def list_agents():
    agents = [
        {"name": "AlertAnalystAgent",  "description": "告警研判：分析告警等级、影响范围、攻击路径"},
        {"name": "CVEResearcherAgent", "description": "CVE 研究：漏洞分析、影响版本、修复方案"},
        {"name": "KnowledgeAgent",     "description": "知识检索：RAG 检索安全知识库，返回相关文档"},
        {"name": "ReportAgent",        "description": "报告生成：汇总各 Agent 结果，输出结构化安全报告"},
        {"name": "Coordinator",        "description": "协调器：智能路由查询 + 异步并发编排"},
    ]
    return {"agents": agents, "total": len(agents)}


@app.post("/api/v1/agent/query", response_model=dict)
async def agent_query(body: AgentQuery):
    """提交安全问答，多 Agent 异步并发后返回最终报告。"""
    if not _state.get("coordinator"):
        raise HTTPException(423, "服务未初始化")
    t0 = time.time()
    report = _state["coordinator"].run(body.query)
    return {"query": body.query, "answer": report,
            "latency_ms": round((time.time() - t0) * 1000), "version": "2.0.0"}


@app.post("/api/v1/pipeline/run", response_model=dict)
async def pipeline_run(body: PipelineRequest):
    """数据流水线接口：模拟 SOC 平台接收告警/CVE 并自动研判。"""
    if not _state.get("coordinator"):
        raise HTTPException(423, "服务未初始化")
    coord = _state["coordinator"]

    if body.mode == "alert" and body.alert_id:
        result = coord._run_pipeline_alert(body.alert_id)
    elif body.mode == "cve" and body.cve_id:
        result = coord._run_pipeline_cve(body.cve_id)
    else:
        raise HTTPException(400, "请指定 mode + alert_id 或 cve_id")

    return result


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
    """调试接口：查看查询的路由计划和每个 Agent 的中间输出。"""
    if not _state.get("coordinator"):
        raise HTTPException(423, "服务未初始化")
    coord = _state["coordinator"]
    plan = coord._classify_query(query)

    alert_id  = __import__('re').search(r'ALERT-(\d+)', query)
    cve_id    = __import__('re').search(r'CVE-\d{4}-\d{4,}', query, __import__('re').IGNORECASE)
    ctx = AgentContext(query=query)
    if alert_id:
        ctx.alert_data = coord.alerts.get(f"ALERT-{alert_id.group(1)}", {})
    if cve_id:
        ctx.cve_data = coord.cve_db.get(cve_id.group(0).upper(), {})

    stages = []
    parallel = [n for n in plan if n != "ReportAgent"]
    for name in parallel:
        t0 = time.time()
        try:
            if name == "AlertAnalystAgent":
                r = coord.alert_agent.run(ctx)
            elif name == "CVEResearcherAgent":
                r = coord.cve_agent.run(ctx)
            elif name == "KnowledgeAgent":
                r = coord.kb_agent.run(ctx, coord.index, coord.tokenizer, coord.model, coord.texts)
            else:
                continue
            stages.append({"agent": name, "success": r.success,
                          "latency_ms": round((time.time()-t0)*1000),
                          "output_preview": r.content[:200]})
        except Exception as e:
            stages.append({"agent": name, "success": False, "error": str(e)})

    return {"query": query, "routing_plan": plan, "parallel_stages": stages}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
