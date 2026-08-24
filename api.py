"""
SecOps-Copilot FastAPI 接口
启动方式: uvicorn api:app --host 0.0.0.0 --port 8000
"""
import os, sys, json, time
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI

# ── 路径与缓存（与 agent_main.py 保持一致）─────────────────
BASE_DIR  = Path(__file__).parent
CACHE_DIR = Path(r"D:\RAG1\huggingface_cache")
os.environ["HF_HOME"]            = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FAISS_DIR       = BASE_DIR / "faiss"
FAISS_INDEX_PATH = str(FAISS_DIR / "index_faiss.index")
EMBEDDINGS_PATH  = str(FAISS_DIR / "embeddings.npy")
TEXTS_PATH       = str(FAISS_DIR / "chunk_texts.json")
MODEL_PATH       = str(CACHE_DIR / "hub/models--BAAI--bge-small-zh/snapshots/1d2363c5de6ce9ba9c890c8e23a4c72dce540ca8")

ALERTS_PATH   = str(BASE_DIR / "alerts.json")
CVE_DB_PATH   = str(BASE_DIR / "cve_db.json")
SIM_THRESHOLD = 0.75

BASE_URL  = "https://apihub.agnes-ai.cn/v1"
API_KEY   = os.environ["AGNES_API_KEY"]
MODEL_NAME = "agnes-2.0-flash"
MAX_AGENT_ROUND = 8

# ── 全局状态 ──────────────────────────────────────────────
_state = {}   # index, tokenizer, model, texts, client, tool_map


# ══════════════════════════════════════════════════════════
#  加载模块（与 agent_main.py 逻辑一致，此处内联避免循环导入）
# ══════════════════════════════════════════════════════════
def _load_vector_store():
    if os.path.exists(FAISS_INDEX_PATH):
        index       = faiss.read_index(FAISS_INDEX_PATH)
        embeddings  = np.load(EMBEDDINGS_PATH)
        with open(TEXTS_PATH, encoding="utf-8") as f:
            texts = json.load(f)
        print(f"[API] 加载 FAISS 索引: {index.ntotal} 条")
    else:
        raise RuntimeError(f"FAISS 索引不存在，请先运行 agent_main.py 构建向量库: {FAISS_INDEX_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModel.from_pretrained(MODEL_PATH)
    model.eval()
    return index, tokenizer, model, texts


def _encode_text(tokenizer, model, text: str) -> np.ndarray:
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state[:, 0, :].numpy()[0].astype("float32")
    emb = emb.reshape(1, -1)
    faiss.normalize_L2(emb)
    return emb[0]


def _search_knowledge(query: str, index, tokenizer, model, texts) -> dict:
    q_emb = _encode_text(tokenizer, model, query).reshape(1, -1)
    scores, indices = index.search(q_emb, k=3)
    results = []
    for j in range(len(indices[0])):
        idx = int(indices[0][j])
        score = float(scores[0][j])
        if idx < 0 or score < SIM_THRESHOLD:
            continue
        results.append({"score": round(score, 4), "content": texts[idx]})
    return {"status": "ok" if results else "not_found", "hits": len(results), "results": results}


def _load_db(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return {item["alert_id" if "alert_id" in item else "cve_id"]: item
                for item in json.load(f)}


SYSTEM_PROMPT = """\
你是 SecOps-Copilot，探真科技云原生安全运维 AI 助手。

## 工具调用规则
1. 查询安全文档/最佳实践/响应流程 → search_knowledge
2. 查询具体告警 ID 详情 → query_security_alert
3. 查询 CVE 漏洞信息 → get_cve_info
4. 多个工具可并行调用
5. **未查到文档或数据时，明确告知"暂无相关资料"，禁止编造**

## 输出格式要求
收到告警分析请求后，请按以下结构输出：

### 风险研判
- 告警类型与等级
- 影响范围
- 攻击路径分析

### 处置建议
1. 立即措施（遏制）
2. 根除措施
3. 恢复验证

### 参考依据
"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "查询云原生安全私有知识库（CIS基线、最佳实践、事件响应手册、攻击模式防御）",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string", "description": "查询问题"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_security_alert",
            "description": "根据告警ID查询安全告警详情（等级、影响范围、IOC、处置建议）",
            "parameters": {
                "type": "object",
                "required": ["alert_id"],
                "properties": {"alert_id": {"type": "string", "description": "告警ID"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cve_info",
            "description": "根据CVE编号查询漏洞详情（严重程度、影响版本、修复方案）",
            "parameters": {
                "type": "object",
                "required": ["cve_id"],
                "properties": {"cve_id": {"type": "string", "description": "CVE编号"}},
            },
        },
    },
]


def _run_agent(user_query: str, state: dict) -> str:
    index, tokenizer, model, texts             = state["index"], state["tok"], state["model"], state["texts"]
    alerts_by_id, cve_by_id                    = state["alerts"], state["cve"]
    client, system_prompt, tools, max_rounds   = state["client"], state["prompt"], state["tools"], state["max_round"]

    def search_knowledge(query: str) -> str:
        return json.dumps(_search_knowledge(query, index, tokenizer, model, texts), ensure_ascii=False)

    def query_security_alert(alert_id: str) -> str:
        if alert_id in alerts_by_id:
            return json.dumps(alerts_by_id[alert_id], ensure_ascii=False, indent=2)
        return json.dumps({"status": "not_found", "message": f"告警 {alert_id} 不存在"}, ensure_ascii=False)

    def get_cve_info(cve_id: str) -> str:
        if cve_id in cve_by_id:
            return json.dumps(cve_by_id[cve_id], ensure_ascii=False, indent=2)
        return json.dumps({"status": "not_found", "message": f"CVE {cve_id} 未查询到"}, ensure_ascii=False)

    tool_map = {"search_knowledge": search_knowledge,
                "query_security_alert": query_security_alert,
                "get_cve_info": get_cve_info}

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_query},
    ]
    tool_call_history = []

    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, tools=tools, temperature=0.1)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content

        for tc in msg.tool_calls:
            key = f"{tc.function.name}:{tc.function.arguments}"
            if key in tool_call_history:
                err = f"重复调用检测，跳过 {tc.function.name}"
                messages.append(msg.model_dump())
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": err})
                continue
            tool_call_history.append(key)

            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                err = f"参数解析失败: {e}"
                messages.append(msg.model_dump())
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": err})
                continue

            func  = tool_map.get(tc.function.name)
            ret   = func(**args) if func else json.dumps({"error": f"工具 {tc.function.name} 不存在"}, ensure_ascii=False)
            messages.append(msg.model_dump())
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(ret)})

    return f"达到最大轮次 {max_rounds}，Agent 终止。"


# ══════════════════════════════════════════════════════════
#  FastAPI 应用
# ══════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] 正在初始化向量库...")
    index, tokenizer, model, texts = _load_vector_store()
    alerts   = _load_db(ALERTS_PATH)
    cve_db   = _load_db(CVE_DB_PATH)
    client   = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    _state.update(dict(
        index=index, tok=tokenizer, model=model, texts=texts,
        alerts=alerts, cve=cve_db,
        client=client, prompt=SYSTEM_PROMPT, tools=tools, max_round=MAX_AGENT_ROUND,
    ))
    print(f"[API] 就绪 | 告警库: {len(alerts)} 条 | CVE库: {len(cve_db)} 条 | 知识块: {len(texts)} 条")
    yield
    print("[API] 服务已关闭")

app = FastAPI(
    title="SecOps-Copilot API",
    description="探真科技云原生安全运维 AI 助手 — RAG Agent 服务",
    version="1.0.0",
    lifespan=lifespan,
)


class AgentQuery(BaseModel):
    query: str


@app.get("/api/health")
async def health():
    return {"status": "ok", "alerts": len(_state.get("alerts", {})),
            "cve":   len(_state.get("cve", {})),
            "chunks": len(_state.get("texts", []))}


@app.post("/api/v1/agent/query", response_model=dict)
async def agent_query(body: AgentQuery):
    """提交安全问答，Agent 多轮工具调用后返回最终回答。"""
    if not _state.get("client"):
        raise HTTPException(423, "服务未初始化")
    t0 = time.time()
    answer = _run_agent(body.query, _state)
    return {"query": body.query, "answer": answer, "latency_ms": round((time.time() - t0) * 1000)}


@app.get("/api/v1/alerts/{alert_id}", response_model=dict)
async def get_alert(alert_id: str):
    """查询单条告警详情。"""
    alert = _state.get("alerts", {}).get(alert_id)
    if not alert:
        raise HTTPException(404, f"告警 {alert_id} 不存在")
    return alert


@app.get("/api/v1/cve/{cve_id}", response_model=dict)
async def get_cve(cve_id: str):
    """查询单条 CVE 详情。"""
    cve = _state.get("cve", {}).get(cve_id)
    if not cve:
        raise HTTPException(404, f"CVE {cve_id} 不存在")
    return cve


@app.get("/api/v1/knowledge/search", response_model=dict)
async def knowledge_search(query: str, top_k: int = 3):
    """直接检索知识库，返回相似度分数（供调试用）。"""
    if not _state.get("index"):
        raise HTTPException(423, "服务未初始化")
    index, tok, mod, texts = _state["index"], _state["tok"], _state["model"], _state["texts"]
    return _search_knowledge(query, index, tok, mod, texts)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
