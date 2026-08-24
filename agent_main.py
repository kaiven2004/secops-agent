import os, sys, json, logging, time, faiss, numpy as np, torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI

# ── 路径配置 ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
CACHE_DIR  = Path(r"D:\RAG1\huggingface_cache")
os.environ["HF_HOME"]            = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOG_DIR         = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
FAISS_DIR       = BASE_DIR / "faiss"
FAISS_DIR.mkdir(exist_ok=True)
FAISS_INDEX_PATH = str(FAISS_DIR / "index_faiss.index")
EMBEDDINGS_PATH  = str(FAISS_DIR / "embeddings.npy")
TEXTS_PATH       = str(FAISS_DIR / "chunk_texts.json")

# ── 结构化日志 ────────────────────────────────────────────
_log_fmt = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=_log_fmt,
    handlers=[
        logging.FileHandler(LOG_DIR / "agent.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("SecOpsAgent")

# ── LLM 配置 ─────────────────────────────────────────────
BASE_URL  = "https://apihub.agnes-ai.cn/v1"
API_KEY   = os.environ["AGNES_API_KEY"]
MODEL_NAME = "agnes-2.0-flash"
MAX_AGENT_ROUND = 8
SIM_THRESHOLD   = 0.75
MODEL_PATH      = str(CACHE_DIR / "hub/models--BAAI--bge-small-zh/snapshots/1d2363c5de6ce9ba9c890c8e23a4c72dce540ca8")

# ── 工具配置（从 JSON 文件加载真实数据）─────────────────
ALERTS_PATH   = str(BASE_DIR / "alerts.json")
CVE_DB_PATH   = str(BASE_DIR / "cve_db.json")
CHUNKS_PATH   = str(BASE_DIR / "knowledge_chunks.json")


# ══════════════════════════════════════════════════════════
#  1. 向量库构建（FAISS 持久化）
# ══════════════════════════════════════════════════════════
def build_vector_store(force_rebuild: bool = False):
    """构建 FAISS 向量库并持久化到磁盘。
    首次运行或显式 force_rebuild=True 时重新计算嵌入；
    否则直接加载已有的 .index / .npy / .json 文件。"""

    index_exists = os.path.exists(FAISS_INDEX_PATH) and not force_rebuild

    if index_exists:
        log.info("从磁盘加载已有 FAISS 索引...")
        index       = faiss.read_index(FAISS_INDEX_PATH)
        embeddings  = np.load(EMBEDDINGS_PATH)
        with open(TEXTS_PATH, encoding="utf-8") as f:
            texts = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model     = AutoModel.from_pretrained(MODEL_PATH)
        model.eval()
        log.info(f"加载完成：{index.ntotal} 条向量，维度 {index.d}")
    else:
        log.info("构建向量库（首次或强制重建）...")
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            chunk_list = json.load(f)
        texts = [item["content"] for item in chunk_list]
        log.info(f"共 {len(texts)} 条知识块，正在加载 Embedding 模型...")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model     = AutoModel.from_pretrained(MODEL_PATH)
        model.eval()

        embeddings_list = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            # BGE 使用 mean pooling 更稳定
            emb = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings_list.append(emb)
            log.info(f"  嵌入进度：{min(i + batch_size, len(texts))}/{len(texts)}")

        embeddings = np.vstack(embeddings_list).astype("float32")
        dimension  = embeddings.shape[1]

        # L2 归一化 → 内积 = 余弦相似度
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dimension)   # Inner Product
        index.add(embeddings)

        # 持久化
        faiss.write_index(index, FAISS_INDEX_PATH)
        np.save(EMBEDDINGS_PATH, embeddings)
        with open(TEXTS_PATH, "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False)
        log.info(f"向量库已保存 → {FAISS_INDEX_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model.eval()

    return index, tokenizer, model, texts


# ══════════════════════════════════════════════════════════
#  2. 检索工具
# ══════════════════════════════════════════════════════════
def encode_text(tokenizer, model, text: str) -> np.ndarray:
    """编码单条文本，返回 L2 归一化后的向量。"""
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state[:, 0, :].numpy()[0].astype("float32")
    emb = emb.reshape(1, -1)
    faiss.normalize_L2(emb)
    return emb[0]


def search_knowledge(query: str, index, tokenizer, model, texts) -> str:
    """余弦相似度检索，返回 Top-3 高于阈值的知识块。"""
    q_emb = encode_text(tokenizer, model, query).reshape(1, -1)
    scores, indices = index.search(q_emb, k=3)

    results = []
    for j in range(len(indices[0])):
        idx = int(indices[0][j])
        score = float(scores[0][j])
        if idx < 0 or score < SIM_THRESHOLD:
            continue
        results.append({"score": round(score, 4), "content": texts[idx]})

    if not results:
        return json.dumps(
            {"status": "not_found", "message": "知识库未查询到相关安全文档，请确认问题关键词。"},
            ensure_ascii=False,
        )
    return json.dumps(
        {"status": "ok", "hits": len(results), "results": results},
        ensure_ascii=False,
    )


# ══════════════════════════════════════════════════════════
#  3. 业务查询工具（真实 JSON 数据）
# ══════════════════════════════════════════════════════════
def _load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

alerts_db   = _load_json(ALERTS_PATH)
cve_db_list = _load_json(CVE_DB_PATH)
# 建索引加速查找
alerts_by_id = {a["alert_id"]: a for a in alerts_db}
cve_by_id    = {c["cve_id"]:    c for c in cve_db_list}


def query_security_alert(alert_id: str) -> str:
    """根据告警 ID 查询安全告警详情。"""
    if alert_id in alerts_by_id:
        return json.dumps(alerts_by_id[alert_id], ensure_ascii=False, indent=2)
    return json.dumps(
        {"status": "not_found", "message": f"告警 {alert_id} 在告警库不存在。"},
        ensure_ascii=False,
    )


def get_cve_info(cve_id: str) -> str:
    """根据 CVE 编号查询漏洞详情。"""
    if cve_id in cve_by_id:
        return json.dumps(cve_by_id[cve_id], ensure_ascii=False, indent=2)
    return json.dumps(
        {"status": "not_found", "message": f"CVE {cve_id} 未在漏洞库查询到记录。"},
        ensure_ascii=False,
    )


# ══════════════════════════════════════════════════════════
#  4. Agent 核心循环
# ══════════════════════════════════════════════════════════
SYSTEM_PROMPT = """\
你是 SecOps-Copilot，探真科技云原生安全运维 AI 助手。

## 工具调用规则
1. 查询安全文档/最佳实践/响应流程 → search_knowledge
2. 查询具体告警 ID 详情 → query_security_alert
3. 查询 CVE 漏洞信息 → get_cve_info
4. 多个工具可并行调用，无需等待上一个结果
5. **未查到文档或数据时，明确告知"暂无相关资料"，禁止编造任何内容**

## 输出格式要求
收到告警分析请求后，请按以下结构输出：

### 风险研判
- 告警类型与等级
- 影响范围（集群/命名空间/容器）
- 攻击路径分析

### 处置建议
1. 立即措施（遏制）
2. 根除措施
3. 恢复验证

### 参考依据
- 引用的知识库文档片段
- 关联的 CVE 编号（如有）
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
                "properties": {
                    "query": {"type": "string", "description": "查询问题，应简洁明确"}
                },
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
                "properties": {
                    "alert_id": {"type": "string", "description": "告警ID，格式如 ALERT-0815"}
                },
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
                "properties": {
                    "cve_id": {"type": "string", "description": "CVE编号，格式如 CVE-2024-21762"}
                },
            },
        },
    },
]

tool_map = {
    "search_knowledge":   None,   # 占位，运行时注入
    "query_security_alert": None,
    "get_cve_info":       None,
}

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def agent_run(user_query: str, index, tokenizer, model, texts):
    """Agent 主循环：多轮工具调用 + LLM 最终回答。"""
    # 注入依赖
    tool_map["search_knowledge"]   = lambda **kw: search_knowledge(user_query if "query" not in kw else kw["query"], index, tokenizer, model, texts)
    tool_map["query_security_alert"] = query_security_alert
    tool_map["get_cve_info"]       = get_cve_info

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_query},
    ]

    log.info(f"{'='*60}")
    log.info(f"[START] 用户查询: {user_query}")
    log.info(f"{'='*60}")

    tool_call_history = []  # 防重复调用检测

    for round_num in range(MAX_AGENT_ROUND):
        round_start = time.time()
        log.info(f"── Agent 第 {round_num + 1} 轮 ──")

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            temperature=0.1,
        )
        msg      = resp.choices[0].message
        duration = round(time.time() - round_start, 2)
        log.info(f"LLM 响应耗时: {duration}s, 工具调用数: {len(msg.tool_calls or [])}")

        # 无工具调用 → 最终回答
        if not msg.tool_calls:
            log.info(f"──── 最终回答 ────\n{msg.content}")
            print(f"\n{'='*60}")
            print(f"Agent 最终回答：\n{msg.content}")
            print(f"{'='*60}\n")
            return msg.content

        # 处理工具调用
        for tc in msg.tool_calls:
            func_name = tc.function.name
            log.info(f"调用工具: {func_name}，参数: {tc.function.arguments}")

            # 防重复调用检测
            call_key = f"{func_name}:{tc.function.arguments}"
            if call_key in tool_call_history:
                err = f"重复调用检测：工具 {func_name} 已执行过相同参数，跳过以避免死循环"
                log.warning(err)
                messages.append(msg.model_dump())
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": err})
                continue
            tool_call_history.append(call_key)

            # 解析参数
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                err = f"参数解析失败：{e}"
                log.error(err)
                messages.append(msg.model_dump())
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": err})
                continue

            # 执行工具
            if func_name not in tool_map:
                tool_ret = json.dumps({"status": "error", "message": f"不存在工具 {func_name}"}, ensure_ascii=False)
            else:
                try:
                    tool_ret = tool_map[func_name](**args)
                    log.info(f"工具返回: {tool_ret[:200]}...")
                except Exception as e:
                    tool_ret = json.dumps({"status": "error", "message": f"执行异常：{e}"}, ensure_ascii=False)
                    log.error(f"工具执行异常 {func_name}: {e}")

            messages.append(msg.model_dump())
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(tool_ret),
            })

    log.warning(f"达到最大迭代轮次 {MAX_AGENT_ROUND}，Agent 强制终止")
    print(f"\n⚠  达到最大迭代轮次 {MAX_AGENT_ROUND}，Agent 终止")
    return None


# ══════════════════════════════════════════════════════════
#  5. 入口
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    index, tokenizer, model, texts = build_vector_store()
    log.info("向量库就绪，启动 Agent 服务")

    # 默认测试查询
    default_query = "帮我分析告警ALERT-0815，给出风险研判以及处置建议。"

    # 支持命令行参数
    import sys as _sys
    query = _sys.argv[1] if len(_sys.argv) > 1 else default_query
    agent_run(query, index, tokenizer, model, texts)
