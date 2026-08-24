"""
SecOps-Copilot — 多 Agent 协作版
架构：
  Coordinator → 按查询类型路由到专业 Agent
    ├─ AlertAnalystAgent  → 告警分析与风险研判
    ├─ CVEResearcherAgent → CVE 漏洞研究与修复方案
    ├─ KnowledgeAgent     → 安全知识库 RAG 检索
    └─ ReportAgent        → 结构化报告生成
"""
import os, sys, json, time, re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import logging
import faiss
import numpy as np
import torch
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
ALERTS_PATH      = str(BASE_DIR / "alerts.json")
CVE_DB_PATH      = str(BASE_DIR / "cve_db.json")

# ── 结构化日志 ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "agent.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("SecOpsAgent")

# ── 配置 ──────────────────────────────────────────────────
BASE_URL      = "https://apihub.agnes-ai.cn/v1"
API_KEY       = os.environ["AGNES_API_KEY"]
MODEL_NAME    = "agnes-2.0-flash"
MODEL_PATH    = str(CACHE_DIR / "hub/models--BAAI--bge-small-zh/snapshots/1d2363c5de6ce9ba9c890c8e23a4c72dce540ca8")
SIM_THRESHOLD = 0.75
MAX_ROUNDS    = 8


# ══════════════════════════════════════════════════════════
#  数据结构
# ══════════════════════════════════════════════════════════
@dataclass
class AgentContext:
    """Agent 间共享上下文。"""
    query: str
    alert_data: dict = field(default_factory=dict)
    cve_data:   dict = field(default_factory=dict)
    knowledge:  dict = field(default_factory=dict)
    analysis:   dict = field(default_factory=dict)
    final_report: str = ""
    meta: dict = field(default_factory=dict)   # 跨 Agent 传递的元信息


@dataclass
class AgentResult:
    """单个 Agent 的执行结果。"""
    agent_name: str
    success: bool
    content: str
    extra: dict = field(default_factory=dict)
    latency_s: float = 0.0


# ══════════════════════════════════════════════════════════
#  共享基础设施（向量库 + 数据库）
# ══════════════════════════════════════════════════════════
def load_infra() -> tuple:
    """加载向量库和业务数据。"""
    index = faiss.read_index(FAISS_INDEX_PATH)
    texts = json.load(open(TEXTS_PATH, encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)
    model.eval()
    alerts  = {a["alert_id"]: a for a in json.load(open(ALERTS_PATH, encoding="utf-8"))}
    cve_db  = {c["cve_id"]:    c  for c in json.load(open(CVE_DB_PATH, encoding="utf-8"))}
    client  = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    log.info(f"基础设施就绪 | 向量库:{index.ntotal}条 告警:{len(alerts)}条 CVE:{len(cve_db)}条")
    return index, texts, tokenizer, model, alerts, cve_db, client


def _embed(tokenizer, model, text: str) -> np.ndarray:
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state[:, 0, :].numpy()[0].astype("float32")
    emb = emb.reshape(1, -1)
    faiss.normalize_L2(emb)
    return emb[0]


def _search_knowledge(query: str, index, tokenizer, model, texts) -> list:
    emb = _embed(tokenizer, model, query)
    scores, indices = index.search(emb.reshape(1, -1), k=3)
    results = []
    for j in range(len(indices[0])):
        idx = int(indices[0][j])
        score = float(scores[0][j])
        if idx < 0 or score < SIM_THRESHOLD:
            continue
        results.append({"score": round(score, 4), "content": texts[idx]})
    return results


# ══════════════════════════════════════════════════════════
#  Agent 定义
# ══════════════════════════════════════════════════════════

class AlertAnalystAgent:
    """告警研判 Agent — 分析告警等级、影响范围、攻击路径。"""

    SYSTEM_PROMPT = """\
你是安全研判分析师。根据告警数据给出结构化风险研判。

## 输出格式（严格 JSON）
{
  "alert_type": "告警类型",
  "severity": "critical/high/medium/low",
  "impact_scope": "影响范围描述",
  "attack_path": "攻击路径分析",
  "risk_level": "风险等级数字 1-10"
}

只输出 JSON，不要任何其他内容。"""

    def __init__(self, client: OpenAI, model: str = MODEL_NAME):
        self.client = client
        self.model = model

    def run(self, ctx: AgentContext) -> AgentResult:
        t0 = time.time()
        alert = ctx.alert_data
        if not alert:
            return AgentResult("AlertAnalyst", False, "无告警数据", latency_s=time.time()-t0)

        prompt = f"""分析以下安全告警：

告警ID: {alert.get('alert_id', 'N/A')}
类型: {alert.get('type', 'N/A')}
等级: {alert.get('level', 'N/A')}
集群: {alert.get('cluster', 'N/A')}
命名空间: {alert.get('namespace', 'N/A')}
容器: {alert.get('container', 'N/A')}
镜像: {alert.get('ioc', {}).get('image', 'N/A')}
描述: {alert.get('desc', 'N/A')}
IOC: {json.dumps(alert.get('ioc', {}), ensure_ascii=False)}
建议操作: {alert.get('recommended_action', 'N/A')}

请输出风险研判 JSON。"""

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.SYSTEM_PROMPT},
                      {"role": "user",   "content": prompt}],
            temperature=0.1,
        )
        content = resp.choices[0].message.content or ""
        # 解析 JSON
        try:
            # 提取 JSON 块
            match = re.search(r'\{[^{}]+\}', content, re.DOTALL)
            analysis = json.loads(match.group()) if match else {}
        except json.JSONDecodeError:
            analysis = {"raw": content}

        ctx.analysis = analysis
        log.info(f"[AlertAnalyst] 研判完成 | severity={analysis.get('severity', 'N/A')} risk={analysis.get('risk_level', 'N/A')}")
        return AgentResult("AlertAnalyst", True, content, analysis, latency_s=time.time()-t0)


class CVEResearcherAgent:
    """CVE 研究 Agent — 查询漏洞详情并关联知识库。"""

    SYSTEM_PROMPT = """\
你是漏洞研究专家。根据 CVE 信息给出漏洞分析和修复建议。

## 输出格式（严格 JSON）
{
  "cve_id": "CVE编号",
  "vulnerability_summary": "漏洞概述",
  "attack_vector": "攻击向量",
  "fix_version": "修复版本",
  "workaround": "临时缓解措施",
  "references": ["相关参考"]
}

只输出 JSON。"""

    def __init__(self, client: OpenAI, cve_db: dict, model: str = MODEL_NAME):
        self.client = client
        self.cve_db = cve_db
        self.model = model

    def run(self, ctx: AgentContext) -> AgentResult:
        t0 = time.time()
        # 从原始查询中提取 CVE 编号
        cve_ids = re.findall(r'CVE-\d{4}-\d{4,}', ctx.query.upper())
        if not cve_ids:
            return AgentResult("CVEResearcher", False, "查询中未包含 CVE 编号", latency_s=time.time()-t0)

        cve_id = cve_ids[0]
        cve = self.cve_db.get(cve_id)
        if not cve:
            return AgentResult("CVEResearcher", False, f"CVE {cve_id} 未查询到记录",
                           {"cve_id": cve_id, "status": "not_found"},
                           latency_s=time.time()-t0)

        prompt = f"""分析以下 CVE 漏洞并给出修复建议：

CVE: {cve_id}
产品: {cve.get('product', 'N/A')}
严重程度: {cve.get('severity', 'N/A')} (CVSS: {cve.get('cvss_score', 'N/A')})
影响版本: {cve.get('affect_version', 'N/A')}
修复版本: {cve.get('fixed_version', 'N/A')}
描述: {cve.get('description', 'N/A')}
攻击向量: {cve.get('attack_vector', 'N/A')}
缓解措施: {cve.get('mitigation', 'N/A')}

请输出漏洞分析 JSON。"""

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.SYSTEM_PROMPT},
                      {"role": "user",   "content": prompt}],
            temperature=0.1,
        )
        content = resp.choices[0].message.content or ""
        try:
            match = re.search(r'\{[^{}]+\}', content, re.DOTALL)
            analysis = json.loads(match.group()) if match else {}
        except json.JSONDecodeError:
            analysis = {"raw": content}

        ctx.cve_data = {"id": cve_id, "info": cve, "analysis": analysis}
        log.info(f"[CVEResearcher] 完成 | {cve_id} severity={cve.get('severity')}")
        return AgentResult("CVEResearcher", True, content,
                           {"cve_id": cve_id, "cve_info": cve, "analysis": analysis},
                           latency_s=time.time()-t0)


class KnowledgeAgent:
    """知识检索 Agent — RAG 检索 + 知识总结。"""

    SYSTEM_PROMPT = """\
你是安全知识检索专家。根据检索到的安全文档回答用户问题。

## 规则
1. 只基于提供的文档内容回答
2. 文档未覆盖的内容明确告知"暂无相关资料"
3. 输出格式为结构化的知识要点，包含文档来源和关键信息"""

    def __init__(self, client: OpenAI, model: str = MODEL_NAME):
        self.client = client
        self.model = model

    def run(self, ctx: AgentContext, index=None, tokenizer=None, model=None, texts=None) -> AgentResult:
        t0 = time.time()
        # 根据上下文决定查询词
        if ctx.alert_data:
            query = f"{ctx.alert_data.get('type', '')} {ctx.alert_data.get('desc', '')[:50]} 安全加固 防御措施"
        elif ctx.cve_data:
            cve = ctx.cve_data.get("info", {})
            query = f"{cve.get('product','')} {cve.get('description','')} 漏洞修复 安全最佳实践"
        else:
            query = ctx.query

        kb_results = _search_knowledge(query, index, tokenizer, model, texts)

        if not kb_results:
            return AgentResult("KnowledgeAgent", True, "知识库未检索到相关内容",
                               {"query": query, "hits": 0}, latency_s=time.time()-t0)

        docs_text = "\n\n---\n\n".join(
            [f"[来源: {i+1}] (相似度: {r['score']})\n{r['content']}" for i, r in enumerate(kb_results)]
        )

        prompt = f"""请根据以下安全文档，回答用户问题。

用户问题: {ctx.query}

检索到的安全文档：
{docs_text}

请输出结构化的知识要点。"""

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.SYSTEM_PROMPT},
                      {"role": "user",   "content": prompt}],
            temperature=0.1,
        )
        content = resp.choices[0].message.content or ""
        ctx.knowledge = {"query": query, "hits": len(kb_results), "results": kb_results, "summary": content}
        log.info(f"[KnowledgeAgent] 完成 | 检索到 {len(kb_results)} 条知识")
        return AgentResult("KnowledgeAgent", True, content,
                           {"query": query, "hits": len(kb_results), "top_results": kb_results[:2]},
                           latency_s=time.time()-t0)


class ReportAgent:
    """报告生成 Agent — 汇总各 Agent 结果，生成最终结构化报告。"""

    SYSTEM_PROMPT = """\
你是安全运营报告生成器。将各专家的分析结果整合为一份专业的安全运营报告。

## 报告格式要求
严格按照以下 Markdown 格式输出：

# 安全运营报告

## 1. 执行摘要
（2-3 句话概述事件性质、严重程度和影响）

## 2. 风险研判
### 2.1 事件概况
- 告警/CVE 编号
- 严重程度
- 影响范围

### 2.2 攻击路径分析
（详细说明攻击者可能如何利用此漏洞/告警）

## 3. 处置建议
### 3.1 立即措施（遏制）
### 3.2 根除措施
### 3.3 恢复验证

## 4. 参考依据
（列出引用的知识库文档和 CVE 来源）

注意：报告要专业、简洁、可操作性强。"""

    def __init__(self, client: OpenAI, model: str = MODEL_NAME):
        self.client = client
        self.model = model

    def run(self, ctx: AgentContext, results: list) -> AgentResult:
        t0 = time.time()

        # 组装各 Agent 的结果
        parts = []
        for r in results:
            if r.success:
                parts.append(f"## [{r.agent_name}]\n{r.content}")

        prompt = f"""请将以下各专家的分析结果整合为一份完整的安全运营报告。

用户原始问题: {ctx.query}

各专家分析：
{'\n\n'.join(parts)}

请输出完整报告。"""

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.SYSTEM_PROMPT},
                      {"role": "user",   "content": prompt}],
            temperature=0.3,
        )
        report = resp.choices[0].message.content or ""
        ctx.final_report = report
        log.info(f"[ReportAgent] 报告生成完成 | 长度={len(report)}字符")
        return AgentResult("ReportAgent", True, report,
                           {"total_agents": len(results), "success_agents": sum(1 for r in results if r.success)},
                           latency_s=time.time()-t0)


# ══════════════════════════════════════════════════════════
#  Coordinator — 智能路由与编排
# ══════════════════════════════════════════════════════════
class Coordinator:
    """
    协调多个专业 Agent，根据用户查询类型进行智能路由：

    查询类型判断：
    - 包含 ALERT-xxx → AlertAnalystAgent + KnowledgeAgent + ReportAgent
    - 包含 CVE-xxx   → CVEResearcherAgent + KnowledgeAgent + ReportAgent
    - 其他安全问答   → KnowledgeAgent + ReportAgent
    """

    SYSTEM_PROMPT = """\
你是 SecOps-Copilot 多 Agent 协调器。你的职责是：
1. 理解用户的安全运营问题
2. 判断问题类型并路由到合适的专业 Agent
3. 汇总各 Agent 的结果生成最终报告

## 路由规则
- 用户提到具体告警ID（如 ALERT-xxx）→ 调用 AlertAnalystAgent + KnowledgeAgent
- 用户提到 CVE 编号 → 调用 CVEResearcherAgent + KnowledgeAgent
- 用户询问安全知识/最佳实践 → 调用 KnowledgeAgent
- 所有情况最终都调用 ReportAgent 生成结构化报告

## 重要
- 不要重复调用同一 Agent 处理相同数据
- 如果某个 Agent 失败，记录错误但继续流程
- 最终报告要专业、完整、可操作"""

    def __init__(self, client: OpenAI, alerts: dict, cve_db: dict,
                 index, texts, tokenizer, model):
        self.client = client
        self.alerts = alerts
        self.cve_db = cve_db
        self.index = index
        self.texts = texts
        self.tokenizer = tokenizer
        self.model = model

        # 初始化各 Agent
        self.alert_agent  = AlertAnalystAgent(client)
        self.cve_agent    = CVEResearcherAgent(client, cve_db)
        self.kb_agent     = KnowledgeAgent(client)
        self.report_agent = ReportAgent(client)

    def _classify_query(self, query: str) -> list:
        """根据查询内容决定调用哪些 Agent。"""
        agents = []
        has_alert = bool(re.search(r'ALERT-\d+', query))
        has_cve   = bool(re.search(r'CVE-\d{4}-\d{4,}', query, re.IGNORECASE))

        if has_alert:
            agents.append("AlertAnalystAgent")
        if has_cve:
            agents.append("CVEResearcherAgent")
        # 知识检索几乎总是需要的
        agents.append("KnowledgeAgent")
        # 报告生成最后总是需要的
        agents.append("ReportAgent")
        return agents

    def run(self, query: str) -> str:
        log.info(f"{'='*60}")
        log.info(f"[Coordinator] 收到查询: {query}")
        log.info(f"{'='*60}")

        # 1. 分类路由
        agent_plan = self._classify_query(query)
        log.info(f"[Coordinator] 路由计划: {agent_plan}")

        # 2. 准备上下文
        # 提取告警/CVE 数据
        alert_id  = re.search(r'ALERT-(\d+)', query)
        cve_id    = re.search(r'CVE-\d{4}-\d{4,}', query, re.IGNORECASE)

        ctx = AgentContext(query=query)
        if alert_id:
            aid = f"ALERT-{alert_id.group(1)}"
            ctx.alert_data = self.alerts.get(aid, {"alert_id": aid, "status": "not_found"})

        if cve_id:
            cid = cve_id.group(0).upper()
            ctx.cve_data = self.cve_db.get(cid, {"cve_id": cid, "status": "not_found"})

        # 3. 顺序执行各 Agent
        all_results = []
        for agent_name in agent_plan:
            t_start = time.time()
            log.info(f"── 执行 {agent_name} ──")

            try:
                if agent_name == "AlertAnalystAgent":
                    result = self.alert_agent.run(ctx)
                elif agent_name == "CVEResearcherAgent":
                    result = self.cve_agent.run(ctx)
                elif agent_name == "KnowledgeAgent":
                    result = self.kb_agent.run(ctx, self.index, self.tokenizer, self.model, self.texts)
                elif agent_name == "ReportAgent":
                    result = self.report_agent.run(ctx, all_results)
                else:
                    log.warning(f"未知 Agent: {agent_name}")
                    continue

                all_results.append(result)
                log.info(f"[{agent_name}] {'✓' if result.success else '✗'} ({result.latency_s:.1f}s)")

            except Exception as e:
                log.error(f"[{agent_name}] 执行异常: {e}")
                all_results.append(AgentResult(agent_name, False, str(e), latency_s=time.time()-t_start))

        # 4. 输出最终报告
        report_result = next((r for r in all_results if r.agent_name == "ReportAgent"), None)
        if report_result and report_result.success:
            log.info(f"{'='*60}")
            log.info(f"[Coordinator] 最终报告已生成 ({len(report_result.content)} 字符)")
            log.info(f"{'='*60}")
            return report_result.content
        else:
            error_msgs = [r.content for r in all_results if not r.success]
            return f"Agent 执行失败: {'; '.join(error_msgs)}"


# ══════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    index, texts, tokenizer, model, alerts, cve_db, client = load_infra()
    coordinator = Coordinator(client, alerts, cve_db, index, texts, tokenizer, model)

    import sys as _sys
    query = _sys.argv[1] if len(_sys.argv) > 1 else "帮我分析告警ALERT-0815，给出风险研判以及处置建议。"
    report = coordinator.run(query)
    print(f"\n{'='*60}")
    print(report)
    print(f"{'='*60}\n")
