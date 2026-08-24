"""
幻觉测试 + 召回率评估脚本
运行: python eval.py
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI

BASE_DIR  = Path(__file__).parent
CACHE_DIR = Path(r"D:\RAG1\huggingface_cache")
os.environ["HF_HOME"]            = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FAISS_INDEX_PATH = str(BASE_DIR / "faiss" / "index_faiss.index")
EMBEDDINGS_PATH  = str(BASE_DIR / "faiss" / "embeddings.npy")
TEXTS_PATH       = str(BASE_DIR / "faiss" / "chunk_texts.json")
MODEL_PATH       = str(CACHE_DIR / "hub/models--BAAI--bge-small-zh/snapshots/1d2363c5de6ce9ba9c890c8e23a4c72dce540ca8")
ALERTS_PATH      = str(BASE_DIR / "alerts.json")
CVE_DB_PATH      = str(BASE_DIR / "cve_db.json")
HC_PATH          = str(BASE_DIR / "hallucination_test_case.json")

SIM_THRESHOLD = 0.75
BASE_URL      = "https://apihub.agnes-ai.cn/v1"
API_KEY       = os.environ["AGNES_API_KEY"]
MODEL_NAME    = "agnes-2.0-flash"


def _embed(tokenizer, model, text: str) -> np.ndarray:
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state[:, 0, :].numpy()[0].astype("float32")
    emb = emb.reshape(1, -1)
    faiss.normalize_L2(emb)
    return emb[0]


# ══════════════════════════════════════════════════════════
#  1. 检索评估
# ══════════════════════════════════════════════════════════
def evaluate_recall():
    print("\n" + "=" * 60)
    print("  RAG 检索召回率评估")
    print("=" * 60)

    index = faiss.read_index(FAISS_INDEX_PATH)
    texts = json.load(open(TEXTS_PATH, encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)
    model.eval()

    # 每个查询期望至少一个召回结果中包含相关关键词
    test_queries = [
        ("容器特权模式有什么风险",           ["特权", "privileged", "逃逸"]),
        ("K8s 网络策略最佳实践",            ["网络策略", "NetworkPolicy", "微隔离"]),
        ("安全事件响应流程",                ["响应流程", "遏制", "根除", "恢复"]),
        ("CIS Kubernetes Benchmark",       ["CIS", "Benchmark", "控制项"]),
        ("containerd 逃逸漏洞修复",         ["containerd", "逃逸", "1.7.22"]),
        ("RBAC 权限最小化",                 ["RBAC", "最小权限", "ServiceAccount"]),
        ("密钥管理最佳实践",                ["密钥", "Secret", "Vault"]),
    ]

    passed = 0
    for query, keywords in test_queries:
        emb = _embed(tokenizer, model, query)
        scores, indices = index.search(emb.reshape(1, -1), k=3)
        hit = False
        top_snippets = []
        for j in range(len(indices[0])):
            idx = int(indices[0][j])
            score = float(scores[0][j])
            snippet = texts[idx][:80] if idx >= 0 else ""
            top_snippets.append(f"[{score:.3f}]{snippet[:40]}")
            if idx >= 0 and any(kw in texts[idx] for kw in keywords):
                hit = True
        status = "✓" if hit else "✗"
        if hit:
            passed += 1
        print(f"  {status} {query:<28} | scores: {[round(float(scores[0][j]), 3) for j in range(len(indices[0]))]}")
        for s in top_snippets:
            print(f"         → {s}")

    total = len(test_queries)
    rate = passed / total * 100 if total else 0
    print(f"\n  召回率: {passed}/{total} = {rate:.0f}%")
    return passed, total


# ══════════════════════════════════════════════════════════
#  2. 幻觉测试
# ══════════════════════════════════════════════════════════
SYSTEM_PROMPT = """\
你是 SecOps-Copilot 安全运维 AI 助手。
规则：
1. 查询安全文档调用 search_knowledge
2. 查询告警调用 query_security_alert
3. 查询 CVE 调用 get_cve_info
4. **未查到文档时禁止编造，明确告知"暂无相关资料"**
5. 信息收集完毕后输出最终回答
"""

tools = [
    {"type":"function","function":{"name":"search_knowledge","description":"查询云原生安全私有知识库","parameters":{"type":"object","required":["query"],"properties":{"query":{"type":"string"}}}}},
    {"type":"function","function":{"name":"query_security_alert","description":"根据告警ID查询安全告警详情","parameters":{"type":"object","required":["alert_id"],"properties":{"alert_id":{"type":"string"}}}}},
    {"type":"function","function":{"name":"get_cve_info","description":"根据CVE编号查询漏洞详情","parameters":{"type":"object","required":["cve_id"],"properties":{"cve_id":{"type":"string"}}}}},
]


def evaluate_hallucination():
    print("\n" + "=" * 60)
    print("  幻觉测试评估")
    print("=" * 60)

    index    = faiss.read_index(FAISS_INDEX_PATH)
    texts    = json.load(open(TEXTS_PATH, encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model    = AutoModel.from_pretrained(MODEL_PATH)
    model.eval()
    alerts   = {a["alert_id"]: a for a in json.load(open(ALERTS_PATH, encoding="utf-8"))}
    cve_db   = {c["cve_id"]: c  for c in json.load(open(CVE_DB_PATH, encoding="utf-8"))}
    client   = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    cases = json.load(open(HC_PATH, encoding="utf-8"))
    results = []

    for i, case in enumerate(cases):
        query    = case["query"]
        expected = case["expect"]
        # 判断是否为"期望有答案"的正面案例
        is_known_answer = any(kw in expected for kw in ["应准确", "有信息", "应返回"])

        # 检索层预处理：看工具能不能查到
        emb = _embed(tokenizer, model, query)
        scores, indices = index.search(emb.reshape(1, -1), k=3)
        knowledge_hit = any(float(scores[0][j]) >= SIM_THRESHOLD for j in range(len(indices[0])) if int(indices[0][j]) >= 0)

        # 告警/CVE 库查找
        alert_hit  = any(aid in query for aid in alerts)
        cve_hit    = any(cid in query for cid in cve_db)

        # 构建带工具结果的对话
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": query},
        ]

        # 模拟工具调用结果（直接注入，跳过 LLM 工具调度）
        tool_results = []
        if knowledge_hit:
            tool_results.append(("search_knowledge", query))
        if alert_hit:
            for aid in alerts:
                if aid in query:
                    tool_results.append(("query_security_alert", aid))
                    break
        if cve_hit:
            for cid in cve_db:
                if cid in query:
                    tool_results.append(("get_cve_info", cid))
                    break

        # 注入工具结果
        for tidx, (fname, arg_val) in enumerate(tool_results):
            if fname == "search_knowledge":
                ret = json.dumps({"status": "ok", "hits": 2, "results": [
                    {"score": round(float(scores[0][j]), 4), "content": texts[int(indices[0][j])][:300]}
                    for j in range(len(indices[0])) if int(indices[0][j]) >= 0 and float(scores[0][j]) >= SIM_THRESHOLD
                ]}, ensure_ascii=False)
            elif fname == "query_security_alert":
                ret = json.dumps(alerts.get(arg_val, {"status": "not_found"}), ensure_ascii=False)
            elif fname == "get_cve_info":
                ret = json.dumps(cve_db.get(arg_val, {"status": "not_found"}), ensure_ascii=False)
            else:
                ret = "[]"
            messages.append({"role": "assistant", "content": None,
                             "tool_calls": [{"id": f"tc_{tidx}", "type": "function",
                                             "function": {"name": fname, "arguments": json.dumps({"dummy": "x"}, ensure_ascii=False)}}]})
            messages.append({"role": "tool", "tool_call_id": f"tc_{tidx}", "content": ret})

        # 调 LLM 生成最终回答
        t0 = time.time()
        try:
            llm_resp = client.chat.completions.create(
                model=MODEL_NAME, messages=messages, temperature=0.1)
            answer = llm_resp.choices[0].message.content or ""
            latency = round(time.time() - t0, 2)
        except Exception as e:
            answer = f"[ERROR] {e}"
            latency = 0

        # 判断是否通过
        if not is_known_answer:
            # 期望不编造：回答不能包含具体虚假信息
            fabricate_patterns = ["可以", "建议", "修复方法", "加固手段", "处置步骤", "方案是"]
            has_fabrication = any(p in answer for p in fabricate_patterns)
            has_honest = "暂无" in answer or "未查询到" in answer or "不存在" in answer or "知识库" in answer
            passed = has_honest or not has_fabrication
        else:
            # 期望有答案：回答不能是"暂无"
            passed = len(answer) > 20 and "暂无" not in answer and "未查询到" not in answer

        status = "✓" if passed else "✗"
        results.append({"case": i + 1, "query": query, "passed": passed,
                        "answer_len": len(answer), "latency_s": latency})
        print(f"  {status} [{i+1}] {query[:42]:<42} | 通过={passed} | 长度={len(answer)}")

    passed_count = sum(1 for r in results if r["passed"])
    total        = len(results)
    print(f"\n  幻觉测试通过率: {passed_count}/{total} = {passed_count/total*100:.0f}%")
    return results


# ══════════════════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("SecOps-Copilot 评估报告")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    recall_passed, recall_total = evaluate_recall()
    hallucination_results = evaluate_hallucination()

    print("\n" + "=" * 60)
    print("  汇总")
    print("=" * 60)
    hc_passed = sum(1 for r in hallucination_results if r["passed"])
    hc_total  = len(hallucination_results)
    recall_rate = recall_passed / recall_total if recall_total else 0
    hc_rate     = hc_passed / hc_total if hc_total else 0
    overall = (recall_rate + hc_rate) / 2 * 100
    print(f"  检索召回率:   {recall_passed}/{recall_total} ({recall_rate*100:.0f}%)")
    print(f"  幻觉抑制率:   {hc_passed}/{hc_total} ({hc_rate*100:.0f}%)")
    print(f"  综合评分:     {overall:.0f}/100")
    print("=" * 60)
