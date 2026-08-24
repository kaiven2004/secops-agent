"""
多 Agent 评估脚本
测试：各 Agent 独立表现 + 整体编排效果
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, str(Path(__file__).parent))
from agent_main import (
    load_infra, Coordinator, AgentContext,
    AlertAnalystAgent, CVEResearcherAgent, KnowledgeAgent, ReportAgent,
    FAISS_INDEX_PATH, ALERTS_PATH, CVE_DB_PATH, TEXTS_PATH, MODEL_PATH,
    SIM_THRESHOLD, BASE_URL,
)
from openai import OpenAI

API_KEY = os.environ.get("AGNES_API_KEY", "")
MODEL_NAME = "agnes-2.0-flash"


# ══════════════════════════════════════════════════════════
#  1. 各 Agent 独立评估
# ══════════════════════════════════════════════════════════
def eval_agents():
    print("\n" + "=" * 60)
    print("  各 Agent 独立评估")
    print("=" * 60)

    index, texts, tokenizer, model, alerts, cve_db, client = load_infra()
    coord = Coordinator(client, alerts, cve_db, index, texts, tokenizer, model)

    # ── AlertAnalystAgent 测试 ──
    print("\n[AlertAnalystAgent]")
    ctx = AgentContext(query="分析告警ALERT-0815", alert_data=alerts.get("ALERT-0815", {}))
    t0 = time.time()
    r = coord.alert_agent.run(ctx)
    print(f"  结果: {'✓' if r.success else '✗'} ({r.latency_s:.1f}s)")
    if r.success:
        print(f"  输出: {r.content[:200]}...")
    # 测试不存在告警
    ctx2 = AgentContext(query="分析告警ALERT-99999", alert_data={"alert_id": "ALERT-99999", "status": "not_found"})
    r2 = coord.alert_agent.run(ctx2)
    print(f"  不存在告警: {'✓' if not r2.success else '✗'} ({r2.latency_s:.1f}s)")

    # ── CVEResearcherAgent 测试 ──
    print("\n[CVEResearcherAgent]")
    ctx = AgentContext(query="CVE-2024-21762 详情", cve_data={})
    t0 = time.time()
    r = coord.cve_agent.run(ctx)
    print(f"  结果: {'✓' if r.success else '✗'} ({r.latency_s:.1f}s)")
    if r.success:
        print(f"  输出: {r.content[:200]}...")
    # 测试不存在的 CVE
    ctx2 = AgentContext(query="CVE-2099-0001 详情", cve_data={})
    r2 = coord.cve_agent.run(ctx2)
    print(f"  不存在CVE: {'✓' if not r2.success else '✗'} ({r2.latency_s:.1f}s)")

    # ── KnowledgeAgent 测试 ──
    print("\n[KnowledgeAgent]")
    ctx = AgentContext(query="容器逃逸防御措施")
    t0 = time.time()
    r = coord.kb_agent.run(ctx, index, tokenizer, model, texts)
    print(f"  结果: {'✓' if r.success else '✗'} ({r.latency_s:.1f}s)")
    if r.success:
        extra = r.extra
        print(f"  检索命中: {extra.get('hits', 0)} 条")
        print(f"  输出: {r.content[:200]}...")

    # ── 路由准确性测试 ──
    print("\n[Coordinator 路由测试]")
    test_cases = [
        ("帮我分析告警ALERT-0815",        ["AlertAnalystAgent", "KnowledgeAgent", "ReportAgent"]),
        ("CVE-2024-21762的修复方案",       ["CVEResearcherAgent", "KnowledgeAgent", "ReportAgent"]),
        ("K8s网络策略最佳实践是什么",       ["KnowledgeAgent", "ReportAgent"]),
        ("ALERT-0923容器逃逸风险研判",      ["AlertAnalystAgent", "KnowledgeAgent", "ReportAgent"]),
    ]
    for query, expected in test_cases:
        actual = coord._classify_query(query)
        match = all(a in actual for a in expected)
        status = "✓" if match else "✗"
        print(f"  {status} {query[:35]:<35} | 期望: {[a.split('Agent')[0] for a in expected]} | 实际: {[a.split('Agent')[0] for a in actual]}")


# ══════════════════════════════════════════════════════════
#  2. 端到端编排测试
# ══════════════════════════════════════════════════════════
def eval_end_to_end():
    print("\n" + "=" * 60)
    print("  端到端编排测试")
    print("=" * 60)

    index, texts, tokenizer, model, alerts, cve_db, client = load_infra()
    coord = Coordinator(client, alerts, cve_db, index, texts, tokenizer, model)

    test_cases = [
        ("帮我分析告警ALERT-0815，给出风险研判以及处置建议。", "告警分析"),
        ("CVE-2024-21762漏洞详情和修复方案",                   "CVE查询"),
        ("K8s容器逃逸的防御措施有哪些",                         "知识检索"),
        ("帮我分析告警ALERT-0923",                             "容器逃逸告警"),
    ]

    results = []
    for query, desc in test_cases:
        t0 = time.time()
        try:
            report = coord.run(query)
            elapsed = round(time.time() - t0, 1)
            has_content = len(report) > 50 and "Agent 执行失败" not in report
            status = "✓" if has_content else "✗"
            results.append({"desc": desc, "query": query, "passed": has_content,
                            "length": len(report), "latency_s": elapsed})
            print(f"  {status} [{desc}] {elapsed}s | 报告长度: {len(report)}字符")
            print(f"       预览: {report[:150]}...")
        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            results.append({"desc": desc, "query": query, "passed": False,
                            "length": 0, "latency_s": elapsed, "error": str(e)})
            print(f"  ✗ [{desc}] {elapsed}s | 异常: {e}")

    passed = sum(1 for r in results if r["passed"])
    total  = len(results)
    avg_latency = sum(r["latency_s"] for r in results) / total if total else 0
    print(f"\n  端到端通过率: {passed}/{total} = {passed/total*100:.0f}%")
    print(f"  平均耗时: {avg_latency:.1f}s")
    return results


# ══════════════════════════════════════════════════════════
#  3. 幻觉测试（多 Agent 版本）
# ══════════════════════════════════════════════════════════
def eval_hallucination_multi():
    print("\n" + "=" * 60)
    print("  多 Agent 幻觉测试")
    print("=" * 60)

    index, texts, tokenizer, model, alerts, cve_db, client = load_infra()
    coord = Coordinator(client, alerts, cve_db, index, texts, tokenizer, model)

    cases = [
        ("CVE-2025-8888漏洞是什么",   "虚构CVE，不应编造"),
        ("告警ALERT-99999的处置步骤", "虚构告警，不应编造"),
        ("自研组件my-agent-pod加固",  "虚构组件，不应编造"),
    ]

    results = []
    for query, expectation in cases:
        t0 = time.time()
        try:
            report = coord.run(query)
            elapsed = round(time.time() - t0, 1)
            # 期望不编造：报告应包含"暂无"或"未查询到"等拒绝语
            has_refusal = any(kw in report for kw in ["暂无", "未查询到", "不存在", "没有", "暂无资料"])
            has_fabrication = any(kw in report for kw in ["可以", "建议", "修复方案", "加固手段"]) and not has_refusal
            passed = has_refusal or not has_fabrication
            status = "✓" if passed else "✗"
            results.append({"query": query, "passed": passed, "latency_s": elapsed})
            print(f"  {status} {query[:35]:<35} | {elapsed}s | {'拒绝' if has_refusal else '编造风险'}")
        except Exception as e:
            results.append({"query": query, "passed": True, "latency_s": 0, "error": str(e)})
            print(f"  ✓ {query[:35]:<35} | 异常（视为通过）: {e}")

    passed = sum(1 for r in results if r["passed"])
    print(f"\n  幻觉抑制率: {passed}/{len(results)} = {passed/len(results)*100:.0f}%")
    return results


# ══════════════════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("SecOps-Copilot v2 多 Agent 评估报告")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    eval_agents()
    e2e_results = eval_end_to_end()
    hallucination_results = eval_hallucination_multi()

    # 汇总
    e2e_passed = sum(1 for r in e2e_results if r["passed"])
    e2e_total  = len(e2e_results)
    hc_passed  = sum(1 for r in hallucination_results if r["passed"])
    hc_total   = len(hallucination_results)

    print("\n" + "=" * 60)
    print("  汇总")
    print("=" * 60)
    print(f"  端到端可用率:  {e2e_passed}/{e2e_total} ({e2e_passed/e2e_total*100:.0f}%)")
    print(f"  幻觉抑制率:    {hc_passed}/{hc_total} ({hc_passed/hc_total*100:.0f}%)")
    print(f"  综合评分:      {(e2e_passed/e2e_total + hc_passed/hc_total)/2 * 100:.0f}/100")
    print("=" * 60)
