"""
SecOps-Copilot — 模拟 SOC 数据流水线
模拟真实 SOC 平台：持续接收告警 → Agent 自动研判 → 生成处置报告
用法: python data_pipeline.py
     python data_pipeline.py --alert-id ALERT-0815
     python data_pipeline.py --mode cve --cve-id CVE-2024-21762
"""
import os, sys, json, time, argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from agent_main import (
    load_infra, Coordinator, AgentContext, AgentResult,
    AlertAnalystAgent, CVEResearcherAgent, KnowledgeAgent, ReportAgent,
    ALERTS_PATH, CVE_DB_PATH,
)


class DataPipeline:
    """
    模拟 SOC 数据流水线：
    1. DataIngestion — 模拟告警/CVE 数据接入
    2. AgentProcessing — 多 Agent 并发处理
    3. ReportOutput — 结构化输出
    """

    def __init__(self, coordinator: Coordinator):
        self.coord = coordinator
        self.alerts  = coordinator.alerts
        self.cve_db  = coordinator.cve_db
        self.pipeline_log: list = []

    def process_alert(self, alert_id: str) -> dict:
        """处理单条告警，返回结构化结果。"""
        t0 = time.time()
        alert = self.alerts.get(alert_id)
        if not alert:
            return {"status": "not_found", "alert_id": alert_id, "error": f"告警 {alert_id} 不存在"}

        # 构建查询
        query = f"分析告警{alert_id}，给出风险研判和处置建议"

        # 调用 Coordinator
        report = self.coord.run(query)
        elapsed = round(time.time() - t0, 1)

        result = {
            "pipeline_stage": "alert_processing",
            "alert_id": alert_id,
            "alert_type": alert.get("type"),
            "alert_level": alert.get("level"),
            "processing_time_s": elapsed,
            "report": report,
            "report_length": len(report),
            "timestamp": datetime.now().isoformat(),
            "status": "success",
        }
        self.pipeline_log.append(result)
        return result

    def process_cve(self, cve_id: str) -> dict:
        """处理单条 CVE，返回结构化结果。"""
        t0 = time.time()
        cve = self.cve_db.get(cve_id.upper())
        if not cve:
            return {"status": "not_found", "cve_id": cve_id, "error": f"CVE {cve_id} 不存在"}

        query = f"{cve_id} 漏洞详情和修复方案"
        report = self.coord.run(query)
        elapsed = round(time.time() - t0, 1)

        result = {
            "pipeline_stage": "cve_processing",
            "cve_id": cve_id,
            "severity": cve.get("severity"),
            "product": cve.get("product"),
            "processing_time_s": elapsed,
            "report": report,
            "report_length": len(report),
            "timestamp": datetime.now().isoformat(),
            "status": "success",
        }
        self.pipeline_log.append(result)
        return result

    def batch_process(self, item_ids: list, item_type: str = "alert") -> list:
        """批量处理多条数据。"""
        results = []
        for item_id in item_ids:
            if item_type == "alert":
                r = self.process_alert(item_id)
            else:
                r = self.process_cve(item_id)
            results.append(r)
            status = "✓" if r["status"] == "success" else "✗"
            print(f"  {status} {item_id} | {r.get('processing_time_s', 0)}s")
        return results

    def print_summary(self):
        """打印流水线执行摘要。"""
        print(f"\n{'='*60}")
        print(f"  流水线执行摘要")
        print(f"{'='*60}")
        total = len(self.pipeline_log)
        success = sum(1 for r in self.pipeline_log if r["status"] == "success")
        avg_time = sum(r.get("processing_time_s", 0) for r in self.pipeline_log) / total if total else 0
        total_chars = sum(r.get("report_length", 0) for r in self.pipeline_log)
        print(f"  处理总数:   {total}")
        print(f"  成功:       {success}")
        print(f"  成功率:     {success/total*100:.0f}%")
        print(f"  平均耗时:   {avg_time:.1f}s")
        print(f"  报告总字数: {total_chars}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="SecOps-Copilot 数据流水线")
    parser.add_argument("--mode", choices=["alert", "cve", "batch"], default="alert",
                        help="处理模式")
    parser.add_argument("--alert-id", help="指定告警ID")
    parser.add_argument("--cve-id", help="指定CVE编号")
    parser.add_argument("--batch-alerts", nargs="*", help="批量处理告警ID列表")
    parser.add_argument("--batch-cves", nargs="*", help="批量处理CVE编号列表")
    args = parser.parse_args()

    index, texts, tokenizer, model, alerts, cve_db, client = load_infra()
    coord = Coordinator(client, alerts, cve_db, index, texts, tokenizer, model)
    pipeline = DataPipeline(coord)

    if args.mode == "alert" and args.alert_id:
        print(f"\n[流水线] 处理告警: {args.alert_id}")
        result = pipeline.process_alert(args.alert_id)
        print(f"状态: {result['status']} | 耗时: {result.get('processing_time_s', 0)}s")
        if result["status"] == "success":
            print(f"报告:\n{result['report'][:500]}...")

    elif args.mode == "cve" and args.cve_id:
        print(f"\n[流水线] 处理CVE: {args.cve_id}")
        result = pipeline.process_cve(args.cve_id)
        print(f"状态: {result['status']} | 耗时: {result.get('processing_time_s', 0)}s")
        if result["status"] == "success":
            print(f"报告:\n{result['report'][:500]}...")

    elif args.mode == "batch":
        batch_alerts = args.batch_alerts or list(alerts.keys())[:3]
        batch_cves = args.batch_cves or list(cve_db.keys())[:2]
        print(f"\n[流水线] 批量处理告警: {batch_alerts}")
        pipeline.batch_process(batch_alerts, "alert")
        print(f"\n[流水线] 批量处理CVE: {batch_cves}")
        pipeline.batch_process(batch_cves, "cve")
        pipeline.print_summary()

    else:
        # 默认：处理 ALERT-0815
        print("\n[流水线] 默认处理告警 ALERT-0815")
        result = pipeline.process_alert("ALERT-0815")
        print(f"状态: {result['status']} | 耗时: {result.get('processing_time_s', 0)}s")
        if result["status"] == "success":
            print(f"\n{result['report']}")


if __name__ == "__main__":
    main()
