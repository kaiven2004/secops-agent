import json
import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 配置
MD_FILES = [
    "cis_benchmarks.md",
    "cloud_native_security.md",
    "incident_response.md",
    "k8s_attack_patterns.md"
]
OUTPUT_JSON = "knowledge_chunks.json"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=70,
    separators=["##", "###", "\n\n", "\n", "。", "；"]
)

def load_md(filepath: str) -> str:
    path = Path(filepath)
    text = path.read_text(encoding="utf‑8")
    # 简单清洗多余换行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def main():
    all_chunks = []
    for md_path in MD_FILES:
        print(f"读取文档：{md_path}")
        raw_text = load_md(md_path)
        chunks = splitter.split_text(raw_text)
        for idx, content in enumerate(chunks):
            all_chunks.append({
                "source_file": md_path,
                "chunk_index": idx,
                "content": content.strip()
            })
    with open(OUTPUT_JSON, "w", encoding="utf‑8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"完成，共生成 {len(all_chunks)} 条切片，输出 {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
