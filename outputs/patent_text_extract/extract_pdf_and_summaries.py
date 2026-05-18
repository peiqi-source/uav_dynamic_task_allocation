from __future__ import annotations

from pathlib import Path


ROOT = Path(r"F:\项目\专利\uav_dynamic_task_allocation")
OUT = ROOT / "outputs" / "patent_text_extract"
OUT.mkdir(parents=True, exist_ok=True)


def extract_pdf() -> None:
    pdf_path = ROOT / "专利" / "杨庆丰论文.pdf"
    if not pdf_path.exists():
        print(f"missing: {pdf_path}")
        return
    reader = None
    for module_name in ("pypdf", "PyPDF2"):
        try:
            module = __import__(module_name)
            reader = module.PdfReader(str(pdf_path))
            print(f"pdf reader: {module_name}, pages={len(reader.pages)}")
            break
        except Exception as exc:
            print(f"pdf reader unavailable {module_name}: {exc}")
    if reader is None:
        return
    text_parts: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            text = f"[page {index} extraction failed: {exc}]"
        text_parts.append(f"\n\n===== PAGE {index} =====\n{text}")
    output = OUT / "杨庆丰论文.txt"
    output.write_text("".join(text_parts), encoding="utf-8")
    print(f"wrote {output} chars={output.stat().st_size}")


def write_keyword_summary() -> None:
    sources = [
        OUT / "专利_报告.txt",
        OUT / "杨庆丰论文.txt",
        OUT / "RF摧毁目标集智能决策专利_说明书.txt",
        OUT / "RF摧毁目标集智能决策专利_权利要求书.txt",
        OUT / "基于BAB-枚举组合法的无人机多目标攻击航迹规划求解方法_说明书.txt",
        OUT / "基于BAB-枚举组合法的无人机多目标攻击航迹规划求解方法_权利要求书.txt",
    ]
    keywords = [
        "PPO",
        "近端策略优化",
        "动态",
        "目标分群",
        "区域划分",
        "粒子群",
        "PSO",
        "强化学习",
        "资源分配",
        "仿真",
    ]
    lines: list[str] = []
    for source in sources:
        if not source.exists():
            continue
        text = source.read_text(encoding="utf-8", errors="ignore")
        lines.append(f"## {source.name}")
        for keyword in keywords:
            count = text.count(keyword)
            if count:
                lines.append(f"- {keyword}: {count}")
        paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
        picked: list[str] = []
        for paragraph in paragraphs:
            if any(keyword in paragraph for keyword in keywords):
                picked.append(paragraph)
            if len(picked) >= 10:
                break
        for paragraph in picked:
            lines.append(paragraph[:600])
        lines.append("")
    output = OUT / "patent_reference_keyword_summary.txt"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output} chars={output.stat().st_size}")


if __name__ == "__main__":
    extract_pdf()
    write_keyword_summary()
