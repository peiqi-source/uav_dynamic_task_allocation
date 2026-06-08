from __future__ import annotations

import copy
import shutil
import zipfile
from pathlib import Path

from lxml import etree
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "\u4e13\u5229" / "PPO" / "\u9644\u56fe"
FIG_PATH = FIG_DIR / "\u56fe3_\u672c\u53d1\u660e\u8fd1\u7aef\u7b56\u7565\u4f18\u5316\u52a8\u6001\u91cd\u5206\u7fa4\u8bad\u7ec3\u4e0e\u63a8\u7406\u6d41\u7a0b\u56fe.png"
DOCX_IN = ROOT / "\u4e13\u5229" / "PPO" / "\u8bf4\u660e\u4e66\u9644\u56fe_\u56fe2\u4f18\u5316\u7248.docx"
DOCX_OUT = ROOT / "\u4e13\u5229" / "PPO" / "\u8bf4\u660e\u4e66\u9644\u56fe_\u56fe2\u56fe3\u4f18\u5316\u7248.docx"

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    choices = [
        Path("C:/Windows/Fonts/simhei.ttf") if bold else Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for item in choices:
        if item.exists():
            return ImageFont.truetype(str(item), size=size)
    return ImageFont.load_default()


def text_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fnt: ImageFont.ImageFont, fill="#111827") -> None:
    x, y, w, h = box
    lines = text.split("\n")
    sizes = [draw.textbbox((0, 0), line, font=fnt) for line in lines]
    heights = [b[3] - b[1] for b in sizes]
    total = sum(heights) + (len(lines) - 1) * 8
    cy = y + (h - total) / 2 - 2
    for line, b, lh in zip(lines, sizes, heights):
        tw = b[2] - b[0]
        draw.text((x + (w - tw) / 2, cy), line, font=fnt, fill=fill)
        cy += lh + 8


def rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline="#111827", width=4, radius=0) -> None:
    x, y, w, h = box
    if radius:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill, outline=outline, width=width)
    else:
        draw.rectangle((x, y, x + w, y + h), fill=fill, outline=outline, width=width)


def diamond(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fnt: ImageFont.ImageFont, fill="#fff8d6") -> None:
    x, y, w, h = box
    pts = [(x + w // 2, y), (x + w, y + h // 2), (x + w // 2, y + h), (x, y + h // 2)]
    draw.polygon(pts, fill=fill, outline="#111827")
    draw.line(pts + [pts[0]], fill="#111827", width=4)
    text_box(draw, box, text, fnt)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], width=4) -> None:
    draw.line((start, end), fill="#111827", width=width)
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size = 18
    p1 = (ex - ux * size + px * size * 0.55, ey - uy * size + py * size * 0.55)
    p2 = (ex - ux * size - px * size * 0.55, ey - uy * size - py * size * 0.55)
    draw.polygon([end, p1, p2], fill="#111827")


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt: ImageFont.ImageFont) -> None:
    draw.text(xy, text, font=fnt, fill="#111827")


def build_figure() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    w, h = 1900, 1260
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    f_stage = font(36, bold=True)
    f_node = font(32)
    f_small = font(28)

    # Stage bands keep the same process-flow readability as Figure 1 while separating
    # training from inference.
    rect(d, (95, 70, 770, 1040), "#eef5ff", "#2f5597", width=4, radius=20)
    rect(d, (1035, 70, 770, 1040), "#f0f8ed", "#548235", width=4, radius=20)
    text_box(d, (95, 88, 770, 52), "训练阶段", f_stage, "#1f4e79")
    text_box(d, (1035, 88, 770, 52), "推理阶段", f_stage, "#375623")

    left_nodes = [
        (205, 175, 550, 90, "真实目标数据\n动态事件场景"),
        (205, 315, 550, 90, "构建目标分群环境\n初始化PSO分群"),
        (205, 455, 550, 90, "状态 St 输入PPO智能体"),
        (205, 595, 550, 90, "输出动作 at\n环境执行并反馈 Rt、St+1"),
        (205, 735, 550, 90, "采集轨迹样本\n(St,at,Rt,St+1)"),
        (205, 875, 550, 90, "裁剪目标函数更新\n策略网络和价值网络"),
    ]
    for node in left_nodes:
        rect(d, node[:4], "#ffffff", "#111827", width=4)
        text_box(d, node[:4], node[4], f_node)
    for a, b in zip(left_nodes, left_nodes[1:]):
        arrow(d, (a[0] + a[2] // 2, a[1] + a[3]), (b[0] + b[2] // 2, b[1] - 5))

    # Training loop.
    arrow(d, (755, 920), (815, 920))
    d.line((815, 920, 815, 500, 755, 500), fill="#111827", width=4)
    arrow(d, (815, 500), (755, 500))
    label(d, (770, 470), "迭代", f_small)

    model_box = (360, 1150, 460, 78)
    rect(d, model_box, "#fff2cc", "#806000", width=4, radius=14)
    text_box(d, model_box, "训练后PPO策略模型", f_node, "#4f3b00")
    arrow(d, (480, 965), (590, 1150))

    right_nodes = [
        (1145, 175, 550, 90, "加载训练后策略模型"),
        (1145, 315, 550, 90, "以PSO初始分群\n作为推理起点"),
        (1145, 455, 550, 90, "构建当前状态矩阵\n生成动作掩码"),
        (1145, 595, 550, 90, "选择合法目标移动动作"),
        (1145, 735, 550, 90, "更新目标簇标签\n重算簇中心与负载"),
    ]
    for node in right_nodes:
        rect(d, node[:4], "#ffffff", "#111827", width=4)
        text_box(d, node[:4], node[4], f_node)
    for a, b in zip(right_nodes, right_nodes[1:]):
        arrow(d, (a[0] + a[2] // 2, a[1] + a[3]), (b[0] + b[2] // 2, b[1] - 5))

    dec = (1170, 880, 500, 130)
    diamond(d, dec, "是否满足\n终止条件", f_node, "#fef3d0")
    arrow(d, (1420, 825), (1420, 880))

    # No branch loops back to action mask.
    d.line((1170, 945, 1090, 945, 1090, 500, 1145, 500), fill="#111827", width=4)
    arrow(d, (1090, 500), (1145, 500))
    label(d, (1105, 910), "否", f_small)

    out = (1145, 1148, 550, 80)
    rect(d, out, "#e2f0d9", "#548235", width=4)
    text_box(d, out, "输出动态重分群目标簇集合", f_node, "#375623")
    arrow(d, (1420, 1010), (1420, 1148))
    label(d, (1440, 1045), "是", f_small)

    img.save(FIG_PATH, dpi=(300, 300))


def next_rid(rels_root: etree._Element) -> str:
    nums = []
    for rel in rels_root:
        rid = rel.get("Id", "")
        if rid.startswith("rId") and rid[3:].isdigit():
            nums.append(int(rid[3:]))
    return f"rId{max(nums, default=0) + 1}"


def insert_into_docx() -> None:
    if not DOCX_IN.exists():
        raise FileNotFoundError(DOCX_IN)
    if DOCX_OUT.exists():
        DOCX_OUT.unlink()
    tmp = DOCX_OUT.with_suffix(".tmp.docx")

    with zipfile.ZipFile(DOCX_IN, "r") as zin:
        doc_xml = zin.read("word/document.xml")
        rels_xml = zin.read("word/_rels/document.xml.rels")
        root = etree.fromstring(doc_xml)
        rels_root = etree.fromstring(rels_xml)

        rid = next_rid(rels_root)
        rel = etree.Element("Relationship")
        rel.set("Id", rid)
        rel.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
        rel.set("Target", "media/image5.png")
        rels_root.append(rel)

        body = root.find("w:body", NS)
        paragraphs = body.findall("w:p", NS)
        fig3_idx = None
        template_p = None
        for idx, p in enumerate(paragraphs):
            text = "".join(p.xpath(".//w:t/text()", namespaces=NS))
            if "\u56fe3" in text:
                fig3_idx = idx
                p.xpath(".//w:t", namespaces=NS)[0].text = "\u56fe3 \u672c\u53d1\u660e\u8fd1\u7aef\u7b56\u7565\u4f18\u5316\u52a8\u6001\u91cd\u5206\u7fa4\u8bad\u7ec3\u4e0e\u63a8\u7406\u6d41\u7a0b\u56fe"
                for t in p.xpath(".//w:t", namespaces=NS)[1:]:
                    t.text = ""
            if p.xpath(".//a:blip", namespaces=NS) and template_p is None:
                template_p = p
        if fig3_idx is None or template_p is None:
            raise RuntimeError("Could not locate Figure 3 caption or image template paragraph.")

        new_p = copy.deepcopy(template_p)
        for blip in new_p.xpath(".//a:blip", namespaces=NS):
            blip.set(f"{{{NS['r']}}}embed", rid)
        for extent in new_p.xpath(".//wp:extent", namespaces=NS):
            extent.set("cx", "6000000")
            extent.set("cy", "3978947")
        for ext in new_p.xpath(".//a:ext", namespaces=NS):
            ext.set("cx", "6000000")
            ext.set("cy", "3978947")
        for docpr in new_p.xpath(".//wp:docPr", namespaces=NS):
            docpr.set("id", "3003")
            docpr.set("name", "\u56fe3_\u8bad\u7ec3\u4e0e\u63a8\u7406\u6d41\u7a0b\u56fe")

        # If a previous inserted figure exists after the caption, replace it; otherwise insert.
        next_p = paragraphs[fig3_idx + 1] if fig3_idx + 1 < len(paragraphs) else None
        if next_p is not None and next_p.xpath(".//a:blip", namespaces=NS):
            body.replace(next_p, new_p)
        else:
            body.insert(fig3_idx + 1, new_p)

        doc_xml_new = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")
        rels_xml_new = etree.tostring(rels_root, xml_declaration=True, encoding="UTF-8", standalone="yes")

        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == "word/document.xml":
                    zout.writestr(item, doc_xml_new)
                elif item.filename == "word/_rels/document.xml.rels":
                    zout.writestr(item, rels_xml_new)
                else:
                    zout.writestr(item, zin.read(item.filename))
            zout.writestr("word/media/image5.png", FIG_PATH.read_bytes())
    tmp.replace(DOCX_OUT)


def main() -> None:
    build_figure()
    insert_into_docx()
    print(FIG_PATH)
    print(DOCX_OUT)


if __name__ == "__main__":
    main()
