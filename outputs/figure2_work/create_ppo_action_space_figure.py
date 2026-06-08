from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "\u4e13\u5229" / "PPO" / "\u9644\u56fe"
FIG_NAME = "\u56fe2_\u672c\u53d1\u660e\u8fd1\u7aef\u7b56\u7565\u4f18\u5316\u667a\u80fd\u591a\u6b65\u51b3\u7b56\u52a8\u4f5c\u7a7a\u95f4\u793a\u610f\u56fe.png"
SVG_NAME = "\u56fe2_\u672c\u53d1\u660e\u8fd1\u7aef\u7b56\u7565\u4f18\u5316\u667a\u80fd\u591a\u6b65\u51b3\u7b56\u52a8\u4f5c\u7a7a\u95f4\u793a\u610f\u56fe.svg"
DOCX_NAME = "\u8bf4\u660e\u4e66\u9644\u56fe.docx"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/simhei.ttf") if bold else Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def centered_text(
    draw: ImageDraw.ImageDraw,
    xywh: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: str = "#111111",
) -> None:
    x, y, w, h = xywh
    tw, th = text_size(draw, text, fnt)
    draw.text((x + (w - tw) / 2, y + (h - th) / 2 - 2), text, font=fnt, fill=fill)


def rect(
    draw: ImageDraw.ImageDraw,
    xywh: tuple[int, int, int, int],
    fill: str = "#ffffff",
    outline: str = "#111111",
    width: int = 3,
    radius: int = 0,
) -> None:
    x, y, w, h = xywh
    if radius:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill, outline=outline, width=width)
    else:
        draw.rectangle((x, y, x + w, y + h), fill=fill, outline=outline, width=width)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], width: int = 3) -> None:
    draw.line((start, end), fill="#354052", width=width)
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size = 15
    p1 = (ex - ux * size + px * size * 0.55, ey - uy * size + py * size * 0.55)
    p2 = (ex - ux * size - px * size * 0.55, ey - uy * size - py * size * 0.55)
    draw.polygon((end, p1, p2), fill="#354052")


def dashed_rect(draw: ImageDraw.ImageDraw, xywh: tuple[int, int, int, int], dash: int = 16, gap: int = 9) -> None:
    x, y, w, h = xywh
    for x0 in range(x, x + w, dash + gap):
        draw.line((x0, y, min(x0 + dash, x + w), y), fill="#111111", width=3)
        draw.line((x0, y + h, min(x0 + dash, x + w), y + h), fill="#111111", width=3)
    for y0 in range(y, y + h, dash + gap):
        draw.line((x, y0, x, min(y0 + dash, y + h)), fill="#111111", width=3)
        draw.line((x + w, y0, x + w, min(y0 + dash, y + h)), fill="#111111", width=3)


def row_boxes(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    labels: list[str],
    selected: int | None,
    label_font: ImageFont.ImageFont,
    selected_fill: str = "#dedede",
    normal_fill: str = "#f7f7f7",
) -> list[tuple[int, int, int, int]]:
    count = len(labels)
    cell_w = w // count
    boxes = []
    for i, label in enumerate(labels):
        bx = x + i * cell_w
        bw = cell_w if i < count - 1 else x + w - bx
        fill = selected_fill if selected == i else normal_fill
        rect(draw, (bx, y, bw, h), fill=fill, width=3)
        if selected == i:
            dashed_rect(draw, (bx + 6, y + 6, bw - 12, h - 12), dash=13, gap=8)
        centered_text(draw, (bx, y, bw, h), label, label_font)
        boxes.append((bx, y, bw, h))
    return boxes


def label_box(
    draw: ImageDraw.ImageDraw,
    xywh: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: str,
    outline: str = "#354052",
) -> None:
    rect(draw, xywh, fill=fill, outline=outline, width=3, radius=12)
    centered_text(draw, xywh, text, fnt, fill="#1f2937")


def write_svg(svg_path: Path) -> None:
    svg_path.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" width="2200" height="920" viewBox="0 0 2200 920">
<rect width="2200" height="920" fill="white"/>
<text x="1100" y="58" text-anchor="middle" font-family="SimSun, serif" font-size="42" font-weight="700">近端策略优化智能多步决策动作空间</text>
<text x="1100" y="113" text-anchor="middle" font-family="SimSun, serif" font-size="28">动作 a<tspan baseline-shift="sub" font-size="20">t</tspan>=(k<tspan baseline-shift="sub" font-size="20">s</tspan>,m,k<tspan baseline-shift="sub" font-size="20">d</tspan>) 表示将源目标簇中的第 m 个目标迁移至目标目标簇</text>
<g stroke="#111" stroke-width="3" fill="#f7f7f7" font-family="SimSun, serif" font-size="34">
<text x="220" y="185" font-size="28">当前目标簇集合 G={G1,G2,...,GK}</text>
<rect x="220" y="205" width="900" height="86"/><line x1="370" y1="205" x2="370" y2="291"/><line x1="520" y1="205" x2="520" y2="291"/><line x1="670" y1="205" x2="670" y2="291"/><line x1="820" y1="205" x2="820" y2="291"/><line x1="970" y1="205" x2="970" y2="291"/>
<rect x="520" y="211" width="138" height="74" fill="#dedede" stroke-dasharray="14 8"/>
<text x="295" y="260" text-anchor="middle">G1</text><text x="445" y="260" text-anchor="middle">G2</text><text x="595" y="260" text-anchor="middle">Gks</text><text x="745" y="260" text-anchor="middle">...</text><text x="895" y="260" text-anchor="middle">...</text><text x="1045" y="260" text-anchor="middle">GK</text>
<text x="220" y="390" font-size="28">源目标簇 Gks 内目标序列</text>
<rect x="310" y="410" width="760" height="86"/><line x1="462" y1="410" x2="462" y2="496"/><line x1="614" y1="410" x2="614" y2="496"/><line x1="766" y1="410" x2="766" y2="496"/><line x1="918" y1="410" x2="918" y2="496"/>
<rect x="614" y="416" width="140" height="74" fill="#dedede" stroke-dasharray="14 8"/>
<text x="386" y="465" text-anchor="middle">T1</text><text x="538" y="465" text-anchor="middle">T2</text><text x="690" y="465" text-anchor="middle">Tm</text><text x="842" y="465" text-anchor="middle">...</text><text x="994" y="465" text-anchor="middle">Tnks</text>
<text x="220" y="605" font-size="28">候选目标目标簇集合 G\\{Gks}</text>
<rect x="220" y="625" width="900" height="86"/><line x1="370" y1="625" x2="370" y2="711"/><line x1="520" y1="625" x2="520" y2="711"/><line x1="670" y1="625" x2="670" y2="711"/><line x1="820" y1="625" x2="820" y2="711"/><line x1="970" y1="625" x2="970" y2="711"/>
<rect x="520" y="631" width="138" height="74" fill="#dedede" stroke-dasharray="14 8"/>
<text x="295" y="680" text-anchor="middle">G1</text><text x="445" y="680" text-anchor="middle">G2</text><text x="595" y="680" text-anchor="middle">Gkd</text><text x="745" y="680" text-anchor="middle">...</text><text x="895" y="680" text-anchor="middle">...</text><text x="1045" y="680" text-anchor="middle">GK</text>
<path d="M595 291 L690 405" fill="none" stroke-width="4" marker-end="url(#arrow)"/>
<path d="M690 496 L595 621" fill="none" stroke-width="4" marker-end="url(#arrow)"/>
<text x="705" y="360" font-size="26">选择目标 Tm</text>
<text x="765" y="570" font-size="26">选择目标目标簇</text>
<line x1="520" y1="303" x2="670" y2="303" stroke="#555" stroke-width="2"/>
<line x1="520" y1="613" x2="670" y2="613" stroke="#555" stroke-width="2"/>
</g>
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#111"/></marker></defs>
<g font-family="SimSun, serif" fill="#111">
<rect x="1290" y="205" width="650" height="506" fill="white" stroke="#111" stroke-width="3"/>
<text x="1615" y="265" text-anchor="middle" font-size="34" font-weight="700">动作掩码与多步决策</text>
<line x1="1325" y1="290" x2="1905" y2="290" stroke="#111" stroke-width="2"/>
<text x="1345" y="340" font-size="30">合法动作约束：</text>
<text x="1380" y="390" font-size="30">ks ≠ kd</text>
<text x="1380" y="438" font-size="30">1 ≤ ks,kd ≤ K</text>
<text x="1380" y="486" font-size="30">1 ≤ m ≤ nks</text>
<text x="1345" y="550" font-size="30">单步动作：a<tspan baseline-shift="sub" font-size="22">t</tspan>=(ks,m,kd)</text>
<text x="1345" y="602" font-size="30">多步序列：A={a1,a2,...,aL}</text>
<text x="1345" y="654" font-size="30">直到达到终止条件</text>
<path d="M1120 248 L1285 330M1070 453 L1285 525M1120 668 L1285 580" fill="none" stroke="#111" stroke-width="3" marker-end="url(#arrow)"/>
</g>
<text x="1100" y="850" text-anchor="middle" font-family="SimSun, serif" font-size="30">由当前状态输入 PPO 策略网络，依次选择源簇、目标和目标簇，形成连续目标迁移动作</text>
</svg>
""",
        encoding="utf-8",
    )


def build_png(path: Path) -> None:
    w, h = 2200, 920
    img = Image.new("RGB", (w, h), "#fbfbf8")
    d = ImageDraw.Draw(img)
    f_label = font(33, bold=True)
    f_box = font(38)
    f_note = font(29)
    f_right = font(32, bold=True)
    f_math = font(33)

    # Left-side compact labels, separated from the rows so no text overlaps arrows or cells.
    label_box(d, (150, 145, 210, 78), "当前簇", f_label, "#f4d97b")
    top = row_boxes(d, 420, 145, 830, 78, ["G1", "G2", "Gks", "...", "GK"], 2, f_box, selected_fill="#f6b36b", normal_fill="#fff1c7")

    label_box(d, (150, 390, 210, 78), "源簇目标", f_label, "#9fd0f0")
    mid = row_boxes(d, 420, 390, 830, 78, ["T1", "T2", "Tm", "...", "Tnks"], 2, f_box, selected_fill="#7cc0eb", normal_fill="#d8ecfb")

    label_box(d, (150, 635, 210, 78), "候选簇", f_label, "#a9dc95")
    bot = row_boxes(d, 420, 635, 830, 78, ["G1", "G2", "Gkd", "...", "GK"], 2, f_box, selected_fill="#85ca71", normal_fill="#dff1d8")

    top_src = (top[2][0] + top[2][2] // 2, top[2][1] + top[2][3])
    mid_t_top = (mid[2][0] + mid[2][2] // 2, mid[2][1] - 6)
    arrow(d, top_src, mid_t_top, width=4)
    centered_text(d, (910, 292, 180, 46), "选 Tm", f_note)

    mid_t = (mid[2][0] + mid[2][2] // 2, mid[2][1] + mid[2][3])
    bot_d_top = (bot[2][0] + bot[2][2] // 2, bot[2][1] - 6)
    arrow(d, mid_t, bot_d_top, width=4)
    centered_text(d, (910, 540, 180, 46), "选 Gkd", f_note)

    # Light guide lines indicate the selected source cluster and destination set without
    # cluttering the patent line drawing.
    d.line((top[2][0], top[2][1] + top[2][3] + 9, top[2][0] + top[2][2], top[2][1] + top[2][3] + 9), fill="#354052", width=2)
    d.line((bot[2][0], bot[2][1] - 10, bot[2][0] + bot[2][2], bot[2][1] - 10), fill="#354052", width=2)

    right = (1390, 185, 620, 490)
    rect(d, right, fill="#ffffff", outline="#354052", width=3, radius=16)
    lines = [
        (1450, 245, "a_t=(ks,m,kd)", f_right),
        (1450, 325, "ks \u2260 kd", f_math),
        (1450, 385, "1 \u2264 ks,kd \u2264 K", f_math),
        (1450, 445, "1 \u2264 m \u2264 nks", f_math),
        (1450, 545, "A={a1,a2,...,aL}", f_math),
        (1450, 605, "\u52a8\u4f5c\u63a9\u7801", f_math),
    ]
    for x, y, text, fnt in lines:
        d.text((x, y), text, font=fnt, fill="#1f2937")

    arrow(d, (1250, 184), (1386, 282), width=3)
    arrow(d, (1250, 429), (1386, 430), width=3)
    arrow(d, (1250, 674), (1386, 575), width=3)

    img.save(path, dpi=(300, 300))


def update_docx_with_figure(fig_path: Path) -> Path:
    docx = ROOT / "\u4e13\u5229" / "PPO" / DOCX_NAME
    backup = docx.with_name("\u8bf4\u660e\u4e66\u9644\u56fe_\u56fe2\u66ff\u6362\u524d\u5907\u4efd.docx")
    if not backup.exists():
        shutil.copy2(docx, backup)

    tmp = docx.with_suffix(".tmp.docx")
    old_caption = "\u56fe2 \u672c\u53d1\u660e\u667a\u80fd\u591a\u6b65\u51b3\u7b56\u52a8\u4f5c\u7a7a\u95f4\u793a\u610f\u56fe"
    new_caption = "\u56fe2 \u672c\u53d1\u660e\u8fd1\u7aef\u7b56\u7565\u4f18\u5316\u667a\u80fd\u591a\u6b65\u51b3\u7b56\u52a8\u4f5c\u7a7a\u95f4\u793a\u610f\u56fe"
    with zipfile.ZipFile(docx, "r") as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/media/image2.png":
                data = fig_path.read_bytes()
            elif item.filename == "word/document.xml":
                data = data.replace(old_caption.encode("utf-8"), new_caption.encode("utf-8"))
                data = data.replace(
                    "\u667a\u80fd\u591a\u6b65\u51b3\u7b56\u52a8\u4f5c\u7a7a\u95f4\u793a\u610f\u56fe".encode("utf-8"),
                    "\u8fd1\u7aef\u7b56\u7565\u4f18\u5316\u667a\u80fd\u591a\u6b65\u51b3\u7b56\u52a8\u4f5c\u7a7a\u95f4\u793a\u610f\u56fe".encode("utf-8"),
                    1,
                )
            zout.writestr(item, data)
    try:
        tmp.replace(docx)
    except PermissionError:
        optimized = docx.with_name("\u8bf4\u660e\u4e66\u9644\u56fe_\u56fe2\u4f18\u5316\u7248.docx")
        if optimized.exists():
            optimized.unlink()
        tmp.replace(optimized)
    return backup


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = OUT_DIR / FIG_NAME
    svg_path = OUT_DIR / SVG_NAME
    build_png(fig_path)
    write_svg(svg_path)
    backup = update_docx_with_figure(fig_path)
    print(fig_path)
    print(svg_path)
    print(backup)


if __name__ == "__main__":
    main()
