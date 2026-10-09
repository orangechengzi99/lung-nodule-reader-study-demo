"""Readable plots without downloading or redistributing system fonts."""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

import matplotlib
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font

TRANSLATIONS = {
    "无AI": "Without AI",
    "AI辅助": "With AI",
    "结节级敏感度": "Lesion sensitivity",
    "平均每例假阳性数（FP/scan）": "False positives per scan (FP/scan)",
    "平均阅片时间（分钟）": "Mean reading time (min)",
    "阅片者结节级敏感度：无AI与AI辅助": "Reader lesion sensitivity: without vs with AI",
    "阅片者平均每例假阳性数：无AI与AI辅助": "Reader false positives per scan: without vs with AI",
    "阅片者平均阅片时间：无AI与AI辅助": "Reader reading time: without vs with AI",
    "结节级FROC教学曲线": "Lesion-level FROC (teaching example)",
    "MRMC教学性推断：AI辅助减无AI": "MRMC teaching inference: with AI minus without AI",
    "敏感度差异": "Sensitivity difference",
    "FP/scan差异": "FP/scan difference",
    "阅片时间差异（分钟）": "Reading time difference (min)",
    "阅片时间": "Reading time",
    "虚构模拟数据；双重Bootstrap教学演示，不作为正式临床证据":
        "Simulated data; two-way bootstrap teaching example, not clinical evidence",
}
REQUIRED_GLYPHS = set("".join(TRANSLATIONS) + "：差异，探索性～")
CJK_FAMILIES = {
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans SC",
    "Source Han Sans SC", "PingFang SC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
}


def has_required_glyphs(path):
    try:
        character_map = FT2Font(str(path)).get_charmap()
        return all(ord(character) in character_map for character in REQUIRED_GLYPHS)
    except (OSError, RuntimeError, ValueError):
        return False


def find_chinese_font():
    preferred = Path("C:/Windows/Fonts/msyh.ttc")
    candidates = [preferred] if preferred.is_file() else []
    candidates.extend(Path(font.fname) for font in font_manager.fontManager.ttflist if font.name in CJK_FAMILIES)
    for path in dict.fromkeys(candidates):
        if has_required_glyphs(path):
            return font_manager.FontProperties(fname=str(path)).get_name(), str(path)
    return None


@dataclass(frozen=True)
class PlotStyle:
    requested_language: str
    language: str
    font_family: str
    font_file: str | None

    def text(self, value):
        if self.language == "zh":
            return value
        translated = TRANSLATIONS.get(value, value)
        if any("\u4e00" <= character <= "\u9fff" for character in translated):
            raise ValueError(f"Missing English plot label: {value}")
        return translated


def configure_plotting(requested=None):
    requested = requested or os.environ.get("READER_STUDY_PLOT_LANGUAGE", "auto")
    if requested not in {"auto", "zh", "en"}:
        raise ValueError("plot language must be auto, zh or en")
    chinese_font = None if requested == "en" else find_chinese_font()
    if requested == "zh" and chinese_font is None:
        raise ValueError("没有可用中文绘图字体。请使用 --plot-language auto 或 en，或安装支持中文的字体。")
    if chinese_font is None:
        style = PlotStyle(requested, "en", "DejaVu Sans", None)
    else:
        family, path = chinese_font
        font_manager.fontManager.addfont(path)
        style = PlotStyle(requested, "zh", family, path)
    matplotlib.rcParams["font.family"] = style.font_family
    matplotlib.rcParams["axes.unicode_minus"] = False
    return style


def print_plot_style(style):
    # A structured log line lets the runner record the actually selected font.
    print("PLOT_STYLE_JSON=" + json.dumps(asdict(style), ensure_ascii=False), flush=True)
