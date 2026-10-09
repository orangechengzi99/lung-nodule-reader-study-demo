"""Font-independent, readable fallback; never modify numerical result tables."""
import contextlib
from dataclasses import asdict
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import warnings

from tests.helpers import load_script, study_tables
import matplotlib
from matplotlib.figure import Figure
from matplotlib.text import Text
import pandas as pd
import plot_style as P
import run_all as R


class PlotStyleTests(unittest.TestCase):
    def test_auto_without_chinese_font_uses_english(self):
        with matplotlib.rc_context(), patch.object(P, "find_chinese_font", return_value=None):
            style = P.configure_plotting("auto")
            self.assertEqual(style.language, "en")
            self.assertEqual(style.font_family, "DejaVu Sans")
            self.assertIsNone(style.font_file)
            self.assertEqual(style.text("结节级敏感度"), "Lesion sensitivity")

    def test_forced_english_does_not_search_system_fonts(self):
        with matplotlib.rc_context(), patch.object(P, "find_chinese_font") as find_font:
            style = P.configure_plotting("en")
        find_font.assert_not_called()
        self.assertEqual(style.language, "en")

    def test_forced_chinese_without_font_is_readable_error(self):
        with patch.object(P, "find_chinese_font", return_value=None):
            with self.assertRaisesRegex(ValueError, "没有可用中文绘图字体"):
                P.configure_plotting("zh")

    def test_invalid_language_is_rejected(self):
        with self.assertRaises(ValueError):
            P.configure_plotting("invalid")

    def test_available_chinese_font_is_registered_and_preserves_text(self):
        with matplotlib.rc_context(), patch.object(P, "find_chinese_font", return_value=("Test CJK", "fixture.ttf")), patch.object(P.font_manager.fontManager, "addfont") as register:
            style = P.configure_plotting("auto")
            self.assertEqual(matplotlib.rcParams["font.family"], ["Test CJK"])
        register.assert_called_once_with("fixture.ttf")
        self.assertEqual(style.language, "zh")
        self.assertEqual(style.text("结节级敏感度"), "结节级敏感度")

    def test_glyph_coverage_requires_every_plot_character(self):
        coverage = {ord(character): 1 for character in P.REQUIRED_GLYPHS}
        with patch.object(P, "FT2Font") as font:
            font.return_value.get_charmap.return_value = coverage
            self.assertTrue(P.has_required_glyphs("fixture.ttf"))
            del coverage[ord("结")]
            self.assertFalse(P.has_required_glyphs("fixture.ttf"))

    def test_unreadable_font_is_ignored(self):
        with patch.object(P, "FT2Font", side_effect=OSError("fixture")):
            self.assertFalse(P.has_required_glyphs("fixture.ttf"))

    def test_unknown_chinese_label_cannot_silently_render_boxes(self):
        style = P.PlotStyle("auto", "en", "DejaVu Sans", None)
        with self.assertRaisesRegex(ValueError, "Missing English"):
            style.text("未配置的中文标签")

    def test_all_project_translations_are_ascii(self):
        style = P.PlotStyle("auto", "en", "DejaVu Sans", None)
        for original in P.TRANSLATIONS:
            with self.subTest(label=original):
                self.assertTrue(style.text(original).isascii())

    def test_structured_log_records_actual_selection(self):
        style = P.PlotStyle("auto", "en", "DejaVu Sans", None)
        stream = StringIO()
        with contextlib.redirect_stdout(stream):
            P.print_plot_style(style)
        self.assertEqual(json.loads(stream.getvalue().split("=", 1)[1]), asdict(style))

    def test_cli_language_validation_happens_before_running(self):
        with patch.object(R, "run_pipeline") as pipeline, contextlib.redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as failure:
                R.main(["--plot-language", "invalid"])
        self.assertEqual(failure.exception.code, 2)
        pipeline.assert_not_called()

    def test_runner_passes_language_and_records_actual_style(self):
        selection = asdict(P.PlotStyle("en", "en", "DejaVu Sans", None))

        def fake_run(command, **kwargs):
            self.assertEqual(kwargs["env"]["READER_STUDY_PLOT_LANGUAGE"], "en")
            kwargs["stdout"].write("PLOT_STYLE_JSON=" + json.dumps(selection) + "\n")
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as temporary:
            paths = R.RunPaths.isolated(temporary)
            logs = paths.root / "logs"
            logs.mkdir()
            with patch.object(R.subprocess, "run", side_effect=fake_run):
                step = R.execute_step(R.SCRIPT_DIR / R.SCRIPTS[1], paths, R.StudyParameters(), logs, "en")
        self.assertEqual(step["plot_style"], selection)

    def test_actual_six_fallback_figures_have_no_chinese_or_missing_glyphs(self):
        descriptive = load_script("02_mrmc_descriptive.py")
        froc = load_script("03_lesion_level_froc.py")
        inference = load_script("04_mrmc_inference_teaching.py")
        readers, cases, data = study_tables()
        reader_summary = descriptive.build_reader_condition_summary(readers, cases, data)
        paired = descriptive.build_paired_differences(reader_summary)
        overall = descriptive.build_overall_summary(reader_summary, paired)
        summary = pd.DataFrame({"metric_cn": ["结节级敏感度", "FP/scan", "阅片时间"],
                                "difference": [0.1, 0.1, -0.1], "ci_lower": [0, 0, -0.2],
                                "ci_upper": [0.2, 0.2, 0], "exploratory_p_value": [0.04, 0.04, 0.04]})
        original_save = Figure.savefig
        saved = []

        def checked_save(figure, path, *args, **kwargs):
            for text in figure.findobj(Text):
                self.assertFalse(any("\u4e00" <= character <= "\u9fff" for character in text.get_text()), text.get_text())
            saved.append(Path(path))
            return original_save(figure, path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary, matplotlib.rc_context(), patch.object(P, "find_chinese_font", return_value=None):
            root = Path(temporary)
            style = P.configure_plotting("auto")
            overall.to_csv(root / "overall_descriptive_summary.csv", index=False)
            before = data.copy(deep=True)
            with patch.object(descriptive, "PLOT_STYLE", style), patch.object(froc, "PLOT_STYLE", style), patch.object(inference, "PLOT_STYLE", style), patch.object(descriptive, "OUTPUT_DIR", root), patch.object(froc, "OUTPUT_DIR", root), patch.object(inference, "OUTPUT_DIR", root), patch.object(Figure, "savefig", autospec=True, side_effect=checked_save), warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter("always")
                for metric, label, title, filename in (
                    ("lesion_sensitivity", "结节级敏感度", "阅片者结节级敏感度：无AI与AI辅助", "sensitivity.png"),
                    ("fp_per_scan", "平均每例假阳性数（FP/scan）", "阅片者平均每例假阳性数：无AI与AI辅助", "fp.png"),
                    ("mean_reading_time_min", "平均阅片时间（分钟）", "阅片者平均阅片时间：无AI与AI辅助", "time.png"),
                ):
                    descriptive.save_paired_plot(paired, metric, label, title, filename)
                froc.save_froc_plot(pd.DataFrame({"FP_per_scan": [0, 1], "lesion_sensitivity": [0, 1]}))
                inference.plot_confidence_intervals(summary, root / "intervals.png")
                inference.plot_workbook_summary_preview(root / "preview.png")
            self.assertEqual(len(saved), 6)
            self.assertTrue(all(path.is_file() for path in saved))
            self.assertFalse([str(item.message) for item in emitted if "Glyph" in str(item.message)])
            pd.testing.assert_frame_equal(data, before)


if __name__ == "__main__":
    unittest.main()
