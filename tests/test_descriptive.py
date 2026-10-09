import unittest
from pathlib import Path
import re
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd

from tests.helpers import load_script, study_tables
from study_config import RunPaths

D = load_script("02_mrmc_descriptive.py")
G = load_script("01_generate_mrmc_simulation.py")


class SimulationTests(unittest.TestCase):
    def test_fixed_seeds_reproduce_and_sizes_are_correct(self):
        tables = []
        for _ in range(2):
            rng = np.random.default_rng(G.SEED)
            readers = G.build_readers()
            cases = G.build_cases(rng)
            results = G.simulate_reader_case_results(readers, cases, rng)
            tables.append((readers, cases, results))
        for left, right in zip(*tables):
            pd.testing.assert_frame_equal(left, right)
        readers, cases, results = tables[0]
        self.assertEqual((len(readers), len(cases), len(results)), (8, 80, 1280))
        self.assertEqual(readers.random_sequence.value_counts().to_dict(), {"BA": 4, "AB": 4})
        self.assertEqual(int(cases.reference_lesion_count.sum()), 66)
        self.assertEqual((int(cases.case_label.sum()), int((cases.case_label == 0).sum())), (48, 32))
        self.assertTrue((results.tp_count + results.fn_count == results.reference_lesion_count).all())
        pd.testing.assert_series_equal(results.case_prediction, ((results.tp_count+results.fp_count) > 0).astype(int), check_names=False)


class DescriptiveTests(unittest.TestCase):
    def setUp(self):
        self.readers, self.cases, self.results = study_tables()

    def test_valid_data_passes_existing_23_checks(self):
        checks = D.run_quality_checks(self.readers, self.cases, self.results)
        self.assertEqual(len(checks), 23)
        self.assertTrue(checks.status.eq("PASS").all())

    def test_hand_calculated_reader_metrics(self):
        summary = D.build_reader_condition_summary(self.readers, self.cases, self.results).set_index(["reader_id", "condition"])
        expected = {("R1", "without_ai"): (1/3, 1/3, 1, 1, 4),
                    ("R1", "with_ai"): (1, 2/3, 1, 0, 3),
                    ("R2", "without_ai"): (1, 2/3, 1, 0, 6),
                    ("R2", "with_ai"): (2/3, 2/3, 1, 1, 4)}
        for key, values in expected.items():
            with self.subTest(key=key):
                row = summary.loc[key]
                self.assertEqual((row.tp_count + row.fn_count, row.scan_count), (3, 3))
                for column, value in zip(["lesion_sensitivity", "fp_per_scan", "case_sensitivity", "case_specificity", "mean_reading_time_min"], values):
                    self.assertAlmostEqual(row[column], value)

    def test_pair_direction_and_reader_average(self):
        summary = D.build_reader_condition_summary(self.readers, self.cases, self.results)
        paired = D.build_paired_differences(summary).set_index("reader_id")
        self.assertAlmostEqual(paired.loc["R1", "lesion_sensitivity_difference"], 2/3)
        self.assertAlmostEqual(paired.loc["R2", "lesion_sensitivity_difference"], -1/3)
        self.assertEqual(paired.mean_reading_time_min_difference.tolist(), [-1, -2])
        overall = D.build_overall_summary(summary, paired.reset_index()).set_index("metric")
        self.assertAlmostEqual(overall.loc["lesion_sensitivity", "without_ai_reader_mean"], 2/3)
        self.assertAlmostEqual(overall.loc["lesion_sensitivity", "with_ai_reader_mean"], 5/6)
        self.assertAlmostEqual(overall.loc["lesion_sensitivity", "mean_paired_difference"], 1/6)
        self.assertAlmostEqual(overall.loc["fp_per_scan", "mean_paired_difference"], 1/6)
        self.assertAlmostEqual(overall.loc["mean_reading_time_min", "mean_paired_difference"], -1.5)

    def test_missing_column_is_readable_value_error(self):
        with self.assertRaisesRegex(ValueError, "case_prediction"):
            D.run_quality_checks(self.readers, self.cases, self.results.drop(columns="case_prediction"))

    def test_duplicate_and_missing_pair_are_rejected(self):
        for table in (pd.concat([self.results, self.results.iloc[[0]]], ignore_index=True), self.results.iloc[1:]):
            with self.subTest(rows=len(table)), self.assertRaisesRegex(ValueError, "Quality checks failed"):
                D.run_quality_checks(self.readers, self.cases, table)

    def test_invalid_clinical_record_fields_are_rejected(self):
        for column, value in [("tp_count", -1), ("reading_time_min", 0), ("confidence_score", 6), ("period", 3), ("case_prediction", 2)]:
            table = self.results.copy()
            table.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                D.run_quality_checks(self.readers, self.cases, table)

    def assert_report_paths(self, isolated):
        checks = D.run_quality_checks(self.readers, self.cases, self.results)
        summary = D.build_reader_condition_summary(self.readers, self.cases, self.results)
        paired = D.build_paired_differences(summary)
        overall = D.build_overall_summary(summary, paired)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            if isolated:
                paths = RunPaths.isolated(root)
            else:
                paths = RunPaths(root, root/'03_模拟数据', root/'04_分析结果',
                                 root/'05_综合结果'/'MRMC教学数据与结果.xlsx')
            expected = [paths.data/'reader_case_results.csv']
            expected.extend(paths.results/name for name in (
                'quality_checks.csv', 'reader_condition_summary.csv', 'paired_differences.csv',
                'overall_descriptive_summary.csv', 'sensitivity_change.png', 'fp_per_scan_change.png',
                'reading_time_change.png', 'MRMC教学性推断结果说明.md'))
            expected.append(paths.workbook)
            for file in expected:
                file.parent.mkdir(parents=True, exist_ok=True)
                file.touch()
            report = paths.results/'模拟结果说明.md'
            with patch.object(D, 'PATHS', paths), patch.object(D, 'ROOT', paths.root), \
                 patch.object(D, 'DATA_DIR', paths.data), patch.object(D, 'OUTPUT_DIR', paths.results):
                D.write_descriptive_summary(self.readers, self.cases, self.results, checks, overall, report)
            text = report.read_text(encoding='utf-8')
            # Ignore the FP/scan term and a sibling report name in explanatory text.
            references = [target for target in re.findall(r'`([^`]+\.(?:csv|png|md|xlsx))`', text) if '/' in target]
            self.assertEqual(references, [file.relative_to(root).as_posix() for file in expected])
            self.assertTrue(all((root/target).is_file() for target in references))
            self.assertNotIn(str(root), text)

    def test_report_paths_follow_isolated_run(self):
        self.assert_report_paths(isolated=True)

    def test_report_paths_keep_legacy_layout(self):
        self.assert_report_paths(isolated=False)


if __name__ == "__main__":
    unittest.main()
