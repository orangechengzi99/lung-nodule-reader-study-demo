import unittest
import numpy as np
import pandas as pd
from tests.helpers import froc_tables, load_script

F = load_script("03_lesion_level_froc.py")


class FrocTests(unittest.TestCase):
    def setUp(self):
        self.scans, self.references, self.candidates = froc_tables()

    def calculate(self):
        return F.calculate_froc(self.scans, self.references, self.candidates)

    def test_hand_calculated_thresholds_and_duplicate_match(self):
        result = self.calculate().set_index("threshold")
        expected = {np.inf: (0, 2, 0, 0, 0), .9: (1, 1, 0, .5, 0),
                    .8: (1, 1, 1, .5, .5), .7: (1, 1, 2, .5, 1), .5: (2, 0, 2, 1, 1)}
        self.assertEqual(result.index.tolist(), list(expected))
        for threshold, values in expected.items():
            np.testing.assert_allclose(result.loc[threshold].to_numpy(), values)

    def test_zero_one_and_tied_scores_are_inclusive(self):
        self.candidates["score"] = [1, 1, 0, 0]
        result = self.calculate().set_index("threshold")
        self.assertEqual(result.index.tolist(), [np.inf, 1, 0])
        self.assertEqual(result.loc[1, "TP"], 1)
        self.assertEqual(result.loc[1, "FP"], 1)
        self.assertEqual(result.loc[0, "TP"], 2)
        self.assertEqual(result.loc[0, "FP"], 2)

    def test_empty_candidates_are_valid_zero_detection(self):
        self.candidates = self.candidates.iloc[:0]
        result = self.calculate()
        self.assertEqual(len(result), 1)
        self.assertTrue(np.isposinf(result.iloc[0].threshold))
        self.assertEqual(result.iloc[0].FN, 2)
        self.assertEqual(result.iloc[0].lesion_sensitivity, 0)

    def test_empty_scans_are_readable_error(self):
        self.scans = self.scans.iloc[:0]
        self.references = self.references.iloc[:0]
        self.candidates = self.candidates.iloc[:0]
        with self.assertRaisesRegex(ValueError, "扫描|scan"):
            self.calculate()

    def test_empty_references_are_readable_error(self):
        self.references = self.references.iloc[:0]
        self.candidates["matched_reference_id"] = None
        with self.assertRaisesRegex(ValueError, "参考|reference"):
            self.calculate()

    def test_missing_column_is_rejected(self):
        self.candidates = self.candidates.drop(columns="score")
        with self.assertRaisesRegex(ValueError, "score"):
            self.calculate()

    def test_duplicate_ids_are_rejected(self):
        for table_name, column in [("scans", "scan_id"), ("references", "reference_id"), ("candidates", "candidate_id")]:
            scans, refs, candidates = froc_tables()
            tables = {"scans": scans, "references": refs, "candidates": candidates}
            tables[table_name].loc[1, column] = tables[table_name].loc[0, column]
            with self.subTest(table=table_name), self.assertRaises(ValueError):
                F.calculate_froc(scans, refs, candidates)

    def test_missing_keys_and_score_are_rejected(self):
        for column in ("candidate_id", "scan_id", "score"):
            scans, refs, candidates = froc_tables()
            candidates[column] = candidates[column].astype(object)
            candidates.loc[0, column] = None
            with self.subTest(column=column), self.assertRaises(ValueError):
                F.calculate_froc(scans, refs, candidates)

    def test_out_of_range_nonfinite_and_text_scores_are_rejected(self):
        for value in (-.1, 1.1, np.inf, np.nan, "bad"):
            self.candidates = froc_tables()[2]
            self.candidates["score"] = self.candidates["score"].astype(object)
            self.candidates.loc[0, "score"] = value
            with self.subTest(score=value), self.assertRaises(ValueError):
                self.calculate()

    def test_unknown_scan_and_invalid_reference_match_are_rejected(self):
        for column, value in [("scan_id", "unknown"), ("matched_reference_id", "unknown"), ("matched_reference_id", "L2")]:
            self.candidates = froc_tables()[2]
            self.candidates.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                self.calculate()


if __name__ == "__main__":
    unittest.main()
