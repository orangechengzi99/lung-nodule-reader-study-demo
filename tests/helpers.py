"""Tiny hand-calculable inputs; importing a script never runs its main()."""
import importlib.util
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "02_统计分析代码"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_script(name):
    module_name = "reader_study_test_" + name.removesuffix(".py")
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, ROOT / "02_统计分析代码" / name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  # dataclasses need a registered module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


def study_tables():
    readers = pd.DataFrame({"reader_id": ["R1", "R2"], "experience_years": [2, 10],
                            "experience_group": ["junior", "senior"], "random_sequence": ["AB", "BA"]})
    cases = pd.DataFrame({"case_id": ["C1", "C2", "C3"], "case_label": [1, 1, 0],
                         "reference_lesion_count": [2, 1, 0]})
    # TP, FP and times, in C1/C2/C3 order. R1 time delta=-1, R2=-2.
    examples = {
        ("R1", "without_ai"): ([1, 0, 0], [0, 1, 0], [6, 4, 2]),
        ("R1", "with_ai"): ([2, 1, 0], [1, 0, 1], [5, 3, 1]),
        ("R2", "without_ai"): ([2, 1, 0], [1, 0, 1], [9, 6, 3]),
        ("R2", "with_ai"): ([2, 0, 0], [1, 1, 0], [7, 4, 1]),
    }
    rows = []
    for (reader, condition), (tp, fp, times) in examples.items():
        for i, case in enumerate(cases.itertuples(index=False)):
            prediction = int(tp[i] + fp[i] > 0)
            period = 1 if (reader == "R1") == (condition == "without_ai") else 2
            rows.append({"reader_id": reader, "case_id": case.case_id, "condition": condition,
                         "period": period, "reference_lesion_count": case.reference_lesion_count,
                         "tp_count": tp[i], "fn_count": case.reference_lesion_count-tp[i],
                         "fp_count": fp[i], "case_prediction": prediction,
                         "case_correct": int(prediction == case.case_label), "reading_time_min": times[i],
                         "confidence_score": 3 if condition == "without_ai" else 4})
    return readers, cases, pd.DataFrame(rows)


def froc_tables():
    # At .8: TP=1, FP=1; at .7 the duplicate match adds FP, not TP.
    return (
        pd.DataFrame({"scan_id": ["S1", "S2"]}),
        pd.DataFrame({"reference_id": ["L1", "L2"], "scan_id": ["S1", "S2"]}),
        pd.DataFrame({"candidate_id": ["A", "B", "C", "D"], "scan_id": ["S1", "S1", "S1", "S2"],
                      "score": [.9, .8, .7, .5], "matched_reference_id": ["L1", None, "L1", "L2"]}),
    )
