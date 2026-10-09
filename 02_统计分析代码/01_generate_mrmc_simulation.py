from pathlib import Path

import numpy as np
import pandas as pd
from study_config import load_config


PATHS, PARAMETERS = load_config(__file__)
SEED = PARAMETERS.simulation_seed
SEQUENCE_SEED = PARAMETERS.sequence_seed
ROOT, DATA_DIR, OUTPUT_DIR = PATHS.root, PATHS.data, PATHS.results


def build_readers() -> pd.DataFrame:
    sequence_rng = np.random.default_rng(SEQUENCE_SEED)
    random_sequences = np.array(["AB"] * 4 + ["BA"] * 4)
    sequence_rng.shuffle(random_sequences)

    return pd.DataFrame(
        {
            "reader_id": [f"R{i:02d}" for i in range(1, 9)],
            "experience_years": [2, 3, 5, 7, 9, 12, 15, 18],
            "experience_group": [
                "junior",
                "junior",
                "intermediate",
                "intermediate",
                "intermediate",
                "senior",
                "senior",
                "senior",
            ],
            "random_sequence": random_sequences,
            "baseline_detection_probability": [0.70, 0.72, 0.75, 0.77, 0.79, 0.82, 0.84, 0.86],
            "baseline_fp_rate": [0.62, 0.58, 0.54, 0.50, 0.46, 0.40, 0.36, 0.32],
            "baseline_reading_time_min": [5.4, 5.2, 5.0, 4.8, 4.6, 4.3, 4.1, 3.9],
        }
    )


def build_cases(rng: np.random.Generator) -> pd.DataFrame:
    case_ids = [f"C{i:03d}" for i in range(1, 81)]
    positive_flags = np.array([1] * 48 + [0] * 32)
    rng.shuffle(positive_flags)

    positive_lesion_counts = np.array([1] * 34 + [2] * 10 + [3] * 4)
    rng.shuffle(positive_lesion_counts)
    lesion_counts = np.zeros(80, dtype=int)
    lesion_counts[positive_flags == 1] = positive_lesion_counts

    centers = np.resize(np.array(["Center_A", "Center_B", "Center_C"]), 80)
    rng.shuffle(centers)
    difficulties = rng.choice(
        ["easy", "moderate", "hard"],
        size=80,
        p=[0.30, 0.45, 0.25],
    )
    densities = np.full(80, "none", dtype=object)
    densities[positive_flags == 1] = rng.choice(
        ["solid", "part_solid", "ground_glass"],
        size=int(positive_flags.sum()),
        p=[0.50, 0.25, 0.25],
    )
    vendors = rng.choice(["Vendor_A", "Vendor_B", "Vendor_C"], size=80)

    return pd.DataFrame(
        {
            "case_id": case_ids,
            "center_id": centers,
            "case_label": positive_flags,
            "reference_lesion_count": lesion_counts,
            "difficulty": difficulties,
            "dominant_density": densities,
            "scanner_vendor": vendors,
        }
    )


def simulate_reader_case_results(
    readers: pd.DataFrame,
    cases: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    difficulty_detection_adjustment = {"easy": 0.07, "moderate": 0.00, "hard": -0.11}
    density_adjustment = {
        "none": 0.00,
        "solid": 0.03,
        "part_solid": -0.02,
        "ground_glass": -0.07,
    }
    difficulty_fp_adjustment = {"easy": -0.08, "moderate": 0.00, "hard": 0.16}
    difficulty_time_adjustment = {"easy": -0.35, "moderate": 0.15, "hard": 0.75}

    rows: list[dict[str, object]] = []
    for reader in readers.itertuples(index=False):
        period_by_condition = (
            {"without_ai": 1, "with_ai": 2}
            if reader.random_sequence == "AB"
            else {"with_ai": 1, "without_ai": 2}
        )
        for case in cases.itertuples(index=False):
            for condition in ["without_ai", "with_ai"]:
                ai_gain = 0.09 if condition == "with_ai" else 0.00
                if condition == "with_ai" and case.difficulty == "hard":
                    ai_gain += 0.02

                detection_probability = (
                    reader.baseline_detection_probability
                    + difficulty_detection_adjustment[case.difficulty]
                    + density_adjustment[case.dominant_density]
                    + ai_gain
                )
                detection_probability = float(np.clip(detection_probability, 0.20, 0.97))

                reference_count = int(case.reference_lesion_count)
                tp_count = int(rng.binomial(reference_count, detection_probability))
                fn_count = reference_count - tp_count

                fp_rate = (
                    reader.baseline_fp_rate
                    + difficulty_fp_adjustment[case.difficulty]
                    + (0.20 if condition == "with_ai" else 0.00)
                )
                fp_rate = max(fp_rate, 0.05)
                fp_count = int(rng.poisson(fp_rate))

                reading_time = (
                    reader.baseline_reading_time_min
                    + difficulty_time_adjustment[case.difficulty]
                    - (0.70 if condition == "with_ai" else 0.00)
                    + rng.normal(0.0, 0.32)
                )
                reading_time = round(max(reading_time, 1.5), 2)

                case_prediction = int((tp_count + fp_count) > 0)
                case_correct = int(case_prediction == case.case_label)

                confidence_mean = 3.0 + 1.3 * (detection_probability - 0.5)
                if condition == "with_ai":
                    confidence_mean += 0.25
                confidence_score = int(np.clip(round(rng.normal(confidence_mean, 0.55)), 1, 5))

                rows.append(
                    {
                        "reader_id": reader.reader_id,
                        "case_id": case.case_id,
                        "condition": condition,
                        "period": period_by_condition[condition],
                        "reference_lesion_count": reference_count,
                        "tp_count": tp_count,
                        "fn_count": fn_count,
                        "fp_count": fp_count,
                        "case_prediction": case_prediction,
                        "case_correct": case_correct,
                        "reading_time_min": reading_time,
                        "confidence_score": confidence_score,
                    }
                )

    return pd.DataFrame(rows)


def build_froc_teaching_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scans = pd.DataFrame({"scan_id": [f"S{i:03d}" for i in range(1, 6)]})
    references = pd.DataFrame(
        {
            "reference_id": ["R01", "R02", "R03", "R04", "R05", "R06"],
            "scan_id": ["S001", "S001", "S002", "S003", "S004", "S005"],
        }
    )
    candidates = pd.DataFrame(
        {
            "candidate_id": [
                "C01", "C02", "C03", "C04", "C05", "C06", "C07",
                "C08", "C09", "C10", "C11", "C12", "C13",
            ],
            "scan_id": [
                "S001", "S001", "S001", "S002", "S002", "S002", "S003",
                "S003", "S004", "S004", "S005", "S005", "S001",
            ],
            "score": [
                0.95, 0.40, 0.90, 0.60, 0.80, 0.70, 0.30,
                0.55, 0.50, 0.85, 0.20, 0.65, 0.75,
            ],
            "matched_reference_id": [
                "R01", None, "R02", "R03", None, None, None,
                "R04", None, "R05", "R06", None, "R01",
            ],
        }
    )
    return scans, references, candidates


def build_data_dictionary() -> pd.DataFrame:
    definitions = [
        ("readers.csv", "reader_id", "阅片者匿名编号", "text"),
        ("readers.csv", "experience_years", "胸部CT阅片经验年数", "integer"),
        ("readers.csv", "experience_group", "经验分层：junior/intermediate/senior", "category"),
        (
            "readers.csv",
            "random_sequence",
            "按固定随机种子平衡随机分配的阅片序列：AB或BA",
            "category",
        ),
        ("readers.csv", "baseline_detection_probability", "生成教学数据时使用的基础结节检出概率", "number"),
        ("readers.csv", "baseline_fp_rate", "生成教学数据时使用的基础每例假阳性率", "number"),
        ("readers.csv", "baseline_reading_time_min", "生成教学数据时使用的基础阅片时间，单位为分钟", "number"),
        ("cases.csv", "case_id", "CT病例匿名编号", "text"),
        ("cases.csv", "center_id", "教学设定的数据来源中心编号", "category"),
        ("cases.csv", "case_label", "病例级参考标签：1有目标结节，0无目标结节", "binary"),
        ("cases.csv", "reference_lesion_count", "该CT中的参考结节数量", "integer"),
        ("cases.csv", "difficulty", "教学设定的病例难度", "category"),
        ("cases.csv", "dominant_density", "主要结节密度类型；阴性病例为none", "category"),
        ("cases.csv", "scanner_vendor", "教学设定的CT设备厂商编码", "category"),
        ("reader_case_results.csv", "reader_id", "关联readers.csv的阅片者编号", "text"),
        ("reader_case_results.csv", "case_id", "关联cases.csv的病例编号", "text"),
        ("reader_case_results.csv", "condition", "阅片条件：without_ai或with_ai", "category"),
        ("reader_case_results.csv", "period", "交叉阅片周期：1或2，由AB/BA随机序列决定", "integer"),
        ("reader_case_results.csv", "reference_lesion_count", "该病例的参考结节数量", "integer"),
        ("reader_case_results.csv", "tp_count", "该阅片者在该病例中正确检出的参考结节数", "integer"),
        ("reader_case_results.csv", "fn_count", "该阅片者在该病例中漏检的参考结节数", "integer"),
        ("reader_case_results.csv", "fp_count", "该阅片者在该病例中的未匹配标记数", "integer"),
        (
            "reader_case_results.csv",
            "case_prediction",
            "病例级判断：本模拟中，存在至少一个结节标记（TP+FP>0）记为1，否则为0",
            "binary",
        ),
        ("reader_case_results.csv", "case_correct", "病例级判断是否正确", "binary"),
        ("reader_case_results.csv", "reading_time_min", "单例阅片时间，单位为分钟", "number"),
        ("reader_case_results.csv", "confidence_score", "教学设定的阅片信心评分，1至5分", "integer"),
        ("froc_scans.csv", "scan_id", "FROC教学数据中的CT扫描编号", "text"),
        ("froc_references.csv", "reference_id", "FROC教学数据中的参考结节编号", "text"),
        ("froc_references.csv", "scan_id", "参考结节所属的CT扫描编号", "text"),
        ("froc_candidates.csv", "candidate_id", "AI候选编号", "text"),
        ("froc_candidates.csv", "scan_id", "AI候选所属的CT扫描编号", "text"),
        ("froc_candidates.csv", "score", "AI候选检测置信分数", "number"),
        ("froc_candidates.csv", "matched_reference_id", "预设匹配的参考结节编号；空值表示未匹配", "text"),
    ]
    return pd.DataFrame(definitions, columns=["file", "column", "definition", "data_type"])


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    readers = build_readers()
    cases = build_cases(rng)
    results = simulate_reader_case_results(readers, cases, rng)
    froc_scans, froc_references, froc_candidates = build_froc_teaching_data()
    data_dictionary = build_data_dictionary()

    readers.to_csv(DATA_DIR / "readers.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(DATA_DIR / "cases.csv", index=False, encoding="utf-8-sig")
    results.to_csv(DATA_DIR / "reader_case_results.csv", index=False, encoding="utf-8-sig")
    froc_scans.to_csv(DATA_DIR / "froc_scans.csv", index=False, encoding="utf-8-sig")
    froc_references.to_csv(
        DATA_DIR / "froc_references.csv", index=False, encoding="utf-8-sig"
    )
    froc_candidates.to_csv(
        DATA_DIR / "froc_candidates.csv", index=False, encoding="utf-8-sig"
    )
    data_dictionary.to_csv(DATA_DIR / "data_dictionary.csv", index=False, encoding="utf-8-sig")

    print(f"Random seed: {SEED}")
    print(f"Sequence randomization seed: {SEQUENCE_SEED}")
    print(f"Readers: {len(readers)}")
    print(f"Cases: {len(cases)}")
    print(f"Reference lesions: {cases['reference_lesion_count'].sum()}")
    print(f"Reader-case-condition rows: {len(results)}")
    print(f"FROC scans: {len(froc_scans)}")
    print(f"FROC reference lesions: {len(froc_references)}")
    print(f"FROC candidates: {len(froc_candidates)}")
    print(f"Data directory: {DATA_DIR}")


if __name__ == "__main__":
    main()
