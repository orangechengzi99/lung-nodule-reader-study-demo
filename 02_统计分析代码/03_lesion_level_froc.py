from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from study_config import load_config
from plot_style import configure_plotting, print_plot_style


PATHS, PARAMETERS = load_config(__file__)
ROOT, DATA_DIR, OUTPUT_DIR = PATHS.root, PATHS.data, PATHS.results

SCAN_PATH = DATA_DIR / "froc_scans.csv"
REFERENCE_PATH = DATA_DIR / "froc_references.csv"
CANDIDATE_PATH = DATA_DIR / "froc_candidates.csv"

PLOT_STYLE = configure_plotting()


def validate_froc_inputs(
    scans: pd.DataFrame,
    references: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    required_columns = {
        "froc_scans.csv": (scans, {"scan_id"}),
        "froc_references.csv": (references, {"reference_id", "scan_id"}),
        "froc_candidates.csv": (
            candidates,
            {"candidate_id", "scan_id", "score", "matched_reference_id"},
        ),
    }
    for file_name, (table, required) in required_columns.items():
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"{file_name} is missing columns: {missing}")

    if scans.empty:
        raise ValueError("FROC扫描表不能为空，无法计算FP/scan。")
    if references.empty:
        raise ValueError("FROC参考结节表不能为空，无法计算结节级敏感度。")
    if not candidates["score"].map(pd.api.types.is_number).all():
        raise ValueError("AI候选评分必须为数值。")
    if not np.isfinite(candidates["score"].to_numpy(dtype=float)).all():
        raise ValueError("AI候选评分必须为有限数值且不能缺失。")

    if scans["scan_id"].isna().any() or not scans["scan_id"].is_unique:
        raise ValueError("froc_scans.csv中的scan_id必须完整且唯一。")
    if references[["reference_id", "scan_id"]].isna().any().any():
        raise ValueError("froc_references.csv存在缺失的参考结节或扫描编号。")
    if not references["reference_id"].is_unique:
        raise ValueError("froc_references.csv中的reference_id必须唯一。")
    if candidates[["candidate_id", "scan_id", "score"]].isna().any().any():
        raise ValueError("froc_candidates.csv的候选编号、扫描编号或评分存在缺失。")
    if not candidates["candidate_id"].is_unique:
        raise ValueError("froc_candidates.csv中的candidate_id必须唯一。")
    if not candidates["score"].between(0, 1, inclusive="both").all():
        raise ValueError("AI候选评分必须位于0至1之间。")

    known_scans = set(scans["scan_id"])
    unknown_reference_scans = sorted(set(references["scan_id"]) - known_scans)
    unknown_candidate_scans = sorted(set(candidates["scan_id"]) - known_scans)
    if unknown_reference_scans or unknown_candidate_scans:
        raise ValueError(
            "FROC表中存在未登记的扫描编号："
            f"references={unknown_reference_scans}, candidates={unknown_candidate_scans}"
        )

    reference_scan_map = references.set_index("reference_id")["scan_id"]
    matched = candidates.dropna(subset=["matched_reference_id"]).copy()
    unknown_references = sorted(
        set(matched["matched_reference_id"]) - set(references["reference_id"])
    )
    if unknown_references:
        raise ValueError(f"候选匹配到不存在的参考结节：{unknown_references}")
    expected_scans = matched["matched_reference_id"].map(reference_scan_map)
    if not (matched["scan_id"] == expected_scans).all():
        raise ValueError("候选与其匹配参考结节不属于同一扫描。")


def calculate_froc(
    scans: pd.DataFrame,
    references: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    validate_froc_inputs(scans, references, candidates)
    reference_lesion_count = len(references)
    scan_count = len(scans)

    thresholds = [np.inf] + sorted(candidates["score"].unique(), reverse=True)
    rows: list[dict[str, float | int]] = []

    for threshold in thresholds:
        active = candidates[candidates["score"] >= threshold]
        matched_references = active["matched_reference_id"].dropna().unique()
        tp = len(matched_references)
        fn = reference_lesion_count - tp
        fp = len(active) - tp
        rows.append(
            {
                "threshold": threshold,
                "TP": tp,
                "FN": fn,
                "FP": fp,
                "lesion_sensitivity": tp / reference_lesion_count,
                "FP_per_scan": fp / scan_count,
            }
        )

    return pd.DataFrame(rows)


def save_froc_plot(froc: pd.DataFrame) -> None:
    plot_data = froc.sort_values(["FP_per_scan", "lesion_sensitivity"])
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    ax.plot(
        plot_data["FP_per_scan"],
        plot_data["lesion_sensitivity"],
        marker="o",
        color="#1F77B4",
    )
    ax.set_xlabel(PLOT_STYLE.text("平均每例假阳性数（FP/scan）"))
    ax.set_ylabel(PLOT_STYLE.text("结节级敏感度"))
    ax.set_title(PLOT_STYLE.text("结节级FROC教学曲线"))
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lesion_level_froc.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print_plot_style(PLOT_STYLE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scans = pd.read_csv(SCAN_PATH)
    references = pd.read_csv(REFERENCE_PATH)
    candidates = pd.read_csv(CANDIDATE_PATH)
    froc = calculate_froc(scans, references, candidates)
    froc.to_csv(
        OUTPUT_DIR / "froc_results.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )
    save_froc_plot(froc)

    print(froc.to_string(index=False))
    print(f"\nScans: {len(scans)}")
    print(f"Reference lesions: {len(references)}")
    print(f"Candidates: {len(candidates)}")
    print("\nImportant: this is a small simulated teaching dataset.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
