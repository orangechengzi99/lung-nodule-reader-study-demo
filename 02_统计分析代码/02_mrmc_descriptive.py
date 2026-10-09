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

EXPECTED_CONDITIONS = {"without_ai", "with_ai"}

PLOT_STYLE = configure_plotting()


def require_columns(df: pd.DataFrame, required: set[str], table_name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{table_name} is missing columns: {missing}")


def run_quality_checks(
    readers: pd.DataFrame,
    cases: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(readers, {"reader_id", "random_sequence"}, "readers.csv")
    require_columns(cases, {"case_id", "case_label", "reference_lesion_count"}, "cases.csv")
    require_columns(
        results,
        {"reader_id", "case_id", "condition", "period", "reference_lesion_count",
         "tp_count", "fn_count", "fp_count", "case_prediction", "case_correct",
         "reading_time_min", "confidence_score"},
        "reader_case_results.csv",
    )
    if readers.empty or cases.empty or results.empty:
        raise ValueError("读者、病例和判读结果表均不能为空。")
    # Preflight guards are separate from the original 23 named data checks.
    for table, columns in (
        (cases, ["case_label", "reference_lesion_count"]),
        (results, ["period", "reference_lesion_count", "tp_count", "fn_count",
                   "fp_count", "case_prediction", "case_correct", "reading_time_min", "confidence_score"]),
    ):
        for column in columns:
            if not table[column].map(pd.api.types.is_number).all():
                raise ValueError(f"{column}必须为数值。")
            values = table[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{column}必须为有限数值且不能缺失。")
            if column in {"reference_lesion_count", "tp_count", "fn_count", "fp_count"}:
                if not ((values >= 0) & (values == np.floor(values))).all():
                    raise ValueError(f"{column}必须为非负整数。")
    checks: list[dict[str, object]] = []

    def add_check(check: str, passed: bool, detail: str) -> None:
        checks.append(
            {
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    add_check(
        "reader_id is unique",
        readers["reader_id"].is_unique,
        f"unique={readers['reader_id'].nunique()}, rows={len(readers)}",
    )
    add_check(
        "case_id is unique",
        cases["case_id"].is_unique,
        f"unique={cases['case_id'].nunique()}, rows={len(cases)}",
    )

    add_check(
        "source tables have no missing values",
        not readers.isna().any().any() and not cases.isna().any().any(),
        (
            f"reader missing cells={int(readers.isna().sum().sum())}, "
            f"case missing cells={int(cases.isna().sum().sum())}"
        ),
    )

    valid_sequences = readers["random_sequence"].isin(["AB", "BA"])
    add_check(
        "random sequence values are valid",
        bool(valid_sequences.all()),
        f"invalid rows={int((~valid_sequences).sum())}",
    )
    sequence_counts = readers["random_sequence"].value_counts()
    sequences_are_balanced = (
        len(sequence_counts) == 2
        and sequence_counts.get("AB", 0) == sequence_counts.get("BA", 0)
    )
    add_check(
        "AB and BA sequence allocation is balanced",
        bool(sequences_are_balanced),
        f"counts={sequence_counts.to_dict()}",
    )

    key_columns = ["reader_id", "case_id", "condition"]
    duplicate_count = int(results.duplicated(key_columns).sum())
    add_check(
        "reader-case-condition key has no duplicates",
        duplicate_count == 0,
        f"duplicate rows={duplicate_count}",
    )

    missing_cell_count = int(results.isna().sum().sum())
    add_check(
        "reader-case results have no missing values",
        missing_cell_count == 0,
        f"missing cells={missing_cell_count}",
    )

    observed_conditions = set(results["condition"].unique())
    add_check(
        "condition values are valid",
        observed_conditions == EXPECTED_CONDITIONS,
        f"observed={sorted(observed_conditions)}",
    )

    expected_rows = len(readers) * len(cases) * len(EXPECTED_CONDITIONS)
    add_check(
        "full-crossed row count is complete",
        len(results) == expected_rows,
        f"observed={len(results)}, expected={expected_rows}",
    )

    expected_pairs = len(EXPECTED_CONDITIONS)
    pair_counts = results.groupby(["reader_id", "case_id"])["condition"].nunique()
    incomplete_pairs = int((pair_counts != expected_pairs).sum())
    add_check(
        "each reader-case pair has both conditions",
        incomplete_pairs == 0,
        f"incomplete pairs={incomplete_pairs}",
    )

    period_counts = results.groupby(["reader_id", "case_id"])["period"].nunique()
    incomplete_periods = int((period_counts != 2).sum())
    add_check(
        "each reader-case pair has both periods",
        incomplete_periods == 0,
        f"incomplete period pairs={incomplete_periods}",
    )

    known_reader_ids = set(readers["reader_id"])
    unknown_readers = sorted(set(results["reader_id"]) - known_reader_ids)
    add_check(
        "all reader IDs exist in readers.csv",
        not unknown_readers,
        f"unknown={unknown_readers}",
    )

    known_case_ids = set(cases["case_id"])
    unknown_cases = sorted(set(results["case_id"]) - known_case_ids)
    add_check(
        "all case IDs exist in cases.csv",
        not unknown_cases,
        f"unknown={unknown_cases}",
    )

    case_reference_map = cases.set_index("case_id")["reference_lesion_count"]
    mapped_reference_counts = results["case_id"].map(case_reference_map)
    reference_mismatches = int(
        (results["reference_lesion_count"] != mapped_reference_counts).sum()
    )
    add_check(
        "reference lesion counts match cases.csv",
        reference_mismatches == 0,
        f"mismatched rows={reference_mismatches}",
    )

    binary_columns = ["case_prediction", "case_correct"]
    invalid_binary = int((~results[binary_columns].isin([0, 1])).sum().sum())
    add_check(
        "binary result fields contain only 0 or 1",
        invalid_binary == 0,
        f"invalid cells={invalid_binary}",
    )

    valid_periods = results["period"].isin([1, 2])
    add_check(
        "period values are valid",
        bool(valid_periods.all()),
        f"invalid rows={int((~valid_periods).sum())}",
    )

    sequence_map = readers.set_index("reader_id")["random_sequence"]
    mapped_sequences = results["reader_id"].map(sequence_map)
    expected_period = pd.Series(
        [
            1 if (condition == "without_ai") == (sequence == "AB") else 2
            for condition, sequence in zip(results["condition"], mapped_sequences)
        ],
        index=results.index,
    )
    sequence_period_mismatches = int((results["period"] != expected_period).sum())
    add_check(
        "period agrees with AB or BA sequence",
        sequence_period_mismatches == 0,
        f"mismatched rows={sequence_period_mismatches}",
    )

    expected_case_prediction = (
        (results["tp_count"] + results["fp_count"]) > 0
    ).astype(int)
    prediction_rule_mismatches = int(
        (results["case_prediction"] != expected_case_prediction).sum()
    )
    add_check(
        "case_prediction agrees with the prespecified marker-based rule",
        prediction_rule_mismatches == 0,
        f"mismatched rows={prediction_rule_mismatches}",
    )

    labels = results["case_id"].map(cases.set_index("case_id")["case_label"])
    expected_correct = (results["case_prediction"] == labels).astype(int)
    correctness_mismatches = int((results["case_correct"] != expected_correct).sum())
    add_check(
        "case_correct agrees with prediction and reference label",
        correctness_mismatches == 0,
        f"mismatched rows={correctness_mismatches}",
    )

    valid_confidence = results["confidence_score"].between(1, 5, inclusive="both")
    add_check(
        "confidence scores are between 1 and 5",
        bool(valid_confidence.all()),
        f"invalid rows={int((~valid_confidence).sum())}",
    )

    positive_reading_time = results["reading_time_min"] > 0
    add_check(
        "reading times are positive",
        bool(positive_reading_time.all()),
        f"invalid rows={int((~positive_reading_time).sum())}",
    )

    counts_are_nonnegative = bool(
        (results[["reference_lesion_count", "tp_count", "fn_count", "fp_count"]]
         >= 0).all().all()
    )
    add_check(
        "lesion counts are nonnegative",
        counts_are_nonnegative,
        "reference, TP, FN and FP counts must be >= 0",
    )

    lesion_balance_ok = bool(
        (results["tp_count"] + results["fn_count"]
         == results["reference_lesion_count"]).all()
    )
    add_check(
        "TP plus FN equals reference lesion count",
        lesion_balance_ok,
        "checked for every reader-case-condition row",
    )

    quality_checks = pd.DataFrame(checks)
    if (quality_checks["status"] == "FAIL").any():
        failed = quality_checks.loc[quality_checks["status"] == "FAIL", "check"].tolist()
        raise ValueError(f"Quality checks failed: {failed}")
    return quality_checks


def build_reader_condition_summary(
    readers: pd.DataFrame,
    cases: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    data = results.merge(
        cases[["case_id", "case_label"]],
        on="case_id",
        how="left",
        validate="many_to_one",
    )

    summary = (
        data.groupby(["reader_id", "condition"], as_index=False)
        .agg(
            scan_count=("case_id", "nunique"),
            reference_lesion_count=("reference_lesion_count", "sum"),
            tp_count=("tp_count", "sum"),
            fn_count=("fn_count", "sum"),
            fp_count=("fp_count", "sum"),
            mean_reading_time_min=("reading_time_min", "mean"),
            mean_confidence_score=("confidence_score", "mean"),
        )
    )
    summary["lesion_sensitivity"] = (
        summary["tp_count"] / summary["reference_lesion_count"]
    )
    summary["fp_per_scan"] = summary["fp_count"] / summary["scan_count"]

    case_rates = (
        data.groupby(["reader_id", "condition", "case_label"])["case_correct"]
        .mean()
        .unstack("case_label")
        .rename(columns={0: "case_specificity", 1: "case_sensitivity"})
        .reset_index()
    )
    summary = summary.merge(
        case_rates,
        on=["reader_id", "condition"],
        how="left",
        validate="one_to_one",
    )
    summary = summary.merge(
        readers[["reader_id", "experience_years", "experience_group", "random_sequence"]],
        on="reader_id",
        how="left",
        validate="many_to_one",
    )

    ordered_columns = [
        "reader_id",
        "experience_years",
        "experience_group",
        "random_sequence",
        "condition",
        "scan_count",
        "reference_lesion_count",
        "tp_count",
        "fn_count",
        "fp_count",
        "lesion_sensitivity",
        "fp_per_scan",
        "case_sensitivity",
        "case_specificity",
        "mean_reading_time_min",
        "mean_confidence_score",
    ]
    return summary[ordered_columns].sort_values(["reader_id", "condition"])


def build_paired_differences(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "lesion_sensitivity",
        "fp_per_scan",
        "case_sensitivity",
        "case_specificity",
        "mean_reading_time_min",
        "mean_confidence_score",
    ]
    paired = summary.pivot(index="reader_id", columns="condition", values=metrics)
    paired.columns = [f"{metric}_{condition}" for metric, condition in paired.columns]
    paired = paired.reset_index()

    reader_info = summary[
        ["reader_id", "experience_years", "experience_group", "random_sequence"]
    ].drop_duplicates()
    paired = reader_info.merge(paired, on="reader_id", how="left", validate="one_to_one")

    for metric in metrics:
        paired[f"{metric}_difference"] = (
            paired[f"{metric}_with_ai"] - paired[f"{metric}_without_ai"]
        )

    ordered_columns = [
        "reader_id",
        "experience_years",
        "experience_group",
        "random_sequence",
    ]
    for metric in metrics:
        ordered_columns.extend(
            [
                f"{metric}_without_ai",
                f"{metric}_with_ai",
                f"{metric}_difference",
            ]
        )
    return paired[ordered_columns].sort_values("reader_id")


def build_overall_summary(
    reader_summary: pd.DataFrame,
    paired: pd.DataFrame,
) -> pd.DataFrame:
    metric_labels = {
        "lesion_sensitivity": "Reader-average lesion sensitivity",
        "fp_per_scan": "Reader-average false positives per scan",
        "case_sensitivity": "Reader-average case sensitivity",
        "case_specificity": "Reader-average case specificity",
        "mean_reading_time_min": "Reader-average reading time (min)",
        "mean_confidence_score": "Reader-average confidence score",
    }
    rows: list[dict[str, object]] = []
    for metric, label in metric_labels.items():
        condition_means = reader_summary.groupby("condition")[metric].mean()
        differences = paired[f"{metric}_difference"]
        rows.append(
            {
                "metric": metric,
                "metric_label": label,
                "without_ai_reader_mean": condition_means["without_ai"],
                "with_ai_reader_mean": condition_means["with_ai"],
                "mean_paired_difference": differences.mean(),
                "minimum_reader_difference": differences.min(),
                "maximum_reader_difference": differences.max(),
                "interpretation": "descriptive teaching result only",
            }
        )
    return pd.DataFrame(rows)


def write_descriptive_summary(
    readers: pd.DataFrame,
    cases: pd.DataFrame,
    results: pd.DataFrame,
    quality_checks: pd.DataFrame,
    overall: pd.DataFrame,
    path: Path,
) -> None:
    metrics = overall.set_index("metric")

    def values(metric: str) -> tuple[float, float, float]:
        row = metrics.loc[metric]
        return (
            float(row["without_ai_reader_mean"]),
            float(row["with_ai_reader_mean"]),
            float(row["mean_paired_difference"]),
        )

    def percentage_points(value: float) -> str:
        return f"{value * 100 + 1e-12:+.1f}个百分点"

    lesion = values("lesion_sensitivity")
    fp_scan = values("fp_per_scan")
    case_sensitivity = values("case_sensitivity")
    case_specificity = values("case_specificity")
    reading_time = values("mean_reading_time_min")
    confidence = values("mean_confidence_score")
    positive_cases = int((cases["case_label"] == 1).sum())
    negative_cases = int((cases["case_label"] == 0).sum())
    reference_lesions = int(cases["reference_lesion_count"].sum())

    # Paths are relative to this run's root, not hard-coded legacy folders.
    report_files = [
        (DATA_DIR / 'reader_case_results.csv', '模拟阅片长表。'),
        (OUTPUT_DIR / 'quality_checks.csv', '质控结果。'),
        (OUTPUT_DIR / 'reader_condition_summary.csv', '每名阅片者在两种条件下的指标。'),
        (OUTPUT_DIR / 'paired_differences.csv', '逐阅片者配对差值。'),
        (OUTPUT_DIR / 'overall_descriptive_summary.csv', '阅片者平均描述性结果。'),
        (OUTPUT_DIR / 'sensitivity_change.png', '结节级敏感度配对变化图。'),
        (OUTPUT_DIR / 'fp_per_scan_change.png', '`FP/scan`配对变化图。'),
        (OUTPUT_DIR / 'reading_time_change.png', '阅片时间配对变化图。'),
        (OUTPUT_DIR / 'MRMC教学性推断结果说明.md', '教学性推断方法、结果与边界。'),
        (PATHS.workbook, '可筛选、复核的综合工作簿。'),
    ]

    lines = [
        "# MRMC辅助阅片描述性结果说明",
        "",
        "## 数据与设计",
        "",
        (
            f"本教学练习包含{len(readers)}名虚构阅片者、{len(cases)}例虚构胸部CT，"
            f"以及无AI和AI辅助两种阅片条件，共{len(results)}条阅片者-病例-条件记录。"
            f"{len(cases)}例CT中包含{positive_cases}例结节阳性病例、"
            f"{negative_cases}例结节阴性病例和{reference_lesions}个参考结节。"
        ),
        "",
        "所有数据均为固定随机种子生成的虚构数据，不来源于真实患者、医院或AI产品。",
        "",
        "## 数据质控",
        "",
        f"共{len(quality_checks)}项预设数据质量检查，全部通过。检查范围包括：",
        "",
        "1. 阅片者编号和病例编号唯一。",
        "2. 阅片者-病例-条件记录无重复、无缺失。",
        "3. 每个阅片者对每个病例均具有无AI和AI辅助两种结果。",
        "4. AB/BA序列分配平衡，条件与周期对应关系正确。",
        "5. 病例级判断符合统一的预设标记规则。",
        "6. 每条记录均满足`TP + FN = 参考结节数`，且计数非负。",
        "",
        "## 主要描述性结果",
        "",
        "| 指标 | 无AI | AI辅助 | AI辅助减无AI |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| 阅片者平均结节级敏感度 | {lesion[0]:.1%} | {lesion[1]:.1%} | "
            f"{percentage_points(lesion[2])} |"
        ),
        (
            f"| 阅片者平均`FP/scan` | {fp_scan[0]:.2f} | {fp_scan[1]:.2f} | "
            f"{fp_scan[2]:+.2f} |"
        ),
        (
            f"| 阅片者平均病例级敏感度 | {case_sensitivity[0]:.1%} | "
            f"{case_sensitivity[1]:.1%} | {percentage_points(case_sensitivity[2])} |"
        ),
        (
            f"| 阅片者平均病例级特异度 | {case_specificity[0]:.1%} | "
            f"{case_specificity[1]:.1%} | {percentage_points(case_specificity[2])} |"
        ),
        (
            f"| 阅片者平均阅片时间 | {reading_time[0]:.2f}分钟 | "
            f"{reading_time[1]:.2f}分钟 | {reading_time[2]:+.2f}分钟 |"
        ),
        (
            f"| 阅片者平均阅片信心评分 | {confidence[0]:.2f} | "
            f"{confidence[1]:.2f} | {confidence[2]:+.2f} |"
        ),
        "",
        "## 结果解释",
        "",
        (
            "在本组虚构数据中，AI辅助条件下阅片者平均结节级敏感度较高，"
            "平均阅片时间较短，但`FP/scan`较高，病例级特异度较低。"
            "这说明评价辅助阅片产品时，不能只看检出能力，还需要同时考虑误报负担和工作效率。"
        ),
        "",
        (
            "以上差值均为描述性结果，本节不据此判断统计学意义。"
            "项目另使用阅片者－病例双重Bootstrap演示标准误、95%置信区间和探索性P值，"
            "详见`MRMC教学性推断结果说明.md`；该教学性推断仍不能替代正式MRMC主分析，"
            "也不能证明真实AI产品具有临床有效性或临床获益。"
        ),
        "",
        "## 文件对应关系",
        "",
        *[
            f"- `{file.relative_to(ROOT).as_posix()}`：{description}"
            for file, description in report_files
        ],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_paired_plot(
    paired: pd.DataFrame,
    metric: str,
    y_label: str,
    title: str,
    output_name: str,
    percent_axis: bool = False,
) -> None:
    baseline = paired[f"{metric}_without_ai"]
    aided = paired[f"{metric}_with_ai"]

    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    for index, _reader_id in enumerate(paired["reader_id"]):
        ax.plot(
            [0, 1],
            [baseline.iloc[index], aided.iloc[index]],
            color="#8A94A3",
            linewidth=1.3,
            alpha=0.85,
            zorder=1,
        )

    ax.scatter([0] * len(paired), baseline, color="#3B82F6", s=42, label=PLOT_STYLE.text("无AI"), zorder=2)
    ax.scatter([1] * len(paired), aided, color="#15803D", s=42, label=PLOT_STYLE.text("AI辅助"), zorder=2)
    ax.set_xticks([0, 1], [PLOT_STYLE.text("无AI"), PLOT_STYLE.text("AI辅助")])
    ax.set_ylabel(PLOT_STYLE.text(y_label))
    ax.set_title(PLOT_STYLE.text(title))
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best")
    ax.set_xlim(-0.25, 1.12)
    if percent_axis:
        ax.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
        lower = max(0.0, min(baseline.min(), aided.min()) - 0.06)
        upper = min(1.0, max(baseline.max(), aided.max()) + 0.06)
        ax.set_ylim(lower, upper)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / output_name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print_plot_style(PLOT_STYLE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    readers = pd.read_csv(DATA_DIR / "readers.csv")
    cases = pd.read_csv(DATA_DIR / "cases.csv")
    results = pd.read_csv(DATA_DIR / "reader_case_results.csv")

    require_columns(
        readers,
        {"reader_id", "experience_years", "experience_group", "random_sequence"},
        "readers.csv",
    )
    require_columns(
        cases,
        {"case_id", "case_label", "reference_lesion_count"},
        "cases.csv",
    )
    require_columns(
        results,
        {
            "reader_id",
            "case_id",
            "condition",
            "period",
            "reference_lesion_count",
            "tp_count",
            "fn_count",
            "fp_count",
            "case_prediction",
            "case_correct",
            "reading_time_min",
            "confidence_score",
        },
        "reader_case_results.csv",
    )

    quality_checks = run_quality_checks(readers, cases, results)
    reader_summary = build_reader_condition_summary(readers, cases, results)
    paired = build_paired_differences(reader_summary)
    overall = build_overall_summary(reader_summary, paired)

    quality_checks.to_csv(OUTPUT_DIR / "quality_checks.csv", index=False, encoding="utf-8-sig")
    reader_summary.to_csv(
        OUTPUT_DIR / "reader_condition_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )
    paired.to_csv(
        OUTPUT_DIR / "paired_differences.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )
    overall.to_csv(
        OUTPUT_DIR / "overall_descriptive_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )
    write_descriptive_summary(
        readers,
        cases,
        results,
        quality_checks,
        overall,
        OUTPUT_DIR / "模拟结果说明.md",
    )

    save_paired_plot(
        paired,
        metric="lesion_sensitivity",
        y_label="结节级敏感度",
        title="阅片者结节级敏感度：无AI与AI辅助",
        output_name="sensitivity_change.png",
        percent_axis=True,
    )
    save_paired_plot(
        paired,
        metric="fp_per_scan",
        y_label="平均每例假阳性数（FP/scan）",
        title="阅片者平均每例假阳性数：无AI与AI辅助",
        output_name="fp_per_scan_change.png",
    )
    save_paired_plot(
        paired,
        metric="mean_reading_time_min",
        y_label="平均阅片时间（分钟）",
        title="阅片者平均阅片时间：无AI与AI辅助",
        output_name="reading_time_change.png",
    )

    display_columns = [
        "metric_label",
        "without_ai_reader_mean",
        "with_ai_reader_mean",
        "mean_paired_difference",
    ]
    print("All quality checks passed.")
    print("\nReader-average descriptive results:")
    print(overall[display_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nImportant: these are simulated descriptive results, not formal MRMC inference.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
