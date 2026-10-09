"""Shared paths/parameters. The teaching scripts keep their legacy defaults."""
from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class StudyParameters:
    simulation_seed: int = 20260920
    sequence_seed: int = 20260921
    bootstrap_seed: int = 20260923
    bootstrap_iterations: int = 5000

    def __post_init__(self):
        for name in ("simulation_seed", "sequence_seed", "bootstrap_seed"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name}必须为非负整数。")
        if not isinstance(self.bootstrap_iterations, int) or self.bootstrap_iterations < 2:
            raise ValueError("bootstrap_iterations必须为至少2的整数（用于计算样本标准误）。")


@dataclass(frozen=True)
class RunPaths:
    root: Path
    data: Path
    results: Path
    workbook: Path

    @classmethod
    def isolated(cls, root):
        root = Path(root).resolve()
        return cls(root, root / "data", root / "results", root / "workbooks" / "MRMC教学数据与结果.xlsx")


def load_config(script_file):
    """Runner overrides only its children's environment, never the user's shell."""
    defaults = StudyParameters()
    values = {}
    for name in ("simulation_seed", "sequence_seed", "bootstrap_seed", "bootstrap_iterations"):
        variable = "READER_STUDY_" + name.upper()
        try:
            values[name] = int(os.environ.get(variable, getattr(defaults, name)))
        except ValueError as error:
            raise ValueError(f"{variable}必须为整数。") from error
    parameters = StudyParameters(**values)
    run_root = os.environ.get("READER_STUDY_RUN_ROOT")
    if run_root:
        paths = RunPaths.isolated(run_root)
    else:
        script_dir = Path(script_file).resolve().parent
        if script_dir.name == "02_统计分析代码":
            root = script_dir.parent
            paths = RunPaths(root, root / "03_模拟数据", root / "04_分析结果", root / "05_综合结果" / "MRMC教学数据与结果.xlsx")
        else:
            paths = RunPaths(script_dir, script_dir / "data", script_dir / "output", script_dir / "MRMC教学数据与结果.xlsx")
    return paths, parameters
