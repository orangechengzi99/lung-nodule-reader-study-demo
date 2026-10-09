"""One-command simulated teaching pipeline. Never overwrite an existing run."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = PROJECT_ROOT / "02_统计分析代码"
sys.path.insert(0, str(SCRIPT_DIR))
from study_config import RunPaths, StudyParameters

SCRIPTS = (
    "01_generate_mrmc_simulation.py", "02_mrmc_descriptive.py",
    "03_lesion_level_froc.py", "04_mrmc_inference_teaching.py",
)
DEPENDENCIES = ("numpy", "pandas", "matplotlib", "openpyxl")
DATA_FILES = (
    "readers.csv", "cases.csv", "reader_case_results.csv", "data_dictionary.csv",
    "froc_scans.csv", "froc_references.csv", "froc_candidates.csv",
)
RESULT_FILES = (
    "quality_checks.csv", "reader_condition_summary.csv", "paired_differences.csv",
    "overall_descriptive_summary.csv", "froc_results.csv", "mrmc_inference_summary.csv",
    "mrmc_bootstrap_distributions.csv", "模拟结果说明.md", "MRMC教学性推断结果说明.md",
    "sensitivity_change.png", "fp_per_scan_change.png", "reading_time_change.png",
    "lesion_level_froc.png", "mrmc_inference_confidence_intervals.png", "workbook_summary_preview.png",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_record(path, root):
    record = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            record["columns"] = reader.fieldnames
            record["rows"] = sum(1 for _ in reader)
    return record


def resolve_output(value, project_root=PROJECT_ROOT):
    project_root = Path(project_root).resolve()
    output = Path(value).expanduser()
    if not output.is_absolute():
        output = project_root / output
    output = output.resolve()
    allowed_parent = project_root / "outputs"
    if output == project_root or output in project_root.parents:
        raise ValueError("输出目录不能是项目根目录或其上级目录。")
    if output.is_relative_to(project_root):
        if output == allowed_parent or not output.is_relative_to(allowed_parent):
            raise ValueError("项目内部仅允许 outputs 下的新子目录，不能写入代码或基线目录。")
    if output.exists():
        raise ValueError(f"输出路径已存在，不覆盖：{output}。请换一个新的目录名称。")
    return output


def write_manifest(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def execute_step(script, paths, parameters, log_dir, plot_language="auto"):
    """Use the same interpreter as the runner; arguments never pass through a shell."""
    environment = os.environ.copy()
    environment["READER_STUDY_RUN_ROOT"] = str(paths.root)
    environment["READER_STUDY_PLOT_LANGUAGE"] = plot_language
    for key, value in asdict(parameters).items():
        environment["READER_STUDY_" + key.upper()] = str(value)
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLCONFIGDIR"] = str(paths.root / ".cache" / "matplotlib")
    stdout_path = log_dir / (script.stem + ".stdout.log")
    stderr_path = log_dir / (script.stem + ".stderr.log")
    command = [sys.executable, str(script)]
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, stdout=stdout, stderr=stderr, check=False)
    step = {"script": script.name, "command": command, "returncode": completed.returncode,
            "elapsed_seconds": round(time.perf_counter()-started, 3),
            "stdout_log": stdout_path.relative_to(paths.root).as_posix(),
            "stderr_log": stderr_path.relative_to(paths.root).as_posix()}
    for line in stdout_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("PLOT_STYLE_JSON="):
            step["plot_style"] = json.loads(line.split("=", 1)[1])
            break
    return step


def collect_outputs(paths):
    inputs = [file_record(paths.data / name, paths.root) for name in DATA_FILES if (paths.data / name).is_file()]
    artifacts = [file_record(paths.results / name, paths.root) for name in RESULT_FILES if (paths.results / name).is_file()]
    if paths.workbook.is_file():
        artifacts.append(file_record(paths.workbook, paths.root))
    quality = {"status": "not_available", "checks": 0, "passed": 0, "failed": 0}
    quality_path = paths.results / "quality_checks.csv"
    if quality_path.is_file():
        with quality_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        passed = sum(row["status"] == "PASS" for row in rows)
        failed = len(rows)-passed
        quality = {"status": "passed" if rows and not failed else "failed", "checks": len(rows), "passed": passed, "failed": failed}
    return inputs, artifacts, quality


def run_pipeline(output, parameters, plot_language="auto"):
    output = resolve_output(output)
    # exist_ok=False protects even an empty existing directory and concurrent runs.
    output.mkdir(parents=True, exist_ok=False)
    paths = RunPaths.isolated(output)
    log_dir = output / "logs"
    log_dir.mkdir()
    manifest_path = output / "run_manifest.json"
    manifest = {
        "schema_version": 1, "status": "running", "started_at_utc": utc_now(),
        "output_directory": str(output), "parameters": asdict(parameters),
        "plotting": {"requested_language": plot_language},
        "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "dependencies": {}},
        "analysis_scope": "simulated teaching data; preset candidate matching; exploratory bootstrap, not clinical evidence",
        "steps": [], "source_files": [], "input_files": [], "artifacts": [],
        "quality_checks": {"status": "not_available", "checks": 0, "passed": 0, "failed": 0},
    }
    started = time.perf_counter()
    write_manifest(manifest_path, manifest)
    try:
        for dependency in DEPENDENCIES:
            try:
                manifest["environment"]["dependencies"][dependency] = importlib.metadata.version(dependency)
            except importlib.metadata.PackageNotFoundError as error:
                raise RuntimeError(f"缺少依赖 {dependency}，请使用当前解释器安装 requirements.txt。") from error
        for source in [Path(__file__).resolve(), SCRIPT_DIR / "study_config.py", SCRIPT_DIR / "plot_style.py"] + [SCRIPT_DIR / name for name in SCRIPTS]:
            manifest["source_files"].append(file_record(source, PROJECT_ROOT))
        write_manifest(manifest_path, manifest)
        for index, name in enumerate(SCRIPTS, start=1):
            print(f"[{index}/4] 开始 {name}", flush=True)
            step = execute_step(SCRIPT_DIR / name, paths, parameters, log_dir, plot_language)
            manifest["steps"].append(step)
            write_manifest(manifest_path, manifest)
            if step["returncode"] != 0:
                raise RuntimeError(f"{name} 失败（退出码 {step['returncode']}）；查看 {step['stderr_log']}。后续步骤未运行。")
            print(f"[{index}/4] 完成（{step['elapsed_seconds']:.3f} 秒）", flush=True)
        inputs, artifacts, quality = collect_outputs(paths)
        if len(inputs) != len(DATA_FILES) or len(artifacts) != len(RESULT_FILES)+1:
            raise RuntimeError("流程结束但缺少预期产物，请检查运行日志。")
        if quality["status"] != "passed" or quality["checks"] != 23:
            raise RuntimeError("流程结束但23项数据质控未全部通过。")
        manifest.update(input_files=inputs, artifacts=artifacts, quality_checks=quality, status="success")
    except (Exception, KeyboardInterrupt) as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        try:
            inputs, artifacts, quality = collect_outputs(paths)
            manifest.update(input_files=inputs, artifacts=artifacts, quality_checks=quality)
        except Exception as collection_error:
            manifest["collection_error"] = f"{type(collection_error).__name__}: {collection_error}"
        print(f"运行失败：{error}", file=sys.stderr, flush=True)
    finally:
        manifest["finished_at_utc"] = utc_now()
        manifest["elapsed_seconds"] = round(time.perf_counter()-started, 3)
        write_manifest(manifest_path, manifest)
    if manifest["status"] == "success":
        print(f"完成。23项质控通过；结果：{paths.results}\n工作簿：{paths.workbook}\n运行记录：{manifest_path}", flush=True)
        return 0
    print(f"失败记录：{manifest_path}", file=sys.stderr, flush=True)
    return 1


def main(argv=None):
    defaults = StudyParameters()
    parser = argparse.ArgumentParser(description="肺结节辅助阅片教学全流程：新输出目录、失败即停、可追溯运行记录。")
    default_name = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S-%fZ")
    parser.add_argument("--output", default="outputs/" + default_name, help="新运行目录；相对路径以项目根目录为基准，默认按时间命名。")
    parser.add_argument("--simulation-seed", type=int, default=defaults.simulation_seed)
    parser.add_argument("--sequence-seed", type=int, default=defaults.sequence_seed)
    parser.add_argument("--bootstrap-seed", type=int, default=defaults.bootstrap_seed)
    parser.add_argument("--bootstrap-iterations", type=int, default=defaults.bootstrap_iterations, help="至少2，默认5000；小次数仅用于调试。")
    parser.add_argument("--plot-language", choices=("auto", "zh", "en"), default="auto", help="auto 优先中文字体，无可用字体时用英文；zh 要求中文字体；en 强制英文。")
    arguments = parser.parse_args(argv)
    try:
        parameters = StudyParameters(arguments.simulation_seed, arguments.sequence_seed, arguments.bootstrap_seed, arguments.bootstrap_iterations)
        output = resolve_output(arguments.output)
        return run_pipeline(output, parameters, arguments.plot_language)
    except (ValueError, OSError) as error:
        print(f"参数或输出路径错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
