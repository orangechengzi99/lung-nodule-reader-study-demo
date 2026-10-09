"""Safety/configuration tests; no full bootstrap is run in these unit tests."""
import contextlib
from dataclasses import asdict
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_all as R
from study_config import load_config, RunPaths, StudyParameters


class ConfigurationTests(unittest.TestCase):
    def test_legacy_paths_and_defaults_are_preserved(self):
        with patch.dict(os.environ, {}, clear=True):
            paths, parameters = load_config(R.SCRIPT_DIR / R.SCRIPTS[0])
        self.assertEqual(paths.data, R.PROJECT_ROOT / "03_模拟数据")
        self.assertEqual(paths.results, R.PROJECT_ROOT / "04_分析结果")
        self.assertEqual(paths.workbook, R.PROJECT_ROOT / "05_综合结果" / "MRMC教学数据与结果.xlsx")
        self.assertEqual(asdict(parameters), {"simulation_seed": 20260920, "sequence_seed": 20260921,
                                             "bootstrap_seed": 20260923, "bootstrap_iterations": 5000})

    def test_isolated_paths_and_each_parameter_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = {"READER_STUDY_RUN_ROOT": temporary, "READER_STUDY_SIMULATION_SEED": "11",
                           "READER_STUDY_SEQUENCE_SEED": "12", "READER_STUDY_BOOTSTRAP_SEED": "13",
                           "READER_STUDY_BOOTSTRAP_ITERATIONS": "20"}
            with patch.dict(os.environ, environment, clear=True):
                paths, parameters = load_config(R.SCRIPT_DIR / R.SCRIPTS[3])
            self.assertEqual(paths, RunPaths.isolated(temporary))
            self.assertEqual(asdict(parameters), {"simulation_seed": 11, "sequence_seed": 12,
                                                 "bootstrap_seed": 13, "bootstrap_iterations": 20})

    def test_invalid_parameters_are_rejected(self):
        for values in ({"simulation_seed": -1}, {"sequence_seed": -1}, {"bootstrap_seed": -1},
                       {"bootstrap_iterations": 1}, {"bootstrap_iterations": 2.5}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                StudyParameters(**values)
        with patch.dict(os.environ, {"READER_STUDY_SIMULATION_SEED": "bad"}, clear=True):
            with self.assertRaisesRegex(ValueError, "READER_STUDY_SIMULATION_SEED"):
                load_config(R.SCRIPT_DIR / R.SCRIPTS[0])


class RunnerSafetyTests(unittest.TestCase):
    def test_relative_output_is_anchored_to_project_not_cwd(self):
        with tempfile.TemporaryDirectory() as temporary:
            fake_project = Path(temporary) / "project"
            expected = fake_project / "outputs" / "new run 中文"
            self.assertEqual(R.resolve_output("outputs/new run 中文", fake_project), expected.resolve())

    def test_project_baseline_code_and_parent_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            for name in (".", "..", "outputs", "02_统计分析代码/new", "03_模拟数据/new", "04_分析结果/new", "docs/new", "tests/new", "outputs/../03_模拟数据/new"):
                with self.subTest(path=name), self.assertRaises(ValueError):
                    R.resolve_output(name, project)

    def test_existing_directory_and_file_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sentinel = root / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            for path in (root, sentinel):
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, "已存在"):
                    R.resolve_output(path)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_new_external_directory_is_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            new = Path(temporary) / "new"
            self.assertEqual(R.resolve_output(new), new.resolve())

    def test_canonical_path_blocks_alias_to_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            baseline = project / "03_模拟数据"
            baseline.mkdir(parents=True)
            outputs = project / "outputs"
            outputs.mkdir()
            alias = outputs / "alias"
            try:
                alias.symlink_to(baseline, target_is_directory=True)
            except OSError:
                self.skipTest("This Windows account cannot create symbolic links.")
            with self.assertRaises(ValueError):
                R.resolve_output(alias / "new", project)

    def test_invalid_cli_returns_nonzero_without_running(self):
        with patch.object(R, "run_pipeline") as pipeline, contextlib.redirect_stderr(StringIO()):
            self.assertEqual(R.main(["--bootstrap-iterations", "1"]), 2)
        pipeline.assert_not_called()

    def test_subprocess_gets_same_interpreter_parameters_and_isolated_paths(self):
        parameters = StudyParameters(11, 12, 13, 20)
        with tempfile.TemporaryDirectory() as temporary:
            paths = RunPaths.isolated(temporary)
            logs = paths.root / "logs"
            logs.mkdir()
            before = os.environ.copy()
            with patch.object(R.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as execute:
                step = R.execute_step(R.SCRIPT_DIR / R.SCRIPTS[0], paths, parameters, logs)
            args, kwargs = execute.call_args
            self.assertEqual(args[0][0], sys.executable)
            self.assertEqual(kwargs["cwd"], R.PROJECT_ROOT)
            self.assertEqual(kwargs["env"]["READER_STUDY_RUN_ROOT"], str(paths.root))
            self.assertEqual(kwargs["env"]["READER_STUDY_PLOT_LANGUAGE"], "auto")
            for key, value in asdict(parameters).items():
                self.assertEqual(kwargs["env"]["READER_STUDY_"+key.upper()], str(value))
            self.assertEqual(os.environ, before)
            self.assertNotIn("shell", kwargs)
            self.assertEqual(step["returncode"], 0)
            self.assertTrue((paths.root / step["stdout_log"]).is_file())

    def test_failure_stops_next_steps_and_saves_failed_manifest(self):
        def fake_execute(script, paths, parameters, log_dir, plot_language="auto"):
            return {"script": script.name, "returncode": 7 if script.name == R.SCRIPTS[1] else 0,
                    "elapsed_seconds": 0, "stdout_log": "logs/mock.out", "stderr_log": "logs/mock.err"}

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "failed"
            with patch.object(R, "execute_step", side_effect=fake_execute) as execute, contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                self.assertEqual(R.run_pipeline(output, StudyParameters()), 1)
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(execute.call_count, 2)
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual([step["script"] for step in manifest["steps"]], list(R.SCRIPTS[:2]))
            self.assertIn("退出码 7", manifest["error"])
            self.assertIn("finished_at_utc", manifest)
            self.assertEqual(manifest["quality_checks"]["status"], "not_available")

    def test_missing_dependency_stops_before_first_script(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "missing"
            with patch.object(R.importlib.metadata, "version", side_effect=R.importlib.metadata.PackageNotFoundError("numpy")), patch.object(R, "execute_step") as execute, contextlib.redirect_stderr(StringIO()):
                self.assertEqual(R.run_pipeline(output, StudyParameters()), 1)
            execute.assert_not_called()
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["steps"], [])
            self.assertIn("缺少依赖 numpy", manifest["error"])

    def test_missing_artifacts_cannot_be_marked_success(self):
        completed = {"script": "mock", "returncode": 0, "elapsed_seconds": 0,
                     "stdout_log": "logs/mock.out", "stderr_log": "logs/mock.err"}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "empty"
            with patch.object(R, "execute_step", return_value=completed), contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                self.assertEqual(R.run_pipeline(output, StudyParameters()), 1)
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("缺少预期产物", manifest["error"])


if __name__ == "__main__":
    unittest.main()
