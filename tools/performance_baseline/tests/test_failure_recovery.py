"""CPU-only regressions for baseline process recovery and JSON output."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from unittest import mock


BASELINE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "baseline_extended", BASELINE / "benchmark_nvidia_extended.py"
)
EXTENDED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXTENDED)


class SharedRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="baseline-recovery-"
        )
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.binary = self.directory / "fake_native"
        self.logs = self.directory / "logs"
        self.logs.mkdir()

    def write_binary(self, body):
        self.binary.write_text(
            "#!" + sys.executable + "\n" + textwrap.dedent(body)
        )
        self.binary.chmod(0o700)

    def run_cases(self, names, timeout=1.0):
        args = types.SimpleNamespace(
            binary=self.binary, extra_variants=False, timeout=timeout
        )
        cases = [
            dict(case_name=name, op="fake", status="catalog") for name in names
        ]
        output = io.StringIO()
        # Bound a broken no-progress retry loop while preserving
        # real processes.
        original_popen = subprocess.Popen
        launches = 0

        def bounded_popen(*args, **kwargs):
            nonlocal launches
            launches += 1
            self.assertLessEqual(
                launches,
                len(cases) + 1,
                "shared runner restarted without making progress",
            )
            return original_popen(*args, **kwargs)

        with mock.patch.object(EXTENDED.subprocess, "Popen", bounded_popen):
            with contextlib.redirect_stderr(io.StringIO()):
                EXTENDED.run_shared(args, cases, set(), output, self.logs)
        return [json.loads(line) for line in output.getvalue().splitlines()]

    def test_closed_stdout_still_obeys_timeout(self):
        self.write_binary(
            """
            import os, time
            os.close(1)
            time.sleep(0.5)
        """
        )
        rows = self.run_cases(["closed_stdout"], timeout=0.1)
        self.assertEqual(rows[0]["stage"], "timeout")
        self.assertEqual(rows[0]["status"], "failed")

    def test_exit_three_without_result_advances_to_next_case(self):
        self.write_binary(
            """
            import json, sys
            from pathlib import Path
            names = (
                Path(sys.argv[sys.argv.index('--case-file') + 1])
                .read_text()
                .splitlines()
            )
            if names[0] == 'missing':
                raise SystemExit(3)
            for name in names:
                print(
                    json.dumps(dict(case_name=name, status='ok')),
                    flush=True,
                )
        """
        )
        rows = self.run_cases(["missing", "following"])
        self.assertEqual(
            [(r["case_name"], r["status"]) for r in rows],
            [("missing", "failed"), ("following", "ok")],
        )
        self.assertEqual(rows[0]["stage"], "process")
        self.assertIn("exit=3", rows[0]["reason"])

    def test_reported_failure_exit_three_preserves_following_case(self):
        self.write_binary(
            """
            import json, sys
            from pathlib import Path
            names = (
                Path(sys.argv[sys.argv.index('--case-file') + 1])
                .read_text()
                .splitlines()
            )
            if names[0] == 'reported_failure':
                print(
                    json.dumps(dict(
                        case_name=names[0], status='failed', stage='build'
                    )),
                    flush=True,
                )
                raise SystemExit(3)
            for name in names:
                print(
                    json.dumps(dict(case_name=name, status='ok')),
                    flush=True,
                )
        """
        )
        rows = self.run_cases(["reported_failure", "following"])
        self.assertEqual(
            [(r["case_name"], r["status"]) for r in rows],
            [("reported_failure", "failed"), ("following", "ok")],
        )

    def test_invalid_timeouts_fail_before_invoking_binary(self):
        for value in ("0", "-1", "nan", "inf", "-inf"):
            with self.subTest(timeout=value):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(BASELINE / "benchmark_nvidia_extended.py"),
                        "--skip-build",
                        "--binary",
                        str(self.directory / "missing"),
                        "--timeout=" + value,
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("positive finite", result.stderr)

    def test_build_requires_explicit_cudnn_root(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(EXTENDED.subprocess, "run") as run:
                with self.assertRaisesRegex(SystemExit, "CUDNN_ROOT"):
                    EXTENDED.build(self.directory / "benchmark")
                run.assert_not_called()

    def test_relative_binary_builds_and_runs_from_callers_directory(self):
        compiler = self.directory / "fake_compiler"
        executable = (
            "#!" + sys.executable + '\nprint(\'{"case_name":"cpu","op":"fake",'
            '"status":"blocked"}\')\n'
        )
        compiler.write_text(
            "#!"
            + sys.executable
            + "\n"
            + textwrap.dedent(
                f"""
            import sys
            from pathlib import Path
            output = Path(sys.argv[sys.argv.index('-o') + 1])
            output.write_text({executable!r})
            output.chmod(0o700)
        """
            )
        )
        compiler.chmod(0o700)
        environment = dict(
            os.environ,
            CXX=str(compiler),
            CUDNN_ROOT=str(self.directory / "cudnn"),
        )
        tilde_binary = str(
            Path("~")
            / os.path.relpath(self.directory, Path.home())
            / "nested/benchmark"
        )
        for binary, cudnn_root in (
            ("nested/benchmark", str(self.directory / "cudnn")),
            (tilde_binary, str(self.directory / "cudnn")),
            ("nested/benchmark", "/usr"),
        ):
            environment["CUDNN_ROOT"] = cudnn_root
            environment["CUDNN_FRONTEND_INCLUDE_DIR"] = str(
                self.directory / "frontend"
            )
            with self.subTest(binary=binary, cudnn_root=cudnn_root):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(BASELINE / "benchmark_nvidia_extended.py"),
                        "--build-only",
                        "--binary",
                        binary,
                        "--output",
                        str(self.directory / "catalog.jsonl"),
                    ],
                    cwd=self.directory,
                    env=environment,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(
                    (self.directory / "nested/benchmark").is_file()
                )
                rows = (
                    (self.directory / "catalog.catalog.jsonl")
                    .read_text()
                    .splitlines()
                )
                self.assertEqual(json.loads(rows[0])["case_name"], "cpu")


@unittest.skipUnless(shutil.which("c++"), "CPU C++ compiler is required")
class PointwiseJsonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="baseline-json-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.binary = Path(cls.temporary.name) / "emit"
        source = (BASELINE / "benchmark_nvidia_pointwise_gaps.cpp").read_text()
        # Compile the actual production emitter. Only its GPU-facing metadata
        # dependencies are replaced, so no CUDA installation or device is used.
        emitter = source[
            source.index("void emit(") : source.index(
                "}  // namespace gap_reference"
            )
        ]
        declarations = r"""
            #include <cmath>
            #include <ctime>
            #include <iomanip>
            #include <iostream>
            #include <limits>
            #include <sstream>
            #include <string>
            #include <string_view>
            #include <vector>
            struct Tensor { int data_type = 0; };
            struct Attributes {
              int flags = 0;
              double relu_lower_clip = 0, relu_upper_clip = 1;
              double relu_lower_clip_slope = 0, swish_beta = 1;
            };
            struct Spec {
              std::string name = "nonfinite_output";
              std::vector<Tensor> inputs{Tensor{}};
              Tensor output;
              int mode = 0;
              double alpha = 1, absolute_tolerance = 0, relative_tolerance = 0;
              Attributes attributes;
            };
            struct Entry {
              Spec spec;
              std::string op = "relu_backward", group = "attributes";
              bool add = false, square = false;
            };
            struct Result {
              std::string stage = "correctness", correctness = "FAIL";
              std::string error = "CPU attribute derivative mismatch index=0";
              std::size_t checked = 0, workspace = 0;
              double max_abs = 0, max_rel = 0, build_us = 0, warmup_us = 0;
              std::vector<double> samples, batches;
            };
            std::string quote(std::string_view value) {
              return "\"" + std::string(value) + "\"";
            }
            const char* dtype(int) { return "fp32"; }
            int cudnnGetVersion() { return 0; }
            void tensor(
                std::ostream& out, const Tensor&, std::string_view) {
              out << "{}";
            }
            template<class T>
            void array(std::ostream& out, const std::vector<T>&) {
              out << "[]";
            }
            namespace io {
              std::size_t element_count(const Tensor&) { return 1; }
            }
            namespace timing {
              double percentile(const std::vector<double>&, double) {
                return 0;
              }
            }
        """
        main = r"""
            int main() {
              for (double value : {std::numeric_limits<double>::infinity(),
                                   -std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::quiet_NaN(),
                                   0.123456789123}) {
                Result result; result.max_abs = value; result.max_rel = value;
                emit(Entry{}, result, "CPU test", 0, 0);
              }
            }
        """
        subprocess.run(
            ["c++", "-std=c++17", "-x", "c++", "-", "-o", str(cls.binary)],
            input=declarations + emitter + main,
            text=True,
            check=True,
            capture_output=True,
        )

    def test_nonfinite_error_metrics_remain_valid_json(self):
        output = subprocess.check_output([str(self.binary)], text=True)
        try:
            rows = [json.loads(line) for line in output.splitlines()]
        except json.JSONDecodeError as error:
            self.fail(
                f"native correctness failure produced invalid JSON: {error}"
            )
        for row in rows[:3]:
            self.assertEqual(row["status"], "ERROR")
            self.assertEqual(row["stage"], "correctness")
            self.assertIn("mismatch index=0", row["error"])
            self.assertIsNone(row["correctness"]["max_abs"])
            self.assertIsNone(row["correctness"]["max_rel"])
        self.assertEqual(rows[3]["correctness"]["max_abs"], 0.123456789123)
        self.assertEqual(rows[3]["correctness"]["max_rel"], 0.123456789123)


if __name__ == "__main__":
    unittest.main()
