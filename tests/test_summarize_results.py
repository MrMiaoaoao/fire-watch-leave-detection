import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_results import check_current, find_results


def write_result(path, total_alerts, warn_times, clear_times):
    alerts = [{"type": "warn", "time_s": t} for t in warn_times]
    alerts += [{"type": "clear", "time_s": t} for t in clear_times]
    path.write_text(
        json.dumps({
            "leave_monitor": {
                "summary": {
                    "total_alerts": total_alerts,
                    "total_duration_s": 10.0,
                    "fps": 25.0,
                },
                "alerts": alerts,
            }
        }, ensure_ascii=False),
        encoding="utf-8",
    )


class SummarizeResultsTest(unittest.TestCase):
    def test_find_results_skips_probe_files_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_result(root / "监火员离岗测试_result.json", 1, [96.6], [104.3])
            write_result(root / "监火员离岗测试_probe_result.json", 0, [], [])

            results = find_results(root, include_probes=False)

        self.assertEqual([r.stem for r in results], ["监火员离岗测试"])

    def test_current_gate_detects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_result(root / "监火员离岗测试_result.json", 2, [96.6, 120.0], [104.3])
            results = find_results(root, include_probes=False)

        errors = check_current(results, tolerance=0.2)
        self.assertTrue(any("total_alerts" in e for e in errors))
        self.assertTrue(any("missing result" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
