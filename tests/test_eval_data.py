import json
import tempfile
import unittest
from pathlib import Path

from evaluation.run_eval import load_eval_records


class EvalDataTest(unittest.TestCase):
    def write_records(self, records):
        tmpdir = tempfile.TemporaryDirectory()
        path = Path(tmpdir.name) / "eval.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        self.addCleanup(tmpdir.cleanup)
        return path

    def test_load_eval_records_accepts_valid_schema(self):
        path = self.write_records([
            {
                "dataset": "aul",
                "text": "prompt",
                "input": "1.0,2.0",
                "output": "3.0",
                "normalization": {
                    "kind": "affine",
                    "offset": 1.0,
                    "scale": 2.0,
                    "normalized_offset": -0.5,
                },
            }
        ])

        records = load_eval_records(path, hist_len=2, pred_len=1)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["text"], "prompt")
        self.assertEqual(records[0]["input"].tolist(), [1.0, 2.0])
        self.assertEqual(records[0]["output"].tolist(), [3.0])
        self.assertEqual(records[0]["normalization"]["kind"], "affine")
        self.assertEqual(records[0]["dataset"], "AUL")

    def test_load_eval_records_rejects_wrong_lengths(self):
        path = self.write_records([
            {
                "dataset": "AUL",
                "text": "prompt",
                "input": "1.0,2.0",
                "output": "3.0",
            }
        ])

        with self.assertRaisesRegex(ValueError, "expected hist_len=3"):
            load_eval_records(path, hist_len=3, pred_len=1)

    def test_load_eval_records_rejects_missing_fields(self):
        path = self.write_records([{"text": "prompt", "input": "1.0"}])

        with self.assertRaisesRegex(ValueError, "missing required"):
            load_eval_records(path, hist_len=1, pred_len=1)

    def test_load_eval_records_rejects_non_benchmark_dataset(self):
        path = self.write_records([
            {
                "dataset": "custom",
                "text": "prompt",
                "input": "1.0",
                "output": "2.0",
            }
        ])

        with self.assertRaisesRegex(ValueError, "Unsupported TempoWAVE benchmark"):
            load_eval_records(path, hist_len=1, pred_len=1)


if __name__ == "__main__":
    unittest.main()
