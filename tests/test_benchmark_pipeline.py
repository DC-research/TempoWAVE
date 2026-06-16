import json
import tempfile
import unittest
from pathlib import Path

from data.benchmark_pipeline import prepare_benchmark_outputs
from data.prepare_benchmarks import build_benchmark_record


TEMPLATE = {
    "prompt_input": "SYSTEM\n{input}\n{instruction}\nASSISTANT\n",
}


class BenchmarkPipelineTest(unittest.TestCase):
    def test_writes_dataset_specific_training_and_eval_artifacts(self):
        records = {"train": [], "val": [], "test": []}
        for split in records:
            _, record = build_benchmark_record(
                {
                    "split": split,
                    "dataset": "AUL",
                    "history": [-0.5, 0.0],
                    "target": [0.5],
                },
                pred_len=1,
            )
            records[split].append(record)

        with tempfile.TemporaryDirectory() as tmpdir:
            prepare_benchmark_outputs(
                records,
                tmpdir,
                TEMPLATE,
                pred_len=1,
                horizon_label="step",
            )
            root = Path(tmpdir)
            eval_records = json.loads(
                (root / "eval" / "eval.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (root / "benchmark_manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(eval_records[0]["dataset"], "AUL")
        self.assertEqual(manifest["datasets"]["AUL"]["splits"]["test"], 1)
        self.assertIn("reference", manifest["datasets"]["AUL"])


if __name__ == "__main__":
    unittest.main()
