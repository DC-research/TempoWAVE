import unittest

from data.prepare_benchmarks import build_benchmark_record, validate_benchmark_splits


class PrepareBenchmarksTest(unittest.TestCase):
    def test_preserves_official_split_and_context(self):
        split, record = build_benchmark_record(
            {
                "split": "validation",
                "dataset": "AUL",
                "history": [-0.5, 0.0, 0.5],
                "target": [0.25],
                "situational_context": "Region VIC; target date 2021-05-13.",
                "news": "A relevant market event occurred.",
                "catch22": {"CO_f1ecac": 4.6154},
            }
        )

        self.assertEqual(split, "val")
        self.assertEqual(record["dataset"], "AUL")
        self.assertIn("situational_context", record["context"])
        self.assertIn("catch22", record["context"])
        self.assertIn("<|digit_5|>", record["numeric"]["instruction"]["digit_context"])

    def test_rejects_target_length_that_disagrees_with_forecast_horizon(self):
        with self.assertRaisesRegex(ValueError, "expected pred_len=2"):
            build_benchmark_record(
                {
                    "split": "train",
                    "dataset": "AUL",
                    "history": [-0.5, 0.0, 0.5],
                    "target": [0.25],
                },
                pred_len=2,
            )

    def test_requires_a_supported_paper_benchmark(self):
        base = {
            "split": "test",
            "history": [-0.5, 0.0, 0.5],
            "target": [0.25],
        }
        with self.assertRaisesRegex(ValueError, "requires.*dataset"):
            build_benchmark_record(base)
        with self.assertRaisesRegex(ValueError, "Unsupported TempoWAVE benchmark"):
            build_benchmark_record({**base, "dataset": "custom"})

    def test_requires_all_official_splits_for_each_dataset(self):
        splits = {
            "train": [{"dataset": "AUL"}, {"dataset": "BIT"}],
            "val": [{"dataset": "AUL"}],
            "test": [{"dataset": "AUL"}, {"dataset": "BIT"}],
        }
        with self.assertRaisesRegex(ValueError, "BIT.*val"):
            validate_benchmark_splits(splits)


if __name__ == "__main__":
    unittest.main()
