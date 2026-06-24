import unittest

from data.prepare_finetune import build_sft_record
from utils.tools import DigitSerializer


TEMPLATE = {
    "prompt_input": "SYSTEM\n{input}\n{instruction}\nASSISTANT\n",
}


class PrepareFinetuneTests(unittest.TestCase):
    def setUp(self):
        serializer = DigitSerializer()
        self.record = {
            "raw": {
                "instruction": "1.0,2.0",
                "output": "3.0,4.0",
            },
            "numeric": {
                "instruction": {
                    "digit_context": serializer.serialize([-0.5, 0.5])
                },
                "output": {
                    "digit_context": serializer.serialize([0.7, 0.9])
                },
            },
            "context": {
                "situational_context": "The target date is 2021-05-13.",
                "catch22": {"CO_f1ecac": 4.6154},
            },
            "normalization": {"kind": "identity"},
        }

    def test_builds_full_context_training_record(self):
        result = build_sft_record(
            self.record,
            TEMPLATE,
            pred_len=2,
            include_response=True,
        )

        self.assertIn("<|digit_5|>", result["text"])
        self.assertIn("Catch22", result["text"])
        self.assertIn("target date", result["text"])
        self.assertIn("2021-05-13", result["text"])
        self.assertIn("4.6154", result["text"])
        self.assertIn(self.record["numeric"]["output"]["digit_context"], result["text"])
        self.assertTrue(result["text"].endswith("<|im_end|>\n"))
        self.assertEqual(result["output"], "3.0,4.0")

    def test_can_build_response_free_prompt(self):
        result = build_sft_record(
            self.record,
            TEMPLATE,
            include_response=False,
            pred_len=2,
        )
        self.assertNotIn(
            self.record["numeric"]["output"]["digit_context"],
            result["text"],
        )

    def test_rejects_missing_representation(self):
        record = dict(self.record)
        record.pop("numeric")
        with self.assertRaisesRegex(ValueError, "missing representation"):
            build_sft_record(record, TEMPLATE)

    def test_rejects_output_length_that_disagrees_with_prompt(self):
        with self.assertRaisesRegex(ValueError, "expected pred_len=3"):
            build_sft_record(self.record, TEMPLATE, pred_len=3)


if __name__ == "__main__":
    unittest.main()
