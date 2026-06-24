import unittest

import numpy as np

from utils.tools import (
    AffineNormalizer,
    DIGIT_TOKENS,
    DigitSerializer,
    validate_digit_tokenizer,
)


class DigitSerializerTest(unittest.TestCase):
    def test_fixed_precision_round_trip(self):
        serializer = DigitSerializer(integer_precision=1, fractional_precision=4)
        encoded = serializer.serialize([-0.5, 0.1234])

        self.assertIn(DIGIT_TOKENS[5], encoded)
        self.assertNotIn("###", encoded)
        np.testing.assert_allclose(
            serializer.inverse_serialize(encoded),
            np.array([-0.5, 0.1234]),
        )

    def test_parser_falls_back_to_plain_fixed_precision_numbers(self):
        serializer = DigitSerializer()
        np.testing.assert_allclose(
            serializer.inverse_serialize("prediction: -0.2500, 0.5000"),
            np.array([-0.25, 0.5]),
        )

    def test_parser_prioritizes_digit_token_values_over_prose_numbers(self):
        serializer = DigitSerializer()
        encoded = serializer.serialize([0.25, 0.5])
        actual = serializer.inverse_serialize(f"Here are 2 values: {encoded}")
        np.testing.assert_allclose(actual, [0.25, 0.5])

    def test_history_minmax_is_reversible(self):
        normalizer = AffineNormalizer.fit_history_minmax([10.0, 20.0])
        normalized = normalizer.transform([10.0, 15.0, 20.0])
        np.testing.assert_allclose(normalized, [-0.5, 0.0, 0.5])
        np.testing.assert_allclose(
            normalizer.inverse_transform(normalized),
            [10.0, 15.0, 20.0],
        )

    def test_checkpoint_requires_all_ten_digit_tokens(self):
        class Tokenizer:
            def get_vocab(self):
                return {token: index for index, token in enumerate(DIGIT_TOKENS)}

        self.assertEqual(validate_digit_tokenizer(Tokenizer()), list(range(10)))

        class IncompleteTokenizer:
            def get_vocab(self):
                return {token: index for index, token in enumerate(DIGIT_TOKENS[:-1])}

        with self.assertRaisesRegex(ValueError, "missing TempoWAVE"):
            validate_digit_tokenizer(IncompleteTokenizer())


if __name__ == "__main__":
    unittest.main()
