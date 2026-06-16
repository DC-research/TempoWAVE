import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from training.embeddings.inject import (
    freeze_tempowave_input_embeddings,
    untie_output_embeddings,
)
from training.embeddings.mwne import TempoWaveEmbedding


class TempoWaveEmbeddingTest(unittest.TestCase):
    def test_builds_ten_injective_digit_embeddings(self):
        model = TempoWaveEmbedding(
            embedding_dim=64,
            wavelet_types=["haar", "db4", "mexh"],
            scales=[1.0, 2.0, 4.0],
            device="cpu",
        )

        embeddings = model(torch.arange(10))

        self.assertEqual(tuple(model.feature_table.shape), (10, 9))
        self.assertEqual(tuple(embeddings.shape), (10, 64))
        self.assertEqual(torch.unique(embeddings, dim=0).shape[0], 10)
        self.assertGreater(model.minimum_separation().item(), 0)

    def test_digit_zero_is_a_real_codeword(self):
        model = TempoWaveEmbedding(
            embedding_dim=16,
            wavelet_types=["mexh"],
            scales=[1.0, 2.0],
            alignment="project",
            device="cpu",
        )

        self.assertTrue(torch.any(model(torch.tensor([0])) != 0))

    def test_rejects_non_digit_inputs(self):
        model = TempoWaveEmbedding(embedding_dim=16, device="cpu")
        with self.assertRaisesRegex(ValueError, "digit IDs"):
            model(torch.tensor([10]))

    def test_unties_qwen_output_head_before_freezing_input_codebook(self):
        model = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                tie_word_embeddings=True,
            )
        )
        self.assertIs(
            model.get_input_embeddings().weight,
            model.get_output_embeddings().weight,
        )

        self.assertTrue(untie_output_embeddings(model))
        freeze_tempowave_input_embeddings(model)

        self.assertIsNot(
            model.get_input_embeddings().weight,
            model.get_output_embeddings().weight,
        )
        self.assertFalse(model.config.tie_word_embeddings)
        self.assertFalse(model.get_input_embeddings().weight.requires_grad)
        self.assertTrue(model.get_output_embeddings().weight.requires_grad)

    def test_rejects_freezing_a_tied_output_head(self):
        model = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                tie_word_embeddings=True,
            )
        )
        with self.assertRaisesRegex(ValueError, "untied"):
            freeze_tempowave_input_embeddings(model)


if __name__ == "__main__":
    unittest.main()
