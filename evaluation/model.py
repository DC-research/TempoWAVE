"""TempoWAVE generation and fixed-precision numeric parsing."""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from utils.tools import AffineNormalizer, DigitSerializer, validate_digit_tokenizer


class TimeSeriesForecaster:
    def __init__(
        self,
        model_path,
        pred_len,
        integer_precision=1,
        fractional_precision=4,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        seed=42,
    ):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        self.pred_len = pred_len
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.serializer = DigitSerializer(
            integer_precision=integer_precision,
            fractional_precision=fractional_precision,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        validate_digit_tokenizer(self.tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.generator = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def predict(self, prompt, normalization=None):
        digits_per_value = (
            self.serializer.integer_precision
            + self.serializer.fractional_precision
        )
        max_new_tokens = self.pred_len * (digits_per_value + 3) + 16
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "do_sample": self.do_sample,
            "num_return_sequences": 1,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.do_sample:
            generation_args.update(
                {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                }
            )

        generated = self.generator(prompt, **generation_args)[0]["generated_text"]
        completion = generated[len(prompt) :] if generated.startswith(prompt) else generated
        normalized_prediction = self.serializer.inverse_serialize(completion)
        normalized_prediction = normalized_prediction[: self.pred_len]

        invalid_count = max(0, self.pred_len - len(normalized_prediction))
        if invalid_count:
            normalized_prediction = np.concatenate(
                [
                    normalized_prediction,
                    np.full(invalid_count, np.nan, dtype=float),
                ]
            )

        normalizer = AffineNormalizer.from_metadata(normalization)
        prediction = normalizer.inverse_transform(normalized_prediction)
        return prediction, {
            "completion": completion,
            "invalid_count": int(np.isnan(prediction).sum()),
        }
