"""Numeric formatting and reversible normalization for TempoWAVE."""

import re

import numpy as np


DIGIT_TOKENS = tuple(f"<|digit_{digit}|>" for digit in range(10))


def validate_digit_tokenizer(tokenizer):
    vocabulary = tokenizer.get_vocab()
    missing = [token for token in DIGIT_TOKENS if token not in vocabulary]
    if missing:
        raise ValueError(
            "Checkpoint tokenizer is missing TempoWAVE digit tokens: "
            + ", ".join(missing)
        )
    token_ids = [vocabulary[token] for token in DIGIT_TOKENS]
    if len(set(token_ids)) != 10:
        raise ValueError("TempoWAVE digit tokens must map to ten distinct token IDs")
    return token_ids


class AffineNormalizer:
    """Map values through a reversible affine transform."""

    def __init__(self, offset=0.0, scale=1.0, normalized_offset=0.0):
        self.offset = float(offset)
        self.scale = float(scale)
        self.normalized_offset = float(normalized_offset)

    @classmethod
    def fit_history_minmax(cls, history, normalized_low=-0.5):
        history = np.asarray(history, dtype=float)
        if history.ndim != 1 or history.size == 0:
            raise ValueError("history must be a non-empty one-dimensional array")
        if not np.isfinite(history).all():
            raise ValueError("history contains NaN or infinite values")

        minimum = float(history.min())
        value_range = float(history.max() - minimum)
        if value_range == 0:
            value_range = 1.0
        return cls(
            offset=minimum,
            scale=value_range,
            normalized_offset=normalized_low,
        )

    @classmethod
    def from_metadata(cls, metadata):
        if not metadata or metadata.get("kind", "identity") == "identity":
            return cls()
        if metadata.get("kind") != "affine":
            raise ValueError(f"Unsupported normalization kind: {metadata.get('kind')!r}")
        return cls(
            offset=metadata["offset"],
            scale=metadata["scale"],
            normalized_offset=metadata.get("normalized_offset", 0.0),
        )

    def transform(self, values):
        values = np.asarray(values, dtype=float)
        return (values - self.offset) / self.scale + self.normalized_offset

    def inverse_transform(self, values):
        values = np.asarray(values, dtype=float)
        return (values - self.normalized_offset) * self.scale + self.offset

    def metadata(self):
        return {
            "kind": "affine",
            "offset": self.offset,
            "scale": self.scale,
            "normalized_offset": self.normalized_offset,
        }


class DigitSerializer:
    """Render every decimal digit as one dedicated tokenizer token."""

    def __init__(
        self,
        integer_precision=1,
        fractional_precision=4,
        value_separator=", ",
        digit_tokens=DIGIT_TOKENS,
    ):
        if integer_precision <= 0 or fractional_precision < 0:
            raise ValueError("integer_precision must be positive and fractional_precision non-negative")
        if len(digit_tokens) != 10:
            raise ValueError("TempoWAVE requires exactly ten dedicated digit tokens")

        self.integer_precision = integer_precision
        self.fractional_precision = fractional_precision
        self.value_separator = value_separator
        self.digit_tokens = tuple(digit_tokens)
        self._token_to_digit = {
            token: str(digit) for digit, token in enumerate(self.digit_tokens)
        }
        token_pattern = "|".join(re.escape(token) for token in self.digit_tokens)
        self._digit_token_pattern = re.compile(token_pattern)
        digit_group = f"(?:{token_pattern})"
        fractional = (
            rf"\.{digit_group}{{{self.fractional_precision}}}"
            if self.fractional_precision
            else ""
        )
        self._encoded_number_pattern = re.compile(
            rf"[-+]?{digit_group}{{{self.integer_precision}}}{fractional}"
        )
        plain_fractional = (
            rf"\.\d{{{self.fractional_precision}}}"
            if self.fractional_precision
            else ""
        )
        self._plain_fixed_pattern = re.compile(
            rf"(?<![A-Za-z0-9_])[-+]?\d{{{self.integer_precision}}}"
            rf"{plain_fractional}(?![A-Za-z0-9_])"
        )

    def _encode_digits(self, text):
        return "".join(self.digit_tokens[int(character)] for character in text)

    def serialize_number(self, value):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("TempoWAVE fixed-precision values must be finite")

        sign = "-" if np.signbit(value) else ""
        fixed = f"{abs(value):.{self.fractional_precision}f}"
        integer, _, fractional = fixed.partition(".")
        if len(integer) > self.integer_precision:
            raise ValueError(
                f"{value} exceeds integer_precision={self.integer_precision}"
            )
        integer = integer.zfill(self.integer_precision)
        encoded = sign + self._encode_digits(integer)
        if self.fractional_precision:
            encoded += "." + self._encode_digits(fractional)
        return encoded

    def serialize(self, values):
        return self.value_separator.join(
            self.serialize_number(value) for value in np.asarray(values).reshape(-1)
        )

    def detokenize_digits(self, text):
        return self._digit_token_pattern.sub(
            lambda match: self._token_to_digit[match.group(0)],
            text,
        )

    def inverse_serialize(self, text):
        encoded_matches = self._encoded_number_pattern.findall(text)
        if encoded_matches:
            return np.asarray(
                [
                    float(self.detokenize_digits(encoded_number))
                    for encoded_number in encoded_matches
                ],
                dtype=float,
            )
        return np.asarray(
            [float(match.group(0)) for match in self._plain_fixed_pattern.finditer(text)],
            dtype=float,
        )

# Kept as the short public name used by earlier release scripts.
Serializer = DigitSerializer
