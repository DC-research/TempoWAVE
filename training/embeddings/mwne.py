"""Paper-faithful Multi-Wavelet Number Embedding digit codebook."""

import math

import torch
import torch.nn as nn


_DB4_LOW_PASS = (
    -0.010597401785069032,
    0.0328830116668852,
    0.030841381835560764,
    -0.18703481171888114,
    -0.027983769416859854,
    0.6308807679298587,
    0.7148465705529154,
    0.2303778133088965,
)


def _cascade_wavelet(low_pass, level=10):
    """Approximate a compactly supported mother wavelet from its filter bank."""

    dtype = torch.float64
    low_pass = torch.tensor(low_pass, dtype=dtype)
    high_pass = torch.tensor(
        [((-1) ** index) * value for index, value in enumerate(reversed(low_pass.tolist()))],
        dtype=dtype,
    )

    scaling = torch.ones(1, dtype=dtype)
    for _ in range(level - 1):
        upsampled = torch.zeros(2 * scaling.numel() - 1, dtype=dtype)
        upsampled[::2] = scaling
        scaling = torch.conv1d(
            upsampled.view(1, 1, -1),
            (math.sqrt(2.0) * low_pass).flip(0).view(1, 1, -1),
            padding=low_pass.numel() - 1,
        ).reshape(-1)

    upsampled = torch.zeros(2 * scaling.numel() - 1, dtype=dtype)
    upsampled[::2] = scaling
    wavelet = torch.conv1d(
        upsampled.view(1, 1, -1),
        (math.sqrt(2.0) * high_pass).flip(0).view(1, 1, -1),
        padding=high_pass.numel() - 1,
    ).reshape(-1)
    support = torch.linspace(0.0, float(low_pass.numel() - 1), wavelet.numel(), dtype=dtype)
    return support, wavelet


class TempoWaveEmbedding(nn.Module):
    """Construct the ten digit embeddings described in the TempoWAVE paper."""

    supported_wavelets = {"haar", "db4", "mexh", "morlet"}

    def __init__(
        self,
        embedding_dim,
        wavelet_types=("haar", "db4", "mexh"),
        scales=(1.0, 2.0, 4.0),
        grid_resolution=1000,
        alignment="project",
        projection_seed=3407,
        device=None,
    ):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if grid_resolution < 10:
            raise ValueError("grid_resolution must be at least 10")
        if not scales or any(scale <= 0 for scale in scales):
            raise ValueError("scales must contain positive values")

        wavelet_types = tuple(wavelet_types)
        unknown = sorted(set(wavelet_types) - self.supported_wavelets)
        if unknown:
            raise ValueError(f"Unsupported wavelet type(s): {unknown}")

        self.embedding_dim = int(embedding_dim)
        self.wavelet_types = wavelet_types
        self.scales = tuple(float(scale) for scale in scales)
        self.grid_resolution = int(grid_resolution)
        self.alignment = alignment

        grid = torch.linspace(0.0, 1.0, self.grid_resolution, dtype=torch.float64)
        digit_indices = torch.round(
            torch.arange(10, dtype=torch.float64) / 9.0 * (self.grid_resolution - 1)
        ).long()
        coefficients = []
        for wavelet_name in self.wavelet_types:
            for scale in self.scales:
                sampled = self._sample_scaled_wavelet(wavelet_name, grid, scale)
                coefficients.append(sampled[digit_indices])
        feature_table = torch.stack(coefficients, dim=1).float()
        self.register_buffer("feature_table", feature_table)

        feature_dim = feature_table.shape[1]
        if alignment == "pad":
            if feature_dim > self.embedding_dim:
                raise ValueError(
                    f"Cannot pad {feature_dim} features into embedding_dim={self.embedding_dim}"
                )
            codebook = torch.zeros(10, self.embedding_dim)
            codebook[:, :feature_dim] = feature_table
        elif alignment == "project":
            generator = torch.Generator().manual_seed(projection_seed)
            projection = torch.randn(
                feature_dim,
                self.embedding_dim,
                generator=generator,
            ) / math.sqrt(feature_dim)
            self.register_buffer("projection", projection)
            codebook = feature_table @ projection
        else:
            raise ValueError("alignment must be 'pad' or 'project'")

        self.register_buffer("codebook", codebook)
        self._validate_injective(self.feature_table, "wavelet feature")
        self._validate_injective(self.codebook, "aligned embedding")
        self.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    @staticmethod
    def _interpolate(samples_x, samples_y, query):
        query = query.clamp(float(samples_x[0]), float(samples_x[-1]))
        indices = torch.searchsorted(samples_x, query).clamp(1, samples_x.numel() - 1)
        left = indices - 1
        right = indices
        weight = (query - samples_x[left]) / (samples_x[right] - samples_x[left])
        return samples_y[left] + weight * (samples_y[right] - samples_y[left])

    def _mother_wavelet(self, name, x):
        if name == "haar":
            result = torch.zeros_like(x)
            result[(x >= 0.0) & (x < 0.5)] = 1.0
            result[(x >= 0.5) & (x < 1.0)] = -1.0
            return result
        if name == "mexh":
            return (1.0 - x**2) * torch.exp(-(x**2) / 2.0)
        if name == "morlet":
            return torch.cos(5.0 * x) * torch.exp(-(x**2) / 2.0)
        if name == "db4":
            support, wavelet = _cascade_wavelet(_DB4_LOW_PASS)
            return self._interpolate(support.to(x.device), wavelet.to(x.device), x)
        raise AssertionError(f"Unhandled wavelet: {name}")

    def _sample_scaled_wavelet(self, name, grid, scale):
        # Equation (1): psi_s,0(t) = 1/sqrt(s) * psi(t/s).
        return self._mother_wavelet(name, grid / scale) / math.sqrt(scale)

    @staticmethod
    def _validate_injective(table, label):
        distances = torch.cdist(table.float(), table.float())
        distances.fill_diagonal_(float("inf"))
        minimum = distances.min().item()
        if not math.isfinite(minimum) or minimum <= 1e-8:
            raise ValueError(f"TempoWAVE {label} codebook is not injective")

    def minimum_separation(self):
        distances = torch.cdist(self.codebook.float(), self.codebook.float())
        distances.fill_diagonal_(float("inf"))
        return distances.min()

    def forward(self, digits):
        digits = torch.as_tensor(digits, device=self.codebook.device)
        if torch.any((digits < 0) | (digits > 9)):
            raise ValueError("TempoWAVE inputs must be digit IDs in [0, 9]")
        return self.codebook[digits.long()]


# Backward-compatible import name; the semantics now match the paper's digit codebook.
MWNE = TempoWaveEmbedding
