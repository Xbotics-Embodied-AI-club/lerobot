# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RunningQuantileStats，原样从 compute_stats 搬出来、只依赖 numpy（Xbotics）。

单独成文件，是为了让边录边编的统计进程（:mod:`lerobot.datasets.stats_worker`）按文件路径直接加载它，
不导入 lerobot 包（``compute_stats`` 一导入就带上 ``lerobot.processor`` 与 torch，端侧计算平台上要几十秒）。
``compute_stats`` 照旧导出这两个名字。
"""

from __future__ import annotations

import numpy as np

DEFAULT_QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]


class RunningQuantileStats:
    """
    Maintains running statistics for batches of vectors, including mean,
    standard deviation, min, max, and approximate quantiles.

    Statistics are computed per feature dimension and updated incrementally
    as new batches are observed. Quantiles are estimated using histograms,
    which adapt dynamically if the observed data range expands.
    """

    def __init__(self, quantile_list: list[float] | None = None, num_quantile_bins: int = 5000):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = num_quantile_bins

        self._quantile_list = quantile_list
        if self._quantile_list is None:
            self._quantile_list = DEFAULT_QUANTILES
        self._quantile_keys = [f"q{int(q * 100):02d}" for q in self._quantile_list]

    def update(self, batch: np.ndarray) -> None:
        """Update the running statistics with a batch of vectors.

        Args:
            batch: An array where all dimensions except the last are batch dimensions.
        """
        batch = batch.reshape(-1, batch.shape[-1])
        raw_uint8 = batch if batch.dtype == np.uint8 else None
        # Promote integer and low-precision inputs before computing squared statistics.
        batch = batch.astype(np.result_type(batch.dtype, np.float32), copy=False)
        num_elements, vector_length = batch.shape

        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")

            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (
            num_elements / self._count
        )

        self._update_histograms(batch, raw_uint8)

    def get_statistics(self) -> dict[str, np.ndarray]:
        """Compute and return the statistics of the vectors processed so far.

        Args:
            quantiles: List of quantiles to compute (e.g., [0.01, 0.10, 0.50, 0.90, 0.99]). If None, no quantiles computed.

        Returns:
            Dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2

        stddev = np.sqrt(np.maximum(0, variance))

        stats = {
            "min": self._min.copy(),
            "max": self._max.copy(),
            "mean": self._mean.copy(),
            "std": stddev,
            "count": np.array([self._count]),
        }

        quantile_results = self._compute_quantiles()
        for i, q in enumerate(self._quantile_keys):
            stats[q] = quantile_results[i]

        return stats

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            old_hist = self._histograms[i]

            # Create new edges with small padding to ensure range coverage
            padding = (self._max[i] - self._min[i]) * 1e-10
            new_edges = np.linspace(
                self._min[i] - padding, self._max[i] + padding, self._num_quantile_bins + 1
            )

            # Redistribute existing histogram counts to new bins
            # We need to map each old bin center to the new bins
            old_centers = (old_edges[:-1] + old_edges[1:]) / 2

            # Xbotics：整批算出每个旧桶落进哪个新桶再累加，结果与逐桶循环逐位相同。逐桶的 Python 循环
            # （5000 桶 × 每通道）在端侧计算平台上一次 50 ms 且一直占着 GIL，边录边编时录制循环被它拖得掉帧。
            occupied = old_hist > 0
            bin_idx = np.searchsorted(new_edges, old_centers[occupied]) - 1
            bin_idx = np.clip(bin_idx, 0, self._num_quantile_bins - 1)
            new_hist = np.bincount(
                bin_idx, weights=old_hist[occupied], minlength=self._num_quantile_bins
            ).astype(np.float64)

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray, raw_uint8: np.ndarray | None = None) -> None:
        """Update histograms with new vectors.

        Xbotics：输入本是 uint8（相机画面）时，每个通道至多 256 种取值：先数每个取值出现几次，再只把出现过的取值
        带着次数分桶。uint8 转 float32 是精确的；np.histogram 在边界是数组时带不带权重都用同一个
        ``_search_sorted_inclusive`` 定桶，计数是整数累加，所以结果与逐个像素分桶逐位相同。
        端侧计算平台上一帧（160×120×3）从约 11 ms 降到 1 ms 内，边录边编时不再拖住录制线程。
        """
        for i in range(batch.shape[1]):
            if raw_uint8 is not None:
                counts = np.bincount(raw_uint8[:, i], minlength=256)
                present = np.flatnonzero(counts)
                hist, _ = np.histogram(
                    present.astype(batch.dtype), bins=self._bin_edges[i], weights=counts[present]
                )
            else:
                hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self) -> list[np.ndarray]:
        """Compute quantiles based on histograms."""
        results = []
        for q in self._quantile_list:
            target_count = q * self._count
            q_values = []

            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                q_value = self._compute_single_quantile(hist, edges, target_count)
                q_values.append(q_value)

            results.append(np.array(q_values))
        return results

    def _compute_single_quantile(self, hist: np.ndarray, edges: np.ndarray, target_count: float) -> float:
        """Compute a single quantile value from histogram and bin edges."""
        cumsum = np.cumsum(hist)
        idx = np.searchsorted(cumsum, target_count)

        if idx == 0:
            return edges[0]
        if idx >= len(cumsum):
            return edges[-1]

        # If not edge case, interpolate within the bin
        count_before = cumsum[idx - 1]
        count_in_bin = cumsum[idx] - count_before

        # If no samples in this bin, use the bin edge
        if count_in_bin == 0:
            return edges[idx]

        # Linear interpolation within the bin
        fraction = (target_count - count_before) / count_in_bin
        return edges[idx] + fraction * (edges[idx + 1] - edges[idx])
