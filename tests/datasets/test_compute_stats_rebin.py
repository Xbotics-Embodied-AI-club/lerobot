"""RunningQuantileStats 重新分桶的向量化写法（Xbotics）与原来的逐桶循环逐位相同。

原写法逐个旧桶 searchsorted 再累加，在端侧计算平台上一次 50 ms 且一直占着 GIL；
这里拿原写法当参照，随机的直方图与范围扩大上对拍，结果必须完全相等（不是近似）。
"""

import copy

import numpy as np
import pytest

from lerobot.datasets.compute_stats import RunningQuantileStats


def reference_adjust(tracker: RunningQuantileStats) -> None:
    """改动前的逐桶循环，原样保留作参照。"""
    for i in range(len(tracker._histograms)):
        old_edges = tracker._bin_edges[i]
        old_hist = tracker._histograms[i]
        padding = (tracker._max[i] - tracker._min[i]) * 1e-10
        new_edges = np.linspace(
            tracker._min[i] - padding, tracker._max[i] + padding, tracker._num_quantile_bins + 1
        )
        old_centers = (old_edges[:-1] + old_edges[1:]) / 2
        new_hist = np.zeros(tracker._num_quantile_bins)
        for old_center, count in zip(old_centers, old_hist, strict=False):
            if count > 0:
                bin_idx = np.searchsorted(new_edges, old_center) - 1
                bin_idx = max(0, min(bin_idx, tracker._num_quantile_bins - 1))
                new_hist[bin_idx] += count
        tracker._histograms[i] = new_hist
        tracker._bin_edges[i] = new_edges


@pytest.mark.parametrize("seed", range(40))
def test_vectorized_rebin_matches_the_loop_exactly(seed):
    rng = np.random.default_rng(seed)
    bins = int(rng.choice([7, 64, 500, 5000]))
    channels = int(rng.integers(1, 4))
    tracker = RunningQuantileStats(num_quantile_bins=bins)
    # 先在一个窄范围里攒直方图，再让范围从一侧或两侧扩大（录制时像素范围逐渐变宽就是这样）
    lo, hi = rng.uniform(50, 100), rng.uniform(100, 150)
    for _ in range(int(rng.integers(1, 5))):
        tracker.update(rng.uniform(lo, hi, size=(int(rng.integers(2, 400)), channels)).astype(np.float32))
    tracker._min = tracker._min - rng.uniform(0, 60, channels) * rng.integers(0, 2, channels)
    tracker._max = tracker._max + rng.uniform(0, 60, channels) * rng.integers(0, 2, channels)

    expected = copy.deepcopy(tracker)
    reference_adjust(expected)
    tracker._adjust_histograms()

    for got, want in zip(tracker._histograms, expected._histograms, strict=True):
        np.testing.assert_array_equal(got, want)
        assert got.dtype == want.dtype
    for got, want in zip(tracker._bin_edges, expected._bin_edges, strict=True):
        np.testing.assert_array_equal(got, want)


def test_rebin_with_empty_histogram():
    tracker = RunningQuantileStats(num_quantile_bins=16)
    tracker.update(np.array([[1.0], [2.0]], dtype=np.float32))
    tracker._histograms = [np.zeros(16)]
    tracker._max = tracker._max + 5
    expected = copy.deepcopy(tracker)
    reference_adjust(expected)
    tracker._adjust_histograms()
    np.testing.assert_array_equal(tracker._histograms[0], expected._histograms[0])


@pytest.mark.parametrize("seed", range(20))
def test_uint8_histogram_shortcut_matches_per_pixel(seed):
    """uint8 画面走「先数取值再分桶」，同一批数据转成 float32 走原来的逐像素分桶：直方图与全部统计逐位相同。

    若捷径定错了桶（左右闭合搞反、漏了最后一桶的右端）或次数没乘对，某个桶的计数就会不等。
    取值集中在少数几个灰度、并让范围逐批扩大（触发重新分桶），覆盖正好落在桶边界上的取值。
    """
    rng = np.random.default_rng(seed)
    bins = int(rng.choice([7, 255, 5000]))
    fast = RunningQuantileStats(num_quantile_bins=bins)
    slow = RunningQuantileStats(num_quantile_bins=bins)
    for step in range(6):
        lo = max(0, 100 - 20 * step)
        hi = min(256, 120 + 25 * step)
        levels = rng.integers(lo, hi, size=int(rng.integers(1, 12)))
        batch = rng.choice(levels, size=(int(rng.integers(2, 3000)), 3)).astype(np.uint8)
        fast.update(batch)
        slow.update(batch.astype(np.float32))
    for got, want in zip(fast._histograms, slow._histograms, strict=True):
        np.testing.assert_array_equal(got, want)
    for key, value in slow.get_statistics().items():
        np.testing.assert_array_equal(fast.get_statistics()[key], value)
