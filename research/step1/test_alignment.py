"""用独立穷举CTC路径检验动态规划，检查缺失掩码和交叠权重。"""
import itertools
import unittest
import numpy as np
from alignment import ctc_viterbi, overlap_pool


class AlignmentTests(unittest.TestCase):
    def test_repeated_labels_against_enumeration(self):
        rng = np.random.default_rng(19)
        logits = rng.normal(size=(5, 3))
        logp = logits - np.log(np.exp(logits).sum(axis=1))[:, None]
        for target in ([1, 1], [1, 2], [2]):
            best = -np.inf
            for seq in itertools.product(range(3), repeat=5):
                collapsed = [v for j, v in enumerate(seq) if (j == 0 or seq[j-1] != v) and v != 0]
                if collapsed == target:
                    best = max(best, sum(logp[t, v] for t, v in enumerate(seq)))
            _, actual = ctc_viterbi(logp, np.array(target))
            self.assertAlmostEqual(best, actual, places=10)

    def test_overlap_and_missing(self):
        intervals = np.array([[.5, 1.5], [1., 2.], [3., 4.]])
        frames = np.array([[0., 1.], [1., 2.]])
        feat = np.array([[2., 4.], [10., 20.]])
        out, valid, weights = overlap_pool(intervals, frames, feat, [True, False])
        np.testing.assert_allclose(out, [[2, 4], [0, 0], [0, 0]])
        np.testing.assert_array_equal(valid, [True, False, False])
        np.testing.assert_allclose(weights, [[1, 0], [0, 0], [0, 0]])

    def test_impossible_ctc(self):
        with self.assertRaises(ValueError):
            ctc_viterbi(np.log(np.full((2, 3), 1/3)), np.array([1, 1]))


if __name__ == "__main__":
    unittest.main()
