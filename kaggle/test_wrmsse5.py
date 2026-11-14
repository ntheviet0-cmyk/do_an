import unittest

import numpy as np

try:
    from wrmsse5 import LEVELS, build_contexts, evaluate_wrmsse5, rmsse_scale
except ModuleNotFoundError:
    from kaggle.wrmsse5 import LEVELS, build_contexts, evaluate_wrmsse5, rmsse_scale


class Wrmsse5Test(unittest.TestCase):
    def setUp(self):
        self.groups = {
            "L1": np.array([0, 0, 0, 0]),
            "L2": np.array([0, 0, 1, 1]),
            "L3": np.array([0, 1, 2, 3]),
            "L9": np.array([0, 1, 2, 3]),
            "L12": np.array([0, 1, 2, 3]),
        }
        self.history = np.array([
            [0, 0, 1, 2, 1, 3],
            [0, 2, 1, 2, 3, 2],
            [1, 1, 2, 2, 3, 4],
            [0, 0, 0, 1, 0, 2],
        ], dtype=float)
        self.prices = np.array([
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
            [1, 1, 1, 1, 1, 1],
            [3, 3, 3, 3, 3, 3],
        ], dtype=float)

    def test_scale_trims_leading_zeros(self):
        scale = rmsse_scale(np.array([[0, 0, 2, 4, 3]], dtype=float))
        self.assertAlmostEqual(scale[0], (4 + 1) / 2)

    def test_perfect_forecast_scores_zero_at_all_levels(self):
        contexts = build_contexts(self.groups, self.history, self.prices, weight_window=3)
        actual = np.array([
            [2, 1],
            [3, 2],
            [4, 3],
            [1, 0],
        ], dtype=float)
        rows, score = evaluate_wrmsse5(actual, actual.copy(), contexts)
        self.assertEqual([row["level"] for row in rows], list(LEVELS))
        self.assertEqual(score, 0.0)
        self.assertTrue(all(row["WRMSSE"] == 0.0 for row in rows))

    def test_weights_only_use_pre_forecast_history(self):
        contexts = build_contexts(self.groups, self.history, self.prices, weight_window=2)
        l12_weights = contexts["L12"]["weights"]
        expected_revenue = np.array([4.0, 10.0, 7.0, 6.0])
        np.testing.assert_allclose(l12_weights, expected_revenue / expected_revenue.sum())

    def test_invalid_scale_is_excluded_and_weights_are_renormalized(self):
        history = self.history.copy()
        history[3] = 0
        contexts = build_contexts(self.groups, history, self.prices, weight_window=3)
        l12 = contexts["L12"]
        self.assertFalse(l12["valid"][3])
        self.assertAlmostEqual(float(l12["weights"].sum()), 1.0)
        self.assertEqual(float(l12["weights"][3]), 0.0)


if __name__ == "__main__":
    unittest.main()
