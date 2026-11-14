import unittest
from pathlib import Path

import numpy as np

try:
    from hurdle_features import (
        HIER_CHANNELS,
        LOCAL_STATE_COLS,
        build_hierarchy_context,
        build_local_state_features,
        fit_feature_scaler,
        hierarchy_feature_names,
    )
except ModuleNotFoundError:
    from kaggle.hurdle_features import (
        HIER_CHANNELS,
        LOCAL_STATE_COLS,
        build_hierarchy_context,
        build_local_state_features,
        fit_feature_scaler,
        hierarchy_feature_names,
    )


class HurdleFeatureTest(unittest.TestCase):
    def setUp(self):
        self.sales = np.array(
            [
                [0, 0, 2, 0, 0, 3, 0, 0],
                [1, 0, 0, 0, 2, 0, 0, 1],
                [0, 4, 0, 0, 0, 0, 2, 0],
                [0, 0, 0, 5, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        self.available = np.array(
            [
                [0, 0, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 1],
            ],
            dtype=bool,
        )
        self.groups = {
            "L1": np.array([0, 0, 0, 0]),
            "L2": np.array([0, 0, 1, 1]),
            "L3": np.array([0, 1, 2, 3]),
            "L9": np.array([0, 1, 2, 3]),
        }

    def test_local_features_capture_occurrence_and_lifecycle(self):
        features = build_local_state_features(self.sales, self.available).astype(np.float32)
        col = {name: i for i, name in enumerate(LOCAL_STATE_COLS)}

        self.assertEqual(features.shape, (4, 8, len(LOCAL_STATE_COLS)))
        self.assertAlmostEqual(features[0, 2, col["nonzero_rate_7"]], 1.0 / 3.0, places=3)
        self.assertEqual(features[0, 1, col["has_ever_sold"]], 0.0)
        self.assertEqual(features[0, 2, col["has_ever_sold"]], 1.0)
        self.assertAlmostEqual(features[0, 2, col["log_zero_run"]], 0.0, places=3)
        self.assertAlmostEqual(
            features[0, 4, col["log_zero_run"]], np.log1p(2), places=3
        )
        self.assertAlmostEqual(
            features[0, 4, col["log_mean_positive_28"]], np.log1p(2), places=3
        )
        self.assertAlmostEqual(features[0, 1, col["listed_rate_28"]], 0.0, places=3)

    def test_future_target_mutation_does_not_change_past_features(self):
        original = build_local_state_features(self.sales, self.available)
        changed_sales = self.sales.copy()
        changed_sales[:, 6:] += 20
        changed = build_local_state_features(changed_sales, self.available)
        np.testing.assert_array_equal(original[:, :6], changed[:, :6])

    def test_hierarchy_context_uses_correct_groups(self):
        context = build_hierarchy_context(self.sales, self.available, self.groups)
        self.assertEqual(context["L1"].shape, (1, 8, len(HIER_CHANNELS)))
        self.assertEqual(context["L2"].shape, (2, 8, len(HIER_CHANNELS)))
        expected_l2_day0 = self.sales[:2, 0].sum()
        self.assertAlmostEqual(np.expm1(context["L2"][0, 0, 0]), expected_l2_day0)
        self.assertAlmostEqual(context["L2"][0, 0, 4], 0.5)

    def test_scaler_ignores_days_after_cutoff(self):
        values = build_local_state_features(self.sales, self.available)
        mean, std = fit_feature_scaler(values, train_end=5)
        changed = values.copy()
        changed[:, 5:] = 100
        changed_mean, changed_std = fit_feature_scaler(changed, train_end=5)
        np.testing.assert_allclose(mean, changed_mean)
        np.testing.assert_allclose(std, changed_std)

    def test_hierarchy_feature_names_are_stable(self):
        names = hierarchy_feature_names()
        self.assertEqual(names[0], "L1_log_sales")
        self.assertEqual(names[-1], "L9_listed_ratio")
        self.assertEqual(len(names), 4 * len(HIER_CHANNELS))

    def test_source_does_not_use_future_price_or_target_mask(self):
        source_path = Path(__file__).with_name("tft_m5_zero_test_src.py")
        source = source_path.read_text(encoding="utf-8")
        future_block = source.split("FUT_COLS = [", 1)[1].split("]", 1)[0]
        for forbidden in ("sell_price", "price_rel", "promo", "price_missing"):
            self.assertNotIn(forbidden, future_block)
        self.assertNotIn("| (SALES > 0)", source)
        self.assertNotIn(".bfill(axis=1)", source)


if __name__ == "__main__":
    unittest.main()
