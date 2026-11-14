"""Unit tests for the Gemini Update Hurdle-TFT architecture and features."""

import unittest
from pathlib import Path

import numpy as np

from hurdle_features import (
    HIER_CHANNELS,
    LOCAL_STATE_COLS,
    build_hierarchy_context,
    build_local_state_features,
    compute_first_active_day,
    fit_feature_scaler,
)
from wrmsse5 import LEVELS, build_contexts, evaluate_wrmsse5


class GeminiUpdateTest(unittest.TestCase):
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
            "L12": np.array([0, 1, 2, 3]),
        }

    def test_twelve_local_state_features_and_causality(self):
        features = build_local_state_features(self.sales, self.available).astype(np.float32)
        self.assertEqual(features.shape, (4, 8, 12))
        col = {name: i for i, name in enumerate(LOCAL_STATE_COLS)}
        self.assertEqual(features[0, 1, col["active_regime"]], 0.0)
        self.assertEqual(features[0, 2, col["active_regime"]], 1.0)
        self.assertGreater(features[0, 4, col["croston_idi_ratio"]], 0.0)
        self.assertGreater(features[0, 5, col["log_tsb_expected_28"]], 0.0)

        # Perturb future days >= 6 and verify days < 6 are strictly unchanged
        mutated_sales = self.sales.copy()
        mutated_sales[:, 6:] += 50.0
        mutated = build_local_state_features(mutated_sales, self.available).astype(np.float32)
        np.testing.assert_array_equal(features[:, :6], mutated[:, :6])

    def test_first_active_day_is_causal_at_origin(self):
        first_day = compute_first_active_day(self.sales, self.available)
        np.testing.assert_array_equal(first_day, np.array([2, 0, 1, 3], dtype=np.int32))

    def test_future_cols_are_strictly_causal(self):
        source = Path(__file__).with_name("tft_m5_gemini_update_src.py").read_text(encoding="utf-8")
        future_block = source.split("FUT_COLS = [", 1)[1].split("]", 1)[0]
        for forbidden in ("sell_price", "price_rel", "promo", "price_missing", "lag_1", "lag_7", "lag_14"):
            self.assertNotIn(f'"{forbidden}"', future_block)
        for expected_lag in ("lag_28", "lag_35", "lag_56", "lag_364"):
            self.assertIn(f'"{expected_lag}"', future_block)


if __name__ == "__main__":
    unittest.main()

