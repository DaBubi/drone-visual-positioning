"""Tests for adaptive matching controller."""

import pytest

from onboard.adaptive import AdaptiveController, MatchParams


class TestMatchParams:
    def test_defaults(self):
        p = MatchParams()
        assert p.min_matches == 15
        assert p.ransac_threshold == 5.0
        assert p.min_inlier_ratio == 0.3
        assert p.max_features == 500

    def test_summary(self):
        p = MatchParams()
        s = p.summary()
        assert "min_matches=" in s
        assert "ransac=" in s


class TestAdaptiveController:
    def test_initial_params(self):
        ctrl = AdaptiveController()
        assert ctrl.params.min_matches == 15
        assert ctrl.recent_fix_rate == 0.0

    def test_record_success(self):
        ctrl = AdaptiveController()
        ctrl.record_result(success=True, inlier_ratio=0.5)
        assert ctrl.recent_fix_rate == 1.0

    def test_record_failure(self):
        ctrl = AdaptiveController()
        ctrl.record_result(success=False)
        assert ctrl.recent_fix_rate == 0.0

    def test_fix_rate_window(self):
        ctrl = AdaptiveController(window_size=10)
        for _ in range(7):
            ctrl.record_result(success=True, inlier_ratio=0.5)
        for _ in range(3):
            ctrl.record_result(success=False)
        assert ctrl.recent_fix_rate == pytest.approx(0.7)

    def test_relaxes_on_low_fix_rate(self):
        ctrl = AdaptiveController(window_size=10, target_fix_rate=0.5)
        initial_min_matches = ctrl.params.min_matches

        # Feed all failures
        for _ in range(10):
            ctrl.record_result(success=False)

        # Should have relaxed thresholds
        assert ctrl.params.min_matches < initial_min_matches

    def test_tightens_on_high_fix_rate(self):
        ctrl = AdaptiveController(window_size=10, target_fix_rate=0.5)
        initial_min_matches = ctrl.params.min_matches

        # Feed all successes
        for _ in range(20):
            ctrl.record_result(success=True, inlier_ratio=0.6)

        # Should have tightened
        assert ctrl.params.min_matches > initial_min_matches

    def test_min_matches_bounded(self):
        ctrl = AdaptiveController(window_size=5, min_min_matches=8, max_min_matches=25)

        # Push min_matches down
        for _ in range(50):
            ctrl.record_result(success=False)
        assert ctrl.params.min_matches >= 8

        # Push min_matches up
        ctrl.reset()
        for _ in range(50):
            ctrl.record_result(success=True, inlier_ratio=0.7)
        assert ctrl.params.min_matches <= 25

    def test_inlier_ratio_relaxes(self):
        ctrl = AdaptiveController(window_size=5, target_fix_rate=0.5)
        initial_ratio = ctrl.params.min_inlier_ratio

        for _ in range(20):
            ctrl.record_result(success=False)

        assert ctrl.params.min_inlier_ratio < initial_ratio

    def test_inlier_ratio_bounded(self):
        ctrl = AdaptiveController(window_size=5)
        for _ in range(100):
            ctrl.record_result(success=False)
        assert ctrl.params.min_inlier_ratio >= 0.15

    def test_reset(self):
        ctrl = AdaptiveController()
        for _ in range(10):
            ctrl.record_result(success=False)
        ctrl.reset()
        assert ctrl.params.min_matches == 15
        assert ctrl.recent_fix_rate == 0.0

    def test_should_skip_blurry_frame(self):
        ctrl = AdaptiveController()
        # Default blur threshold is 50
        assert ctrl.should_skip_frame(blur=30.0)   # too blurry
        assert not ctrl.should_skip_frame(blur=100.0)  # sharp enough

    def test_mean_inlier_ratio(self):
        ctrl = AdaptiveController()
        ctrl.record_result(success=True, inlier_ratio=0.4)
        ctrl.record_result(success=True, inlier_ratio=0.6)
        assert ctrl.recent_mean_inlier_ratio == pytest.approx(0.5)

    def test_history_trimming(self):
        ctrl = AdaptiveController(window_size=5)
        for i in range(100):
            ctrl.record_result(success=i % 2 == 0, inlier_ratio=0.5)
        # Internal lists should be trimmed
        assert len(ctrl._results) <= 10
