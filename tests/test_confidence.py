"""Tests for confidence estimator."""

import pytest

from onboard.confidence import (
    ConfidenceEstimator, QualitySignals, ConfidenceResult, _sigmoid,
)


class TestSigmoid:
    def test_center_is_half(self):
        assert _sigmoid(0.0, center=0.0) == pytest.approx(0.5)

    def test_high_value(self):
        assert _sigmoid(10.0, center=0.0) > 0.99

    def test_low_value(self):
        assert _sigmoid(-10.0, center=0.0) < 0.01

    def test_steepness(self):
        steep = _sigmoid(1.0, center=0.0, steepness=10.0)
        gentle = _sigmoid(1.0, center=0.0, steepness=1.0)
        assert steep > gentle

    def test_no_overflow(self):
        # Should not crash with extreme values
        assert _sigmoid(1e6) == pytest.approx(1.0)
        assert _sigmoid(-1e6) == pytest.approx(0.0)


class TestConfidenceEstimator:
    def test_high_quality_signals(self):
        est = ConfidenceEstimator()
        signals = QualitySignals(
            inlier_ratio=0.7, match_count=50, hdop=1.0,
            ekf_innovation=1.0, blur_score=200.0,
        )
        result = est.evaluate(signals)
        assert result.score > 0.8
        assert result.reliable

    def test_low_quality_signals(self):
        est = ConfidenceEstimator()
        signals = QualitySignals(
            inlier_ratio=0.1, match_count=5, hdop=10.0,
            ekf_innovation=20.0, blur_score=10.0,
        )
        result = est.evaluate(signals)
        assert result.score < 0.3
        assert not result.reliable

    def test_default_signals(self):
        est = ConfidenceEstimator()
        signals = QualitySignals()  # all defaults
        result = est.evaluate(signals)
        assert not result.reliable

    def test_components_present(self):
        est = ConfidenceEstimator()
        signals = QualitySignals(inlier_ratio=0.5, match_count=30)
        result = est.evaluate(signals)
        assert "inlier_ratio" in result.components
        assert "match_count" in result.components
        assert "hdop" in result.components
        assert "blur" in result.components

    def test_reason_when_unreliable(self):
        est = ConfidenceEstimator()
        signals = QualitySignals(inlier_ratio=0.1, match_count=3)
        result = est.evaluate(signals)
        assert "inlier" in result.reason or "matches" in result.reason

    def test_reason_ok_when_reliable(self):
        est = ConfidenceEstimator(threshold=0.3)
        signals = QualitySignals(
            inlier_ratio=0.7, match_count=50, hdop=1.0,
            blur_score=200.0,
        )
        result = est.evaluate(signals)
        if result.reliable:
            assert result.reason == "OK"

    def test_score_bounded(self):
        est = ConfidenceEstimator()
        for ir in [0.0, 0.5, 1.0]:
            for mc in [0, 20, 100]:
                signals = QualitySignals(inlier_ratio=ir, match_count=mc)
                result = est.evaluate(signals)
                assert 0.0 <= result.score <= 1.0

    def test_custom_threshold(self):
        est = ConfidenceEstimator(threshold=0.9)
        signals = QualitySignals(
            inlier_ratio=0.5, match_count=30, hdop=2.0,
            blur_score=100.0,
        )
        result = est.evaluate(signals)
        # Moderate signals should not meet high threshold
        assert result.score < 0.9

    def test_inlier_ratio_dominates(self):
        est = ConfidenceEstimator()
        good = QualitySignals(inlier_ratio=0.8, match_count=50, hdop=1.0, blur_score=200.0)
        bad = QualitySignals(inlier_ratio=0.05, match_count=50, hdop=1.0, blur_score=200.0)
        r_good = est.evaluate(good)
        r_bad = est.evaluate(bad)
        assert r_good.score > r_bad.score

    def test_hdop_effect(self):
        est = ConfidenceEstimator()
        low_hdop = QualitySignals(inlier_ratio=0.5, match_count=30, hdop=1.0, blur_score=200.0)
        high_hdop = QualitySignals(inlier_ratio=0.5, match_count=30, hdop=8.0, blur_score=200.0)
        assert est.evaluate(low_hdop).score > est.evaluate(high_hdop).score
