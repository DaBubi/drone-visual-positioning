"""Position confidence estimator.

Combines multiple quality signals (inlier ratio, match count, HDOP,
EKF innovation, blur level) into a single confidence score [0..1].
Used to decide whether to output a position fix or suppress it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class QualitySignals:
    """Raw quality signals from the matching pipeline."""
    inlier_ratio: float = 0.0     # fraction of RANSAC inliers
    match_count: int = 0          # number of feature matches
    hdop: float = 99.0            # horizontal dilution of precision
    ekf_innovation: float = 0.0   # EKF Mahalanobis distance
    blur_score: float = 100.0     # Laplacian variance (higher = sharper)
    speed_mps: float = 0.0        # ground speed
    altitude_consistency: float = 1.0  # 1.0 = consistent, 0.0 = inconsistent


@dataclass(slots=True)
class ConfidenceResult:
    """Confidence assessment result."""
    score: float          # 0..1 overall confidence
    reliable: bool        # True if score > threshold
    components: dict      # individual component scores
    reason: str           # why unreliable (if not reliable)


class ConfidenceEstimator:
    """Estimates position confidence from quality signals.

    Uses weighted combination of normalized quality components.

    Usage:
        est = ConfidenceEstimator()
        signals = QualitySignals(inlier_ratio=0.6, match_count=45, hdop=1.2)
        result = est.evaluate(signals)
        if result.reliable:
            output_position()
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_matches: int = 10,
        min_inlier_ratio: float = 0.2,
        max_hdop: float = 5.0,
        max_ekf_gate: float = 10.0,
        min_blur: float = 50.0,
    ):
        self._threshold = threshold
        self._min_matches = min_matches
        self._min_inlier_ratio = min_inlier_ratio
        self._max_hdop = max_hdop
        self._max_ekf_gate = max_ekf_gate
        self._min_blur = min_blur

    @property
    def threshold(self) -> float:
        return self._threshold

    def evaluate(self, signals: QualitySignals) -> ConfidenceResult:
        """Evaluate confidence from quality signals."""
        components = {}
        reasons = []

        # Inlier ratio: 0.3→0.5, 0.6→1.0
        inlier_score = _sigmoid(signals.inlier_ratio, center=0.35, steepness=8.0)
        components["inlier_ratio"] = inlier_score
        if signals.inlier_ratio < self._min_inlier_ratio:
            reasons.append(f"low inlier ratio ({signals.inlier_ratio:.2f})")

        # Match count: 10→0.3, 30→0.8, 50→1.0
        match_score = _sigmoid(signals.match_count, center=20, steepness=0.15)
        components["match_count"] = match_score
        if signals.match_count < self._min_matches:
            reasons.append(f"few matches ({signals.match_count})")

        # HDOP: 1→1.0, 3→0.5, 5→0.1
        hdop_score = 1.0 - _sigmoid(signals.hdop, center=3.0, steepness=1.5)
        components["hdop"] = hdop_score
        if signals.hdop > self._max_hdop:
            reasons.append(f"high HDOP ({signals.hdop:.1f})")

        # EKF innovation: 0→1.0, 5→0.5, 10→0.1
        if signals.ekf_innovation > 0:
            ekf_score = 1.0 - _sigmoid(signals.ekf_innovation, center=5.0, steepness=0.5)
        else:
            ekf_score = 1.0  # no EKF data — don't penalize
        components["ekf_innovation"] = ekf_score
        if signals.ekf_innovation > self._max_ekf_gate:
            reasons.append(f"high EKF innovation ({signals.ekf_innovation:.1f})")

        # Blur: sharp images score higher
        blur_score = _sigmoid(signals.blur_score, center=self._min_blur, steepness=0.05)
        components["blur"] = blur_score
        if signals.blur_score < self._min_blur:
            reasons.append(f"blurry image ({signals.blur_score:.0f})")

        # Altitude consistency
        components["altitude"] = signals.altitude_consistency

        # Weighted combination
        weights = {
            "inlier_ratio": 0.30,
            "match_count": 0.20,
            "hdop": 0.15,
            "ekf_innovation": 0.15,
            "blur": 0.10,
            "altitude": 0.10,
        }

        score = sum(components[k] * weights[k] for k in weights)
        score = max(0.0, min(1.0, score))

        reliable = score >= self._threshold and not reasons
        reason = "; ".join(reasons) if reasons else "OK"

        return ConfidenceResult(
            score=score,
            reliable=reliable,
            components=components,
            reason=reason,
        )


def _sigmoid(x: float, center: float = 0.0, steepness: float = 1.0) -> float:
    """Logistic sigmoid function, output in [0, 1]."""
    z = steepness * (x - center)
    z = max(-500, min(500, z))  # prevent overflow
    return 1.0 / (1.0 + math.exp(-z))
