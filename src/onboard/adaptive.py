"""Adaptive matching parameters based on flight conditions.

Adjusts feature extraction and matching thresholds dynamically
based on altitude estimate, speed, blur level, and recent match success.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MatchParams:
    """Current matching parameters (adjusted dynamically)."""
    min_matches: int = 15
    ransac_threshold: float = 5.0
    min_inlier_ratio: float = 0.3
    max_features: int = 500
    match_ratio_threshold: float = 0.75
    blur_reject_threshold: float = 50.0

    def summary(self) -> str:
        return (
            f"min_matches={self.min_matches}, ransac={self.ransac_threshold:.1f}, "
            f"inlier_ratio={self.min_inlier_ratio:.2f}, max_feat={self.max_features}, "
            f"match_ratio={self.match_ratio_threshold:.2f}"
        )


class AdaptiveController:
    """Adjusts matching parameters based on conditions.

    Tracks recent match success rate and adjusts thresholds to
    balance fix rate vs accuracy. When fixes are scarce, it relaxes
    thresholds; when fixes are reliable, it tightens them.

    Usage:
        ctrl = AdaptiveController()
        params = ctrl.params
        # ... use params for matching ...
        ctrl.record_result(success=True, inlier_ratio=0.6, blur=120.0)
        params = ctrl.params  # may have adjusted
    """

    def __init__(
        self,
        window_size: int = 20,
        target_fix_rate: float = 0.5,
        min_min_matches: int = 8,
        max_min_matches: int = 25,
    ):
        self._window_size = window_size
        self._target_fix_rate = target_fix_rate
        self._min_min_matches = min_min_matches
        self._max_min_matches = max_min_matches

        self._params = MatchParams()
        self._results: list[bool] = []
        self._inlier_ratios: list[float] = []
        self._blur_levels: list[float] = []

    @property
    def params(self) -> MatchParams:
        return self._params

    @property
    def recent_fix_rate(self) -> float:
        if not self._results:
            return 0.0
        window = self._results[-self._window_size:]
        return sum(window) / len(window)

    @property
    def recent_mean_inlier_ratio(self) -> float:
        if not self._inlier_ratios:
            return 0.0
        window = self._inlier_ratios[-self._window_size:]
        return sum(window) / len(window)

    def record_result(
        self,
        success: bool,
        inlier_ratio: float = 0.0,
        blur: float = 0.0,
        speed_mps: float = 0.0,
    ) -> None:
        """Record a matching attempt result and adjust parameters."""
        self._results.append(success)
        if success:
            self._inlier_ratios.append(inlier_ratio)
        if blur > 0:
            self._blur_levels.append(blur)

        # Trim history
        if len(self._results) > self._window_size * 2:
            self._results = self._results[-self._window_size:]
        if len(self._inlier_ratios) > self._window_size * 2:
            self._inlier_ratios = self._inlier_ratios[-self._window_size:]
        if len(self._blur_levels) > self._window_size * 2:
            self._blur_levels = self._blur_levels[-self._window_size:]

        self._adjust()

    def _adjust(self) -> None:
        """Adjust parameters based on recent performance."""
        fix_rate = self.recent_fix_rate
        p = self._params

        if fix_rate < self._target_fix_rate * 0.5:
            # Very low fix rate — relax everything
            p.min_matches = max(self._min_min_matches, p.min_matches - 1)
            p.min_inlier_ratio = max(0.15, p.min_inlier_ratio - 0.02)
            p.match_ratio_threshold = min(0.85, p.match_ratio_threshold + 0.02)
            p.max_features = min(1000, p.max_features + 50)
        elif fix_rate < self._target_fix_rate:
            # Below target — relax slightly
            p.min_matches = max(self._min_min_matches, p.min_matches - 1)
            p.min_inlier_ratio = max(0.20, p.min_inlier_ratio - 0.01)
        elif fix_rate > self._target_fix_rate * 1.5:
            # Well above target — tighten for accuracy
            p.min_matches = min(self._max_min_matches, p.min_matches + 1)
            p.min_inlier_ratio = min(0.50, p.min_inlier_ratio + 0.01)
            p.match_ratio_threshold = max(0.65, p.match_ratio_threshold - 0.01)
            p.max_features = max(300, p.max_features - 25)

        # Blur-based adjustment
        if self._blur_levels:
            recent_blur = self._blur_levels[-1]
            if recent_blur < p.blur_reject_threshold:
                # Image is blurry — skip matching
                logger.debug("Blur %.1f below threshold %.1f", recent_blur, p.blur_reject_threshold)

    def reset(self) -> None:
        """Reset to defaults."""
        self._params = MatchParams()
        self._results.clear()
        self._inlier_ratios.clear()
        self._blur_levels.clear()

    def should_skip_frame(self, blur: float) -> bool:
        """Return True if the frame is too blurry to attempt matching."""
        return blur < self._params.blur_reject_threshold
