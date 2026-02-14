"""System status dashboard for real-time monitoring.

Aggregates status from all subsystems into a single snapshot
suitable for logging, display, or telemetry downlink.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class SubsystemStatus:
    """Status of a single subsystem."""
    name: str
    ok: bool = True
    message: str = ""
    last_update_t: float = 0.0

    @property
    def age_s(self) -> float:
        return time.monotonic() - self.last_update_t if self.last_update_t > 0 else -1.0

    @property
    def stale(self) -> bool:
        return self.age_s > 5.0 if self.last_update_t > 0 else False


@dataclass(slots=True)
class SystemSnapshot:
    """Complete system status at a point in time."""
    timestamp: float = 0.0
    uptime_s: float = 0.0
    subsystems: dict[str, SubsystemStatus] = field(default_factory=dict)
    position_source: str = "none"
    fix_rate: float = 0.0
    fps: float = 0.0
    cpu_temp_c: float = 0.0
    memory_mb: float = 0.0

    @property
    def all_ok(self) -> bool:
        return all(s.ok for s in self.subsystems.values())

    @property
    def warnings(self) -> list[str]:
        w = []
        for s in self.subsystems.values():
            if not s.ok:
                w.append(f"{s.name}: {s.message}")
            elif s.stale:
                w.append(f"{s.name}: stale ({s.age_s:.1f}s)")
        return w

    def summary(self) -> str:
        status = "OK" if self.all_ok else "DEGRADED"
        lines = [
            f"System: {status} | up {self.uptime_s:.0f}s",
            f"  Position: {self.position_source} | fix rate: {self.fix_rate:.0%}",
            f"  FPS: {self.fps:.1f} | CPU: {self.cpu_temp_c:.0f}°C | RAM: {self.memory_mb:.0f}MB",
        ]
        for w in self.warnings:
            lines.append(f"  WARN: {w}")
        return "\n".join(lines)


class StatusDashboard:
    """Aggregates subsystem statuses into system snapshots.

    Usage:
        dash = StatusDashboard()
        dash.update("camera", ok=True, message="30fps")
        dash.update("matcher", ok=True, message="15ms avg")
        dash.update("uart", ok=False, message="disconnected")
        snap = dash.snapshot()
    """

    def __init__(self):
        self._start_t = time.monotonic()
        self._subsystems: dict[str, SubsystemStatus] = {}
        self._position_source = "none"
        self._fix_rate = 0.0
        self._fps = 0.0
        self._frame_times: list[float] = []

    def update(self, name: str, ok: bool = True, message: str = "") -> None:
        """Update a subsystem's status."""
        t = time.monotonic()
        if name in self._subsystems:
            s = self._subsystems[name]
            s.ok = ok
            s.message = message
            s.last_update_t = t
        else:
            self._subsystems[name] = SubsystemStatus(
                name=name, ok=ok, message=message, last_update_t=t,
            )

    def record_frame(self, t: float | None = None) -> None:
        """Record a frame timestamp for FPS calculation."""
        if t is None:
            t = time.monotonic()
        self._frame_times.append(t)
        if len(self._frame_times) > 100:
            self._frame_times = self._frame_times[-50:]

        if len(self._frame_times) >= 2:
            dt = self._frame_times[-1] - self._frame_times[0]
            if dt > 0:
                self._fps = (len(self._frame_times) - 1) / dt

    def set_position_info(self, source: str, fix_rate: float) -> None:
        """Update position source and fix rate."""
        self._position_source = source
        self._fix_rate = fix_rate

    def snapshot(self) -> SystemSnapshot:
        """Create a snapshot of current system status."""
        t = time.monotonic()
        cpu_temp = self._read_cpu_temp()
        mem = self._read_memory_mb()

        return SystemSnapshot(
            timestamp=t,
            uptime_s=t - self._start_t,
            subsystems=dict(self._subsystems),
            position_source=self._position_source,
            fix_rate=self._fix_rate,
            fps=self._fps,
            cpu_temp_c=cpu_temp,
            memory_mb=mem,
        )

    @staticmethod
    def _read_cpu_temp() -> float:
        """Read CPU temperature (Linux thermal zone)."""
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return int(f.read().strip()) / 1000.0
        except (OSError, ValueError):
            return 0.0

    @staticmethod
    def _read_memory_mb() -> float:
        """Read process memory usage."""
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024.0  # KB to MB
        except (ImportError, AttributeError):
            return 0.0
