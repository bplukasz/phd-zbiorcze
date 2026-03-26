"""Performance profiler for GAN training iterations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, cast


@dataclass
class TimingStats:
    """Statistics dla pojedynczego kroku."""
    count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    
    def add(self, elapsed: float) -> None:
        self.count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
    
    @property
    def avg_time(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0.0
    
    def __str__(self) -> str:
        if self.count == 0:
            return "no data"
        return (
            f"count={self.count:4d} | "
            f"avg={self.avg_time*1000:7.2f}ms | "
            f"min={self.min_time*1000:7.2f}ms | "
            f"max={self.max_time*1000:7.2f}ms | "
            f"total={self.total_time:8.1f}s"
        )


@dataclass
class IterationProfiler:
    """Profiler dla pętli treningowej."""
    timings: Dict[str, TimingStats] = field(default_factory=dict)
    
    def start_timer(self, name: str) -> Callable[[], None]:
        """Zwraca funkcję stop do użytku z kontekstowym menadżerem."""
        t_start = time.time()
        
        def stop():
            elapsed = time.time() - t_start
            if name not in self.timings:
                self.timings[name] = TimingStats()
            self.timings[name].add(elapsed)
        
        return stop
    
    def context(self, name: str):
        """Context manager do pomiaru czasu."""
        return _TimerContext(self, name)
    
    def get_summary(self) -> str:
        """Zwraca podsumowanie timingów."""
        if not self.timings:
            return "No timings recorded."
        
        # Posortuj po całkowitym czasowi
        sorted_items = sorted(
            self.timings.items(),
            key=lambda x: x[1].total_time,
            reverse=True
        )
        
        total_all = sum(s.total_time for s in self.timings.values())
        
        lines = []
        lines.append("\n" + "="*90)
        lines.append("PROFILER SUMMARY")
        lines.append("="*90)
        
        for name, stats in sorted_items:
            pct = (stats.total_time / total_all * 100) if total_all > 0 else 0
            lines.append(f"{name:30s} | {stats}  | {pct:5.1f}%")
        
        lines.append("-"*90)
        lines.append(f"{'TOTAL':30s} | count={len(sorted_items):4d} iterations | total={total_all:8.1f}s")
        lines.append("="*90)
        
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Resetuj wszystkie timery."""
        self.timings.clear()


class _TimerContext:
    """Context manager dla timera."""
    def __init__(self, profiler: IterationProfiler, name: str):
        self.profiler = profiler
        self.name = name
        self.t_start = None
    
    def __enter__(self):
        self.t_start = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.t_start
        if self.name not in self.profiler.timings:
            self.profiler.timings[self.name] = TimingStats()
        self.profiler.timings[self.name].add(elapsed)


# Globalny profiler dla łatwego dostępu
_global_profiler: Optional[IterationProfiler] = None


def get_global_profiler() -> IterationProfiler:
    """Zwraca globalny profiler, tworzy jeśli nie istnieje."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = IterationProfiler()
    return cast(IterationProfiler, _global_profiler)


def reset_global_profiler() -> None:
    """Resetuj globalny profiler."""
    global _global_profiler
    _global_profiler = None
