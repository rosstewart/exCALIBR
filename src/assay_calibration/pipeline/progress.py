"""
Progress reporting for the assay calibration pipeline.

ProgressReporter writes structured JSON to a file that the web backend
can poll to show real-time progress on the results page.

Usage
-----
    reporter = ProgressReporter(progress_file="/path/to/progress.json")
    reporter.start(n_bootstraps=1000, n_variants=1579, n_samples=3)
    reporter.update_fits(done=50, total=1000)
    reporter.stage("model_selection")
    reporter.stage("visualization", component="2c", prior=0.014, flipped=True)
    reporter.stage("variant_table", variants_assigned=1579)
    reporter.complete(selected_model="2c", selection_method="conservative")

If progress_file is None all calls are no-ops — existing pipeline
behaviour is completely unchanged when run without --progress-file.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, TypeVar

T = TypeVar("T")


class ProgressReporter:
    """
    Writes structured progress JSON to a file for consumption by the web API.

    All public methods are safe to call with progress_file=None — they
    become no-ops so the pipeline works identically without the flag.
    """

    def __init__(self, progress_file: Optional[str] = None):
        self._path = Path(progress_file) if progress_file else None
        self._start_time = time.monotonic()
        self._state: dict = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def start(
        self,
        n_bootstraps: int,
        n_fits_total: int,
        n_variants: int,
        n_samples: int,
    ) -> None:
        """Called once after the dataset is loaded, before fitting begins."""
        self._start_time = time.monotonic()
        self._state = {
            "stage":           "bootstrap",
            "stage_label":     "Bootstrap fitting",
            "fits_done":       0,
            "fits_total":      n_fits_total,
            "bootstraps_total": n_bootstraps,
            "percent":         0,
            "elapsed":         None,
            "remaining":       None,
            "dataset": {
                "n_variants": n_variants,
                "n_samples":  n_samples,
            },
            "results":         {},
            "updated_at":      self._now(),
        }
        self._write()

    def update_fits(self, done: int, total: int) -> None:
        """
        Called after each individual fit completes (in parallel mode, called
        from the main process as results stream in via the generator).
        """
        elapsed = time.monotonic() - self._start_time
        remaining = (elapsed / done * (total - done)) if done > 0 else None

        self._state.update({
            "stage":       "bootstrap",
            "stage_label": "Bootstrap fitting",
            "fits_done":   done,
            "fits_total":  total,
            "percent":     round(done / total * 100) if total > 0 else 0,
            "elapsed":     self._fmt_time(elapsed),
            "remaining":   self._fmt_time(remaining) if remaining else None,
            "updated_at":  self._now(),
        })
        self._write()

    def stage(self, name: str, **kwargs) -> None:
        """
        Signal a stage transition with optional metadata.

        name        stage key (used for label lookup)
        kwargs      arbitrary key-value pairs merged into results
                    e.g. prior=0.014, flipped=True, selected_model="2c"
        """
        labels = {
            "model_selection": "Model selection",
            "visualization":   "Generating visualizations",
            "variant_table":   "Computing variant evidence table",
            "saving":          "Saving results",
        }
        elapsed = time.monotonic() - self._start_time
        self._state.update({
            "stage":       name,
            "stage_label": labels.get(name, name),
            "percent":     self._stage_percent(name),
            "elapsed":     self._fmt_time(elapsed),
            "remaining":   None,
            "updated_at":  self._now(),
        })
        self._state["results"].update(kwargs)
        self._write()

    def complete(self, **kwargs) -> None:
        """Called when the pipeline finishes successfully."""
        elapsed = time.monotonic() - self._start_time
        self._state.update({
            "stage":       "complete",
            "stage_label": "Complete",
            "percent":     100,
            "elapsed":     self._fmt_time(elapsed),
            "remaining":   None,
            "updated_at":  self._now(),
        })
        self._state["results"].update(kwargs)
        self._write()

    def track(self, iterable: Iterator[T], total: int) -> Iterator[T]:
        """
        Wrap a joblib generator so progress is updated as each result arrives.

        Usage:
            results = list(reporter.track(
                Parallel(return_as="generator")(...),
                total=len(tasks),
            ))
        """
        for i, item in enumerate(iterable, start=1):
            self.update_fits(done=i, total=total)
            yield item

    # ── Internals ─────────────────────────────────────────────────────────────

    def _write(self) -> None:
        if self._path is None:
            return
        # Atomic write: write to a temp file then rename so readers never
        # see a partial file.
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self._state, indent=2))
            os.replace(tmp, self._path)
        except Exception:
            pass  # Never let progress I/O break the pipeline

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _fmt_time(seconds: Optional[float]) -> Optional[str]:
        if seconds is None:
            return None
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        minutes, secs = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m {secs}s" if secs else f"{minutes}m"
        hours, mins = divmod(minutes, 60)
        return f"{hours}h {mins}m"

    @staticmethod
    def _stage_percent(name: str) -> int:
        """Approximate percent complete when entering each post-bootstrap stage."""
        return {
            "model_selection": 85,
            "visualization":   90,
            "variant_table":   95,
            "saving":          98,
            "complete":        100,
        }.get(name, 80)