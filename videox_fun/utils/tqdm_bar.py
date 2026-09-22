# Pause-aware tqdm progress bar shared by the training scripts.
"""tqdm subclass whose elapsed time and rate skip explicitly excluded windows."""
from contextlib import contextmanager

from tqdm.auto import tqdm

__all__ = ["PauseAwareTqdm"]


class PauseAwareTqdm(tqdm):
    r"""
    tqdm whose elapsed time and rate ignore the time spent inside `paused()` blocks.

    tqdm has no notion of a pause: the smoothed step rate is derived from the
    time since `last_print_t` and the closing line re-derives itself from the
    overall average since `start_t`. A non-training pause (checkpoint save,
    validation) left uncompensated lands in the interval of the next update, so
    the bar shows that step as much slower, and the pause dilutes the closing
    line. Every tqdm read goes through the instance clock `self._time` (an
    instance attribute holding `time.time` by default), so subtracting the
    total pause time there keeps the elapsed time and every rate reporting the
    training speed only.

    `close()` additionally rebases `start_t` onto the smoothed rate before the
    final display: the closing line is re-derived as the overall average, which
    one-off costs (worker warm-up, first dataloader fetch) would otherwise
    dilute even when all pauses are excluded.

    Verified against tqdm 4.27 - 4.67: `tqdm.auto`, which the training scripts
    import, first ships in 4.27, and the pause compensation works from there.
    The `close()` rebase additionally needs `format_dict` (4.30+); below that
    it is skipped while elapsed time and rates are still compensated. If a
    future release drops an internal this relies on, the class degrades to a
    plain bar instead of raising: the probe below leaves `_base_time` unset
    and `paused()` / the `close()` rebase become no-ops.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        base = getattr(self, "_time", None)  # tqdm's own clock (time.time by default)
        self._base_time = base if callable(base) else None
        self._pause_total = 0.0
        # `initial` only became an instance attribute in tqdm ~4.55; keep a private
        # copy so the close() rebase below works on older releases as well.
        initial = getattr(self, "initial", None)
        self._initial = kwargs.get("initial", 0) if initial is None else initial
        if self._base_time is not None:
            def clock():
                return self._base_time() - self._pause_total
            self._time = clock

    @contextmanager
    def paused(self):
        """Keeps the wall time of the block out of the elapsed time and the rates."""
        if self.disable or self._base_time is None:
            yield
            return
        started = self._base_time()
        try:
            yield
        finally:
            self._pause_total += self._base_time() - started

    def close(self):
        # tqdm re-derives the closing line as the overall average (elapsed from bar
        # creation); rebasing start_t onto the smoothed rate the bar was just showing
        # makes that line repeat the pace the run actually sustained.
        if not self.disable and self._base_time is not None:
            rate = getattr(self, "format_dict", {}).get("rate")
            if rate:
                self.start_t = self._time() - (self.n - self._initial) / rate
        super().close()
