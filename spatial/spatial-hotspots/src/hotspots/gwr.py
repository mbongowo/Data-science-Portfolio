"""Optional Geographically Weighted Regression (GWR).

GWR relaxes the stationarity assumption of OLS by fitting a separate local
regression at every location, weighting nearby observations more heavily via a
spatial kernel of a chosen *bandwidth*. It is a useful follow-up to ESDA: once
LISA/Gi* show *where* clustering is, GWR can suggest whether the *relationship*
between a response and covariates varies across space.

Caveats (state them, don't hide them):

* GWR is exploratory. Local coefficients are correlated and the effective
  number of parameters is large; naive t-tests overstate significance. Use the
  ``mgwr`` corrected critical values / adjusted alpha.
* Results are sensitive to the bandwidth and kernel. Report both.
* Local multicollinearity is common and can flip coefficient signs locally.

The ``mgwr`` import is guarded so that importing this module never breaks the
package when ``mgwr`` is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

try:  # pragma: no cover - exercised only when mgwr is installed
    from mgwr.gwr import GWR
    from mgwr.sel_bw import Sel_BW

    _HAS_MGWR = True
except Exception:  # noqa: BLE001 - any import failure means "unavailable"
    GWR = None  # type: ignore[assignment]
    Sel_BW = None  # type: ignore[assignment]
    _HAS_MGWR = False


def mgwr_available() -> bool:
    """Return True if ``mgwr`` could be imported."""
    return _HAS_MGWR


@dataclass
class GWRResult:
    """Container for a fitted GWR model."""

    bandwidth: float
    local_params: "NDArray[np.float64]"
    local_r2: "NDArray[np.float64]"
    aicc: float
    kernel: str
    fixed: bool
    model: Any  # the underlying mgwr results object, for advanced use


def _require_mgwr() -> None:
    if not _HAS_MGWR:
        raise ImportError(
            "mgwr is not installed. Install it (e.g. `pip install mgwr` or via "
            "pixi) to use GWR. The rest of the hotspots package works without "
            "it."
        )


def select_bandwidth(
    coords: "NDArray[np.float64]",
    y: "NDArray[np.float64]",
    x: "NDArray[np.float64]",
    *,
    kernel: str = "bisquare",
    fixed: bool = False,
    criterion: str = "AICc",
) -> float:
    """Select a GWR bandwidth via ``mgwr.sel_bw.Sel_BW``.

    Parameters
    ----------
    coords:
        ``(n, 2)`` array of projected (x, y) location coordinates.
    y:
        ``(n, 1)`` response array.
    x:
        ``(n, k)`` covariate array (no intercept column; mgwr adds it).
    kernel:
        ``"bisquare"``, ``"gaussian"`` or ``"exponential"``.
    fixed:
        If True, bandwidth is a fixed distance; if False (default), an adaptive
        nearest-neighbour count.
    criterion:
        Selection criterion, e.g. ``"AICc"`` or ``"CV"``.

    Returns
    -------
    float
        The selected bandwidth.
    """
    _require_mgwr()
    selector = Sel_BW(coords, y, x, kernel=kernel, fixed=fixed)
    return float(selector.search(criterion=criterion))


def fit_gwr(
    coords: "NDArray[np.float64]",
    y: "NDArray[np.float64]",
    x: "NDArray[np.float64]",
    *,
    bandwidth: float | None = None,
    kernel: str = "bisquare",
    fixed: bool = False,
    criterion: str = "AICc",
) -> GWRResult:
    """Fit a GWR model, selecting the bandwidth automatically if not given.

    Parameters
    ----------
    coords, y, x:
        See :func:`select_bandwidth`. ``y`` must be shape ``(n, 1)`` and ``x``
        shape ``(n, k)``.
    bandwidth:
        If ``None``, selected via :func:`select_bandwidth`.
    kernel, fixed, criterion:
        Passed through to selection / fitting.

    Returns
    -------
    GWRResult
        With per-location coefficients (``local_params`` of shape
        ``(n, k + 1)`` including the intercept) and local R-squared.
    """
    _require_mgwr()

    coords = np.asarray(coords, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if bandwidth is None:
        bandwidth = select_bandwidth(
            coords, y, x, kernel=kernel, fixed=fixed, criterion=criterion
        )

    model = GWR(coords, y, x, bw=bandwidth, kernel=kernel, fixed=fixed)
    results = model.fit()

    return GWRResult(
        bandwidth=float(bandwidth),
        local_params=np.asarray(results.params, dtype=float),
        local_r2=np.asarray(results.localR2, dtype=float).ravel(),
        aicc=float(results.aicc),
        kernel=kernel,
        fixed=fixed,
        model=results,
    )
