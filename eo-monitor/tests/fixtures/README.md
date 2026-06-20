# Test fixtures

Tiny, committed synthetic data so the test suite runs offline with no network,
no STAC access, and no heavy geospatial stack.

- `baseline_stack.npy` — a `(time=3, y=2, x=2)` float64 stack of a synthetic
  index, used to exercise the anomaly baseline statistics on a real file.

Larger fixtures (none currently) should stay well under a few hundred KB; the
`pre-commit` `check-added-large-files` hook enforces a 512 KB cap.
