"""A one-command, dependency-light demo of the data-quality core end to end.

This drives the **real** pure-pandas core (:mod:`dwh.dq` and
:mod:`dwh.dimensional`) over a small, deterministic, synthetic IMDb-like
dataset. It needs only numpy / pandas / pyyaml + stdlib — no dbt, no duckdb, no
warehouse — so it reproduces real, committed metrics anywhere, including CI.

What it does:

1. Deterministically synthesizes two small source tables with referential
   integrity (``numpy.random.default_rng(seed)``):

   * ``titles``  — ``tconst``, ``primaryTitle``, ``titleType``, ``startYear``
   * ``ratings`` — ``tconst`` (FK to titles), ``averageRating``, ``numVotes``

2. Builds warehouse marts the way the dbt project does, using the real
   :mod:`dwh.dimensional` helpers: a surrogate ``title_key`` per title and a
   year/date dimension spanning the titles' release years.

3. Runs the full generic-test suite via :func:`dwh.dq.run_suite` — the same four
   semantics dbt ships (``not_null`` / ``unique`` / ``accepted_values`` /
   ``relationships``) — over the synthetic marts.

4. Writes artifacts (``dq_results.csv``, ``dim_sample.csv``, ``summary.json``)
   and returns a small dict of real counts.

A second *dirty* dataset (a planted null, duplicate, bad ``titleType`` and an
orphan rating) is run for contrast so the demo also shows the suite catching
defects, but the headline metrics come from the CLEAN data, which passes every
test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dwh.dimensional import build_date_dim, surrogate_key
from dwh.dq import TestSpec, run_suite

# Allowed title types, mirroring the dbt accepted_values test on titleType.
_TITLE_TYPES = ["movie", "short", "tvSeries", "tvEpisode", "documentary"]


def synthesize_titles(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Synthesize ``n`` clean, distinct IMDb-like title rows."""
    tconst = [f"tt{i:07d}" for i in range(1, n + 1)]
    title_type = rng.choice(_TITLE_TYPES, size=n)
    start_year = rng.integers(1990, 2021, size=n)
    primary_title = [f"Title {i:04d}" for i in range(1, n + 1)]
    return pd.DataFrame(
        {
            "tconst": tconst,
            "primaryTitle": primary_title,
            "titleType": title_type,
            "startYear": start_year.astype(int),
        }
    )


def synthesize_ratings(
    rng: np.random.Generator, titles: pd.DataFrame, frac: float = 0.8
) -> pd.DataFrame:
    """Synthesize one rating per rated title, FK-clean against ``titles``.

    A deterministic subset (``frac``) of titles receive a rating, so the child
    table is a strict, referentially-intact subset of the parent keys.
    """
    n = len(titles)
    k = max(1, int(round(n * frac)))
    idx = np.sort(rng.choice(n, size=k, replace=False))
    rated = titles.iloc[idx]
    average_rating = np.round(rng.uniform(1.0, 10.0, size=k), 1)
    num_votes = rng.integers(5, 100_000, size=k)
    return pd.DataFrame(
        {
            "tconst": rated["tconst"].to_numpy(),
            "averageRating": average_rating,
            "numVotes": num_votes.astype(int),
        }
    ).reset_index(drop=True)


def _rating_band(average_rating: pd.Series) -> pd.Series:
    """Derive the BI ``rating_band`` (high/medium/low), mirroring the mart."""
    bins = [-np.inf, 5.0, 7.5, np.inf]
    return pd.cut(
        average_rating, bins=bins, labels=["low", "medium", "high"]
    ).astype("object")


def build_marts(titles: pd.DataFrame, ratings: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build dim_title / fct_title_rating / dim_date from source tables.

    Uses the real :mod:`dwh.dimensional` helpers, the same building blocks the
    dbt marts express in SQL.
    """
    dim_title = titles.rename(columns={"tconst": "title_id", "titleType": "title_type"})
    dim_title = dim_title.assign(title_key=surrogate_key(dim_title, ["title_id"]))

    # fct_title_rating: ratings joined to the dimension's surrogate key.
    # Deduplicate the lookup so a (defect) duplicate title_id does not break the
    # join; the duplicate itself is caught by the unique test on dim_title.
    key_map = (
        dim_title.drop_duplicates(subset="title_id", keep="first")
        .set_index("title_id")["title_key"]
    )
    fct = ratings.rename(columns={"tconst": "title_id"}).copy()
    fct["title_key"] = fct["title_id"].map(key_map)
    fct["rating_band"] = _rating_band(fct["averageRating"])

    # dim_date: a date dimension spanning the release years (Jan 1 of each).
    years = titles["startYear"]
    dim_date = build_date_dim(f"{int(years.min())}-01-01", f"{int(years.max())}-12-31")

    return {"dim_title": dim_title, "fct_title_rating": fct, "dim_date": dim_date}


def _marts_suite(marts: dict[str, pd.DataFrame]) -> list[TestSpec]:
    """The generic-test suite over the marts, mirroring the dbt _marts.yml block."""
    dim_title = marts["dim_title"]
    fct = marts["fct_title_rating"]
    return [
        # dim_title: surrogate key + natural key + type
        TestSpec("not_null", dim_title, "title_key", table="dim_title"),
        TestSpec("unique", dim_title, "title_key", table="dim_title"),
        TestSpec("not_null", dim_title, "title_id", table="dim_title"),
        TestSpec("unique", dim_title, "title_id", table="dim_title"),
        TestSpec("not_null", dim_title, "title_type", table="dim_title"),
        TestSpec(
            "accepted_values",
            dim_title,
            "title_type",
            table="dim_title",
            values=_TITLE_TYPES,
        ),
        # fct_title_rating: keys, FK, rating, derived band
        TestSpec("not_null", fct, "title_key", table="fct_title_rating"),
        TestSpec(
            "relationships",
            fct,
            "title_key",
            table="fct_title_rating",
            parent=dim_title,
            parent_col="title_key",
        ),
        TestSpec("not_null", fct, "title_id", table="fct_title_rating"),
        TestSpec("unique", fct, "title_id", table="fct_title_rating"),
        TestSpec("not_null", fct, "averageRating", table="fct_title_rating"),
        TestSpec("not_null", fct, "rating_band", table="fct_title_rating"),
        TestSpec(
            "accepted_values",
            fct,
            "rating_band",
            table="fct_title_rating",
            values=["high", "medium", "low"],
        ),
    ]


def _make_dirty(
    rng: np.random.Generator, titles: pd.DataFrame, ratings: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plant one of each defect for contrast (null, duplicate, bad type, orphan)."""
    dirty_titles = titles.copy()
    # Planted null title type on the first row.
    dirty_titles.loc[0, "titleType"] = None
    # Planted out-of-set title type.
    dirty_titles.loc[1, "titleType"] = "hologram"
    # Planted duplicate tconst (whole row copied with a new title).
    dup = dirty_titles.iloc[2].copy()
    dup["primaryTitle"] = "Duplicate"
    dirty_titles = pd.concat(
        [dirty_titles, dup.to_frame().T], ignore_index=True
    )

    dirty_ratings = ratings.copy()
    # Planted orphan rating (tconst absent from titles).
    orphan = pd.DataFrame(
        {"tconst": ["tt9999999"], "averageRating": [9.9], "numVotes": [42]}
    )
    dirty_ratings = pd.concat([dirty_ratings, orphan], ignore_index=True)
    return dirty_titles, dirty_ratings


def run_demo(seed: int = 0, out_dir: str = "outputs") -> dict:
    """Run the data-quality core end to end on synthetic IMDb-like data.

    Parameters
    ----------
    seed:
        Seed for ``numpy.random.default_rng``; fixes every random draw, so the
        returned counts and written artifacts are fully reproducible.
    out_dir:
        Directory for the artifacts (created if missing).

    Returns
    -------
    dict
        ``num_titles``, ``num_ratings``, ``num_tests``, ``num_passed``,
        ``num_failed``, the per-test ``breakdown`` (kind -> passed/total), and
        the contrast ``dirty_num_failed`` (failing tests on the dirty data).
    """
    rng = np.random.default_rng(seed)

    n_titles = 200
    titles = synthesize_titles(rng, n_titles)
    ratings = synthesize_ratings(rng, titles)

    marts = build_marts(titles, ratings)
    suite = _marts_suite(marts)
    results = run_suite(suite)

    # Contrast run: plant defects, confirm the suite catches them.
    dirty_titles, dirty_ratings = _make_dirty(rng, titles, ratings)
    dirty_marts = build_marts(dirty_titles, dirty_ratings)
    dirty_results = run_suite(_marts_suite(dirty_marts))

    num_tests = int(len(results))
    num_passed = int(results["passed"].sum())
    num_failed = num_tests - num_passed

    # Per-generic-test breakdown (passed / total per kind).
    breakdown: dict[str, dict[str, int]] = {}
    for kind, group in results.groupby("test"):
        breakdown[str(kind)] = {
            "passed": int(group["passed"].sum()),
            "total": int(len(group)),
        }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "dq_results.csv", index=False)
    marts["dim_title"].head(10).to_csv(out / "dim_sample.csv", index=False)

    summary = {
        "seed": seed,
        "num_titles": int(len(titles)),
        "num_ratings": int(len(ratings)),
        "num_tests": num_tests,
        "num_passed": num_passed,
        "num_failed": num_failed,
        "breakdown": breakdown,
        "dirty_num_tests": int(len(dirty_results)),
        "dirty_num_failed": int((~dirty_results["passed"]).sum()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run_demo(0), indent=2))
