# Data

The worked example uses the **IMDb non-commercial datasets**: large, public,
refreshed daily, with clean business entities (titles, ratings, people) that map
naturally to a star schema.

Raw files are git-ignored (see `data/raw/.gitignore`) and reproducible by
download. The DuckDB warehouse file (`data/warehouse.duckdb`) is also ignored.

## Download

The datasets live at <https://datasets.imdbws.com/> and are documented at
<https://developer.imdb.com/non-commercial-datasets/>. They are gzipped,
tab-separated, UTF-8, and use `\N` for NULL.

Fetch the three the models use, into `data/raw/`:

```bash
cd data/raw
curl -O https://datasets.imdbws.com/title.basics.tsv.gz
curl -O https://datasets.imdbws.com/title.ratings.tsv.gz
curl -O https://datasets.imdbws.com/name.basics.tsv.gz
gunzip *.gz        # Windows: use 7-Zip, or `tar -xzf` per file
```

This leaves:

| File | Grain | Key columns |
|---|---|---|
| `title.basics.tsv`  | one row per title  | `tconst`, `titleType`, `primaryTitle`, `startYear`, `genres` |
| `title.ratings.tsv` | one row per rated title | `tconst`, `averageRating`, `numVotes` |
| `name.basics.tsv`   | one row per person | `nconst`, `primaryName`, `primaryProfession` |

`tconst` is the natural key shared by `title.basics` and `title.ratings`; the
`relationships` test checks that every rated title exists in the title table.

## Load into the warehouse

```bash
dwh seed --config config/warehouse.yaml     # writes raw.* tables into DuckDB
```

## Alternatives (same pipeline, different loader)

The dbt models and the data-quality core are source-agnostic. Two documented
alternatives with the same shape (large public set, clear entities):

- **GitHub Archive** (<https://www.gharchive.org/>) — hourly public event JSON;
  entities are repos, actors, events.
- **Stack Overflow data dump** (<https://archive.org/details/stackexchange>) —
  posts, users, votes, tags.

Only the seed/staging step changes; the marts and tests carry over.

## IMDb terms

The IMDb datasets are free for **personal and non-commercial use** only. Read
the licence at <https://developer.imdb.com/non-commercial-datasets/> before use.
