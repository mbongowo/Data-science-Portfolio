# Data

The recommender trains on explicit-feedback ratings. Raw data is git-ignored and
not committed; fetch it with the steps below.

## Primary: MovieLens-25M

[MovieLens-25M](https://grouplens.org/datasets/movielens/25m/) — 25 million
ratings (0.5–5.0 stars) from ~162,000 users on ~62,000 movies. The standard
public benchmark for collaborative filtering.

Fetch and unpack into `data/raw/`:

```bash
mkdir -p data/raw
curl -L -o data/raw/ml-25m.zip https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip data/raw/ml-25m.zip -d data/raw
```

This produces `data/raw/ml-25m/ratings.csv` with columns
`userId, movieId, rating, timestamp`, which is the path `config/movielens.yaml`
points at. The lighter `ml-latest-small` archive (100k ratings) is a drop-in
substitute for quick local runs — point `data.ratings_path` at its
`ratings.csv`.

Terms of use: MovieLens data is released by GroupLens for research; see the
README inside the archive.

## Heavier alternative: Amazon Reviews

[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab) —
hundreds of millions of reviews across product categories. Use this when you
want a dataset large and sparse enough to justify the Spark MLlib ALS path
(`recsys.spark_als`) rather than the in-memory numpy ALS. Map its
`user_id, parent_asin, rating, timestamp` fields onto the same user/item/rating
columns in the config and the pipeline is unchanged; only the loader differs.
