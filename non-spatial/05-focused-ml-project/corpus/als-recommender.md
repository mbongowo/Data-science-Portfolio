# ALS Recommender — matrix-factorisation collaborative filtering

The ALS Recommender builds a movie recommender by matrix factorisation using
Alternating Least Squares (ALS) on the MovieLens-25M dataset. It compares the
personalised ALS model against a non-personalised popularity baseline on ranking
metrics — Precision, Recall and NDCG at K — to show how much personalisation
adds. A pure-numpy ALS reference implementation is paired with a Spark MLlib
wrapper for scale.

The project includes explicit-feedback, bias-corrected and implicit-feedback
variants of ALS, covering the main flavours of collaborative filtering used in
production recommender systems.
