"""Streamlit entry point for the Crop Recommender web app.

Run locally with::

    streamlit run app/streamlit_app.py

The app loads the bundled synthetic sample dataset, trains the pure-numpy
``SoftmaxClassifier`` once (cached), then takes seven soil-and-climate inputs and
returns the recommended crop with a ranked top-3 and confidences. It needs only
numpy / pandas / streamlit because it trains the model itself — there is no
pre-trained binary to ship.
"""

from __future__ import annotations

# When a host runs this file directly (Streamlit Community Cloud runs
# app/streamlit_app.py), only this file's directory is on sys.path, so the
# `croprec` package under ../src is not importable. Put it on the path.
import sys as _sys
from pathlib import Path as _Path

_app_dir = _Path(__file__).resolve().parent
_src_dir = _app_dir.parent / "src"
for _p in (str(_src_dir), str(_app_dir.parent)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import pandas as pd
import streamlit as st

from croprec.data import FEATURE_COLUMNS, encode_labels, stratified_split
from croprec.metrics import accuracy
from croprec.model import SoftmaxClassifier, standardize
from croprec.recommend import recommend

SAMPLE_CSV = _app_dir / "sample_data" / "crop_samples.csv"

# Sensible widget ranges and defaults for the seven features.
FEATURE_RANGES: dict[str, tuple[float, float, float]] = {
    # name: (min, max, default)
    "N": (0.0, 140.0, 70.0),
    "P": (0.0, 140.0, 45.0),
    "K": (0.0, 140.0, 45.0),
    "temperature": (8.0, 44.0, 26.0),
    "humidity": (10.0, 100.0, 70.0),
    "ph": (3.5, 9.0, 6.2),
    "rainfall": (20.0, 300.0, 120.0),
}
FEATURE_HELP: dict[str, str] = {
    "N": "Soil nitrogen (kg/ha)",
    "P": "Soil phosphorus (kg/ha)",
    "K": "Soil potassium (kg/ha)",
    "temperature": "Average temperature (degrees C)",
    "humidity": "Relative humidity (%)",
    "ph": "Soil pH",
    "rainfall": "Rainfall (mm)",
}

st.set_page_config(
    page_title="Crop Recommender",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner=False)
def _load_samples() -> pd.DataFrame:
    """Load the bundled synthetic sample dataset."""
    return pd.read_csv(SAMPLE_CSV)


@st.cache_resource(show_spinner=False)
def _train_model():
    """Train the numpy SoftmaxClassifier once on the bundled data (cached).

    Returns the fitted model, the crop ``classes``, the training z-score
    statistics and the holdout accuracy used as a trust signal.
    """
    df = _load_samples()
    y, classes = encode_labels(df["label"].to_numpy())
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)

    train_idx, test_idx = stratified_split(y, (0.7, 0.3), seed=0)
    X_train_s, mean, std = standardize(X[train_idx])
    X_test_s, _, _ = standardize(X[test_idx], mean=mean, std=std)

    model = SoftmaxClassifier().fit(
        X_train_s, y[train_idx], lr=0.5, epochs=600, l2=1e-3, seed=0
    )
    holdout_accuracy = accuracy(y[test_idx], model.predict(X_test_s))
    return model, classes, mean, std, float(holdout_accuracy)


def main() -> None:
    st.title("🌱 Crop Recommender")
    st.write(
        "Enter your soil and climate conditions to get a recommended crop with "
        "a ranked top-3 and confidences. Framed for Cameroon smallholders."
    )
    st.caption(
        "The bundled model is trained at startup on a small **synthetic** "
        "dataset for demonstration. It is decision support, not a substitute "
        "for agricultural extension advice."
    )

    model, classes, mean, std, holdout_accuracy = _train_model()

    with st.form("inputs"):
        st.subheader("Soil & climate")
        cols = st.columns(2)
        sample: dict[str, float] = {}
        for i, feat in enumerate(FEATURE_COLUMNS):
            lo, hi, default = FEATURE_RANGES[feat]
            with cols[i % 2]:
                sample[feat] = st.slider(
                    feat,
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(default),
                    help=FEATURE_HELP[feat],
                )
        submitted = st.form_submit_button("Recommend crop", type="primary")

    if submitted:
        ranked = recommend(model, classes, mean, std, sample)
        top_crop, top_prob = ranked[0]

        st.success(f"Recommended crop: **{top_crop}**  ({top_prob:.0%} confidence)")

        st.subheader("Top 3 recommendations")
        top3 = ranked[:3]
        chart_df = pd.DataFrame(
            {"crop": [c for c, _ in top3], "probability": [p for _, p in top3]}
        ).set_index("crop")
        st.bar_chart(chart_df)
        st.table(
            pd.DataFrame(
                {
                    "crop": [c for c, _ in top3],
                    "confidence": [f"{p:.1%}" for _, p in top3],
                }
            )
        )

    st.divider()
    st.metric("Model holdout accuracy", f"{holdout_accuracy:.1%}")
    st.caption(
        f"Pure-numpy softmax classifier trained on {len(classes)} crops "
        "from the bundled synthetic sample. Swap in the real Kaggle Crop "
        "Recommendation CSV (same 7 columns) for a production model."
    )


if __name__ == "__main__":
    main()
else:
    main()
