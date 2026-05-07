"""Streamlit dashboard for the NHL shot-danger model.

Run:
    streamlit run app.py

Drop a MoneyPuck shots CSV at data/shots.csv, or upload one in the sidebar.
See DATA_INSTRUCTIONS.md for download details.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.features import (
    clean_data,
    detect_columns,
    engineer_features,
    feature_columns,
    load_shot_data,
    validate_columns,
)
from src.model import (
    calibration_table,
    coefficient_table,
    evaluate,
    split_data,
    train_logistic,
)
from src.visuals import (
    plot_calibration,
    plot_danger_heatmap,
    plot_goal_rate_by_angle,
    plot_goal_rate_by_distance,
    plot_shot_map,
)

st.set_page_config(page_title="NHL Shot Danger Model", layout="wide")
st.title("NHL Shot Danger Model")
st.caption("Geometry-based scoring probability — trained on MoneyPuck public shot data")

PLOT_DPI = 90
RINK_FIGSIZE = (6.0, 4.6)
GEOMETRY_FIGSIZE = (9.0, 3.2)
CALIBRATION_FIGSIZE = (4.2, 4.2)


def render_plot(fig) -> None:
    """Keep matplotlib plots compact in Streamlit's wide layout."""
    kwargs = {"bbox_inches": "tight"}
    if "width" in inspect.signature(st.pyplot).parameters:
        st.pyplot(fig, width="content", **kwargs)
    else:
        st.pyplot(fig, use_container_width=False, **kwargs)


# --- Sidebar: data loading ---------------------------------------------------
st.sidebar.header("Data")
default_path = Path("data") / "shots.csv"
sample_path = Path("data") / "sample_shots.csv"

uploaded = st.sidebar.file_uploader("Upload a MoneyPuck shots CSV", type=["csv"])
use_default = st.sidebar.checkbox(
    "Use data/shots.csv (MoneyPuck)",
    value=default_path.exists() and uploaded is None,
)
use_sample = st.sidebar.checkbox(
    "Use data/sample_shots.csv (synthetic demo)",
    value=(not default_path.exists() and uploaded is None and sample_path.exists()),
)


@st.cache_data(show_spinner="Loading CSV...")
def _load_uploaded(buf) -> pd.DataFrame:
    df = pd.read_csv(buf, low_memory=False)
    problems = validate_columns(df)
    if problems:
        st.error("Data validation failed:\n\n" + "\n\n".join(problems))
        st.stop()
    return df


@st.cache_data(show_spinner="Loading CSV...")
def _load_path(path: str) -> pd.DataFrame:
    return load_shot_data(path)


if uploaded is not None:
    raw = _load_uploaded(uploaded)
elif use_default and default_path.exists():
    raw = _load_path(str(default_path))
elif use_sample and sample_path.exists():
    raw = _load_path(str(sample_path))
else:
    st.info(
        "**No data found.** Download a MoneyPuck shots CSV and place it at "
        "`data/shots.csv`, or upload one in the sidebar.\n\n"
        "See `DATA_INSTRUCTIONS.md` for step-by-step download instructions.\n\n"
        "For a quick demo without downloading, run `python make_sample_data.py` "
        "to generate a synthetic sample."
    )
    st.stop()

cols = detect_columns(raw)
is_real_data = "xgoal" in cols  # MoneyPuck data has this; synthetic doesn't
data_label = "MoneyPuck" if is_real_data else "synthetic sample"
st.sidebar.success(f"Loaded {len(raw):,} rows ({data_label})")

# --- Sidebar: filters --------------------------------------------------------
st.sidebar.header("Filters")
filtered = raw.copy()

if "season" in cols:
    seasons = sorted(filtered[cols["season"]].dropna().unique())
    chosen = st.sidebar.multiselect(
        "Season", seasons, default=seasons[-1:] if seasons else [],
    )
    if chosen:
        filtered = filtered[filtered[cols["season"]].isin(chosen)]

if "team_code" in cols:
    teams = sorted(filtered[cols["team_code"]].dropna().unique())
    chosen = st.sidebar.multiselect("Team", teams)
    if chosen:
        filtered = filtered[filtered[cols["team_code"]].isin(chosen)]

if "shot_type" in cols:
    types = sorted(filtered[cols["shot_type"]].dropna().astype(str).unique())
    chosen = st.sidebar.multiselect("Shot type", types)
    if chosen:
        filtered = filtered[filtered[cols["shot_type"]].astype(str).isin(chosen)]

strength_filter = st.sidebar.selectbox(
    "Strength state", ["All", "Even strength", "Power play", "Short handed"],
)

# --- Pipeline ----------------------------------------------------------------
clean = clean_data(filtered)
features = engineer_features(clean)

# Apply strength filter after engineering (we need the derived columns)
if strength_filter == "Even strength":
    features = features[features["is_even_strength"] == 1]
elif strength_filter == "Power play":
    features = features[features["is_power_play"] == 1]
elif strength_filter == "Short handed":
    features = features[features["is_short_handed"] == 1]

features = features.copy()
feat_cols = feature_columns(features)

if "goal" not in features.columns or features["goal"].nunique() < 2:
    st.warning(
        "Need shots with both goals and non-goals to train. "
        "Try broadening your filters."
    )
    st.stop()

if len(features) < 100:
    st.warning(f"Only {len(features)} shots after filtering — results will be noisy.")

try:
    X_train, X_test, y_train, y_test = split_data(features, feat_cols)
except ValueError as exc:
    st.warning(str(exc))
    st.stop()

model = train_logistic(X_train, y_train)
metrics = evaluate(model, X_test, y_test)
features["xg_pred"] = model.predict_proba(features[feat_cols])[:, 1]

# --- Metrics row -------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Shots", f"{len(features):,}")
c2.metric("Goal rate", f"{features['goal'].mean()*100:.1f}%")
c3.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
c4.metric("Log loss", f"{metrics['log_loss']:.3f}")
c5.metric("Brier", f"{metrics['brier']:.3f}")

# --- Tabs --------------------------------------------------------------------
tab_map, tab_geom, tab_model, tab_top = st.tabs(
    ["Shot map", "Geometry", "Model", "Top dangerous shots"],
)

with tab_map:
    fig, ax = plt.subplots(figsize=RINK_FIGSIZE, dpi=PLOT_DPI)
    plot_shot_map(features, ax=ax)
    render_plot(fig)
    plt.close(fig)

with tab_geom:
    fig, axs = plt.subplots(1, 2, figsize=GEOMETRY_FIGSIZE, dpi=PLOT_DPI)
    plot_goal_rate_by_distance(features, ax=axs[0])
    plot_goal_rate_by_angle(features, ax=axs[1])
    fig.tight_layout()
    render_plot(fig)
    plt.close(fig)

with tab_model:
    fig, ax = plt.subplots(figsize=RINK_FIGSIZE, dpi=PLOT_DPI)
    plot_danger_heatmap(features, value="xg_pred", ax=ax)
    render_plot(fig)
    plt.close(fig)

    st.subheader("Logistic regression coefficients (standardized features)")
    st.caption(
        "Positive = increases goal probability. Negative = decreases it. "
        "Coefficients are on the standardized scale, so magnitudes are "
        "directly comparable across features."
    )
    st.dataframe(coefficient_table(model, feat_cols), use_container_width=True)

    st.subheader("Calibration")
    st.caption(
        "For an xG model, probabilities should be honest: shots predicted at "
        "roughly 10% should score roughly 10% of the time over a large sample."
    )
    proba = model.predict_proba(X_test)[:, 1]
    cal = calibration_table(y_test, proba)
    fig, ax = plt.subplots(figsize=CALIBRATION_FIGSIZE, dpi=PLOT_DPI)
    plot_calibration(cal, ax=ax)
    render_plot(fig)
    plt.close(fig)
    st.dataframe(cal, use_container_width=True)

with tab_top:
    st.subheader("Highest predicted-danger shots")
    show = ["distance", "angle", "is_rebound", "is_rush", "goal", "xg_pred"]
    if "shot_type" in cols:
        show.insert(0, cols["shot_type"])
    if "shooter_name" in cols:
        show.insert(0, cols["shooter_name"])
    if is_real_data and "xgoal" in cols:
        show.append(cols["xgoal"])
    available = [c for c in show if c in features.columns]
    top = features.nlargest(50, "xg_pred")[available]
    top = top.rename(columns={"is_rush": "is_rush_sparse_proxy"})

    fmt = {"xg_pred": "{:.3f}", "distance": "{:.1f}", "angle": "{:.1f}"}
    if "xgoal" in cols and cols["xgoal"] in available:
        fmt[cols["xgoal"]] = "{:.3f}"
    st.dataframe(top.style.format(fmt), use_container_width=True)
    if "is_rush_sparse_proxy" in top.columns:
        st.caption(
            "Rush is shown as MoneyPuck's public shotRush flag, which can be "
            "sparse and should be treated as a rough transition-offense proxy."
        )

st.divider()
st.caption(
    f"Data: {data_label} ({len(features):,} shots after filtering). "
    "Empty-net shots excluded. Model uses geometry (distance, angle), "
    "shot type, and context flags (rebound, public rush proxy, strength state"
    + (", off-wing, speed, angle change" if is_real_data else "")
    + ")."
)
