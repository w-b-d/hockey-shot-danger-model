"""Plotting helpers for the shot-danger analysis.

All plots accept an optional matplotlib `ax` so they can be composed inside
notebooks, the Streamlit app, or saved as standalone figures.
"""

from __future__ import annotations

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NHL_RED = "#C8102E"
NHL_BLUE = "#0033A0"


def draw_rink_half(ax=None, color: str = "black", lw: float = 1.0):
    """Draw a simplified offensive-zone half rink.

    Coordinates: x in [0, 100], y in [-42.5, 42.5]. Goal line at x=89,
    blue line at x=25 (relative to center ice).
    """
    if ax is None:
        ax = plt.gca()

    # Boards
    ax.add_patch(patches.Rectangle((0, -42.5), 100, 85, fill=False, ec=color, lw=lw))
    # Center red line (we only show the right half of the rink)
    ax.plot([0, 0], [-42.5, 42.5], color=NHL_RED, lw=lw * 1.4)
    # Blue line
    ax.plot([25, 25], [-42.5, 42.5], color=NHL_BLUE, lw=lw * 1.2)
    # Goal line
    ax.plot([89, 89], [-37.85, 37.85], color=NHL_RED, lw=lw)
    # Net
    ax.add_patch(patches.Rectangle((89, -3), 3, 6, fill=False, ec=color, lw=lw))
    # Offensive-zone faceoff circles
    for fy in (-22, 22):
        ax.add_patch(patches.Circle((69, fy), 15, fill=False, ec=NHL_RED, lw=lw))
        ax.plot(69, fy, marker="+", color=NHL_RED, ms=8, mew=lw)
    # Crease
    ax.add_patch(patches.Wedge((89, 0), 6, 90, 270, fill=False, ec=NHL_BLUE, lw=lw))
    # Trapezoid behind the net (approximate)
    ax.plot([89, 100], [-11, -14], color=color, lw=lw)
    ax.plot([89, 100], [11, 14], color=color, lw=lw)

    ax.set_xlim(0, 100)
    ax.set_ylim(-45, 45)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_shot_map(df: pd.DataFrame, sample: int = 5000, ax=None):
    """Scatter shots on a half rink. Goals red, misses blue."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    draw_rink_half(ax)

    plot_df = df if len(df) <= sample else df.sample(sample, random_state=7)
    misses = plot_df[plot_df["goal"] == 0]
    goals = plot_df[plot_df["goal"] == 1]

    ax.scatter(misses["x_off"], misses["y_off"], s=4, alpha=0.15,
               color=NHL_BLUE, label="Non-goal")
    ax.scatter(goals["x_off"], goals["y_off"], s=14, alpha=0.65,
               color=NHL_RED, edgecolor="white", linewidth=0.3, label="Goal")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_title("Shot map — goals (red) vs non-goals (blue)")
    return ax


def plot_goal_rate_by_distance(df: pd.DataFrame, ax=None, n_bins: int = 20):
    """Empirical goal rate as a function of shot distance."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 80, n_bins + 1)
    binned = df.assign(_bin=pd.cut(df["distance"], bins, include_lowest=True))
    rate = (
        binned.groupby("_bin", observed=True)["goal"]
              .agg(["mean", "size"]).reset_index()
    )
    centers = [b.mid for b in rate["_bin"]]
    ax.plot(centers, rate["mean"], marker="o", color=NHL_BLUE)
    ax.set_xlabel("Distance from net (ft)")
    ax.set_ylabel("Goal rate")
    ax.set_title("Goal probability by shot distance")
    ax.grid(alpha=0.3)
    return ax


def plot_goal_rate_by_angle(df: pd.DataFrame, ax=None, n_bins: int = 18):
    """Empirical goal rate as a function of shot angle."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 90, n_bins + 1)
    binned = df.assign(_bin=pd.cut(df["angle"], bins, include_lowest=True))
    rate = (
        binned.groupby("_bin", observed=True)["goal"]
              .agg(["mean", "size"]).reset_index()
    )
    centers = [b.mid for b in rate["_bin"]]
    ax.plot(centers, rate["mean"], marker="o", color=NHL_RED)
    ax.set_xlabel("Shot angle from net (deg, 0 = straight on)")
    ax.set_ylabel("Goal rate")
    ax.set_title("Goal probability by shot angle")
    ax.grid(alpha=0.3)
    return ax


def plot_danger_heatmap(
    df: pd.DataFrame,
    value: str = "goal",
    ax=None,
    gridsize: int = 30,
    vmax: float | None = None,
    mincnt: int = 5,
):
    """Hexbin map of the average value (goal rate or predicted xG) per cell."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    draw_rink_half(ax)
    hb = ax.hexbin(
        df["x_off"], df["y_off"], C=df[value],
        reduce_C_function=np.mean, gridsize=gridsize,
        cmap="Reds", mincnt=mincnt, alpha=0.85, vmax=vmax,
    )
    cb = plt.colorbar(hb, ax=ax, fraction=0.03)
    cb.set_label(f"Mean {value}")
    ax.set_title(f"Shot danger heatmap — average {value}")
    return ax


def plot_feature_importance(imp_df: pd.DataFrame, top_n: int = 12, ax=None):
    """Horizontal bar chart of feature importance or coefficients."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    col = "importance" if "importance" in imp_df.columns else "coefficient"
    plot_df = imp_df.head(top_n).iloc[::-1]
    colors = [NHL_RED if v < 0 else NHL_BLUE for v in plot_df[col]]
    ax.barh(plot_df["feature"], plot_df[col], color=colors)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_title(f"Top {top_n} features")
    return ax


def plot_calibration(cal_df: pd.DataFrame, ax=None):
    """Diagonal = perfect calibration. Points above = under-predicting goals."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, cal_df["mean_pred"].max() * 1.05],
            [0, cal_df["mean_pred"].max() * 1.05],
            color="grey", linestyle="--", lw=1, label="Perfect calibration")
    ax.plot(cal_df["mean_pred"], cal_df["actual"], marker="o", color=NHL_BLUE,
            label="Model")
    ax.set_xlabel("Predicted goal probability (bin mean)")
    ax.set_ylabel("Actual goal rate")
    ax.set_title("Calibration plot")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax
