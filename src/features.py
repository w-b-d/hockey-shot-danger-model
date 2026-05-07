"""Feature engineering for NHL shot data.

Designed for MoneyPuck shot CSVs (124 columns, available at moneypuck.com/data.htm).
Also works with other schemas that share similar column names — see COLUMN_ALIASES.

The functions are deliberately small and well-named so the notebook and the
Streamlit app can re-use the same pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NET_X = 89.0  # goal line x-coordinate (offensive zone, adjusted coords)
NET_Y = 0.0

# ---- Column detection -------------------------------------------------------
# Maps our canonical name → list of possible column names in priority order.
# First match wins. The MoneyPuck name is always listed first.

COLUMN_ALIASES: dict[str, list[str]] = {
    # Geometry
    "x":                    ["xCordAdjusted", "arenaAdjustedXCord", "arenaAdjustedXCoord", "xCord", "x_coord", "x"],
    "y":                    ["yCordAdjusted", "arenaAdjustedYCord", "arenaAdjustedYCoord", "yCord", "y_coord", "y"],
    "shot_distance":        ["shotDistance", "arenaAdjustedShotDistance", "distance", "shot_distance"],
    "shot_angle":           ["shotAngleAdjusted", "arenaAdjustedShotAngle", "shotAngle", "angle", "shot_angle"],

    # Target / event
    "goal":                 ["goal", "isGoal", "is_goal"],
    "event":                ["event", "eventType", "event_type"],  # SHOT / GOAL / MISS

    # Shot context
    "shot_type":            ["shotType", "shot_type", "secondaryType"],
    "rebound":              ["shotRebound", "is_rebound", "rebound"],
    "rush":                 ["shotRush", "is_rush", "rush"],
    "empty_net":            ["shotOnEmptyNet", "empty_net"],
    "off_wing":             ["offWing", "off_wing"],
    "angle_change_rebound": ["shotAnglePlusRebound"],       # angle change from prev shot
    "rebound_royal_road":   ["shotAngleReboundRoyalRoad"],  # puck crossed slot on rebound
    "speed_from_last":      ["speedFromLastEvent"],
    "distance_from_last":   ["distanceFromLastEvent"],
    "last_event_category":  ["lastEventCategory"],
    "shot_on_goal":         ["shotWasOnGoal"],
    "shot_generated_rebound": ["shotGeneratedRebound"],

    # Strength / game state
    "home_skaters":         ["homeSkatersOnIce", "home_skaters_on_ice"],
    "away_skaters":         ["awaySkatersOnIce", "away_skaters_on_ice"],
    "shooter_side":         ["team"],           # MoneyPuck: "HOME" / "AWAY"
    "is_home_team":         ["isHomeTeam"],      # MoneyPuck: binary 0/1
    "is_playoff":           ["isPlayoffGame"],

    # Score state
    "home_goals":           ["homeTeamGoals"],
    "away_goals":           ["awayTeamGoals"],

    # Identity
    "team_code":            ["teamCode", "team_code", "shootingTeam", "teamAbbrev", "teamAbbreviation", "teamAbbr"],
    "opponent_code":        ["teamCodeAgainst", "opponentTeamCode", "opponent_code"],
    "season":               ["season", "Season"],
    "game_id":              ["game_id", "gameId"],
    "period":               ["period"],
    "shooter_name":         ["shooterName", "shooter_name"],
    "shooter_id":           ["shooterPlayerId"],
    "shooter_position":     ["playerPositionThatDidEvent"],  # C / L / R / D
    "shooter_handedness":   ["shooterLeftRight"],
    "goalie_name":          ["goalieNameForShot"],

    # MoneyPuck's own xG (for benchmarking)
    "xgoal":                ["xGoal", "xG", "expectedGoals"],

    # Time
    "time":                 ["time"],
    "time_since_last":      ["timeSinceLastEvent"],
    "time_since_faceoff":   ["timeSinceFaceoff"],
}

# validate_columns requires shot geometry plus either a goal flag or an event
# label that can be converted into a goal flag.

TRUE_VALUES = {"1", "TRUE", "T", "YES", "Y", "GOAL"}
FALSE_VALUES = {"0", "FALSE", "F", "NO", "N", "SHOT", "MISS", "MISSED_SHOT", "SAVED_SHOT", ""}


def _column_preview(df: pd.DataFrame, n: int = 20) -> str:
    shown = ", ".join(map(str, df.columns[:n]))
    return shown + (" ..." if len(df.columns) > n else "")


def _expected_aliases(canonical: str) -> str:
    return ", ".join(f"'{name}'" for name in COLUMN_ALIASES[canonical])


def _to_binary(series: pd.Series, *, true_values: set[str] | None = None) -> pd.Series:
    """Coerce common 0/1, bool, and event-label columns into integer flags."""
    true_values = TRUE_VALUES if true_values is None else true_values
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.upper()

    out = pd.Series(np.nan, index=series.index, dtype="float")
    out = out.mask(text.isin(true_values), 1.0)
    out = out.mask(text.isin(FALSE_VALUES), 0.0)
    out = out.fillna(numeric)
    return out.fillna(0).astype(float).clip(lower=0, upper=1).round().astype(int)


def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map canonical names to actual column names found in this DataFrame.

    Returns e.g. {"x": "xCordAdjusted", "goal": "goal", ...}.
    Missing keys mean the source data lacks that column.
    """
    mapping: dict[str, str] = {}
    cols_lower = {c.lower(): c for c in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                mapping[canonical] = alias
                break
            if alias.lower() in cols_lower:
                mapping[canonical] = cols_lower[alias.lower()]
                break
    return mapping


def validate_columns(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems. Empty list = all good."""
    cols = detect_columns(df)
    problems: list[str] = []

    has_coords = "x" in cols and "y" in cols
    has_fallback = "shot_distance" in cols
    has_target = "goal" in cols or "event" in cols

    if not has_coords and not has_fallback:
        problems.append(
            "Missing shot geometry. Need coordinates such as "
            f"{_expected_aliases('x')} + {_expected_aliases('y')}, or at "
            f"minimum a distance column such as {_expected_aliases('shot_distance')}. "
            "Found columns: " + _column_preview(df)
        )
    if not has_target:
        problems.append(
            "Missing goal indicator. Need a binary goal column such as "
            f"{_expected_aliases('goal')}, or an event column such as "
            f"{_expected_aliases('event')} with GOAL/SHOT labels. "
            "Found columns: " + _column_preview(df)
        )
    return problems


def load_shot_data(path: str | Path) -> pd.DataFrame:
    """Read a shots CSV and validate it has the columns we need."""
    df = pd.read_csv(path, low_memory=False)
    problems = validate_columns(df)
    if problems:
        raise ValueError(
            "Data validation failed:\n  - " + "\n  - ".join(problems) + "\n\n"
            "If you're using MoneyPuck data, make sure you downloaded the shots "
            "file (not skaters or lines). See DATA_INSTRUCTIONS.md for details."
        )
    return df


def clean_data(df: pd.DataFrame, drop_empty_net: bool = True) -> pd.DataFrame:
    """Drop rows that aren't useful for goal-probability modeling.

    - rows missing coordinates are dropped
    - empty-net shots are excluded (they distort goal rates)
    - goal column is coerced to int 0/1
    """
    cols = detect_columns(df)
    out = df.copy()

    # Drop missing geometry
    if "x" in cols and "y" in cols:
        out = out.dropna(subset=[cols["x"], cols["y"]])
    elif "shot_distance" in cols:
        out = out.dropna(subset=[cols["shot_distance"]])

    # Coerce goal to binary int. MoneyPuck has a binary `goal` column, while
    # some public event feeds only expose a SHOT/GOAL event label.
    if "goal" in cols:
        out = out.dropna(subset=[cols["goal"]])
        out["goal"] = _to_binary(out[cols["goal"]])
    elif "event" in cols:
        out = out.dropna(subset=[cols["event"]])
        out["goal"] = _to_binary(out[cols["event"]], true_values={"GOAL"})

    # Drop empty-net shots — they're ~90% goal rate and dominate the model
    if drop_empty_net and "empty_net" in cols:
        out = out[_to_binary(out[cols["empty_net"]]) == 0]

    return out.reset_index(drop=True)


# ---- Geometry helpers --------------------------------------------------------

def _mirror_to_offensive_zone(
    x: pd.Series, y: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """MoneyPuck's xCordAdjusted already standardizes to one end, but if raw
    xCord is used, some shots have x < 0. Flip those so the net is at (89, 0)."""
    sign = np.where(x < 0, -1, 1)
    return x * sign, y * sign


def compute_distance(x: pd.Series, y: pd.Series) -> pd.Series:
    return np.sqrt((NET_X - x) ** 2 + (NET_Y - y) ** 2)


def compute_angle(x: pd.Series, y: pd.Series) -> pd.Series:
    """Angle from the line perpendicular to the goal line, in degrees [0, 90].
    0 = dead-center, 90 = along the goal line."""
    dx = np.maximum(NET_X - x, 0.01)
    return np.degrees(np.arctan2(np.abs(y), dx))


# ---- Main feature pipeline ---------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add geometry and context columns used by the danger model.

    Works on any DataFrame that passes validate_columns(). MoneyPuck-specific
    bonus features (off_wing, speed_from_last, angle change on rebounds, etc.)
    are added when the columns exist, and silently skipped otherwise.
    """
    cols = detect_columns(df)
    out = df.copy()

    # ---- Geometry -----------------------------------------------------------
    if "x" in cols and "y" in cols:
        x_off, y_off = _mirror_to_offensive_zone(
            out[cols["x"]].astype(float), out[cols["y"]].astype(float),
        )
        out["x_off"] = x_off
        out["y_off"] = y_off
        out["distance"] = compute_distance(x_off, y_off)
        out["angle"] = compute_angle(x_off, y_off)
    elif "shot_distance" in cols:
        out["distance"] = out[cols["shot_distance"]].astype(float)
        out["angle"] = (
            out[cols["shot_angle"]].astype(float).abs()
            if "shot_angle" in cols else 0.0
        )
        out["x_off"] = NET_X - out["distance"] * np.cos(np.radians(out["angle"]))
        out["y_off"] = out["distance"] * np.sin(np.radians(out["angle"]))

    out["distance_sq"] = out["distance"] ** 2
    out["angle_x_distance"] = out["distance"] * out["angle"]
    out["log_distance"] = np.log1p(out["distance"])

    # ---- Context flags (core — always present in MoneyPuck) -----------------
    out["is_rebound"] = _to_binary(out[cols["rebound"]]) if "rebound" in cols else 0
    out["is_rush"] = _to_binary(out[cols["rush"]]) if "rush" in cols else 0

    # ---- Strength state from the shooter's perspective ----------------------
    if "is_home_team" in cols:
        is_home = _to_binary(out[cols["is_home_team"]])
    elif "shooter_side" in cols:
        side = out[cols["shooter_side"]].astype("string").str.strip().str.upper()
        is_home = (side == "HOME").astype(int) if side.isin(["HOME", "AWAY"]).any() else None
    else:
        is_home = None

    if {"home_skaters", "away_skaters"} <= cols.keys():
        home = out[cols["home_skaters"]].fillna(5).astype(float)
        away = out[cols["away_skaters"]].fillna(5).astype(float)
        if is_home is not None:
            diff = is_home * (home - away) + (1 - is_home) * (away - home)
        else:
            diff = home - away
    else:
        diff = pd.Series(0, index=out.index)

    out["strength_diff"] = diff
    out["is_even_strength"] = (diff == 0).astype(int)
    out["is_power_play"]   = (diff > 0).astype(int)
    out["is_short_handed"] = (diff < 0).astype(int)

    # ---- Score state from the shooter's perspective -------------------------
    if {"home_goals", "away_goals"} <= cols.keys():
        home_goals = out[cols["home_goals"]].fillna(0).astype(float)
        away_goals = out[cols["away_goals"]].fillna(0).astype(float)
        if is_home is not None:
            score_diff = (
                is_home * (home_goals - away_goals)
                + (1 - is_home) * (away_goals - home_goals)
            )
        else:
            score_diff = home_goals - away_goals
    else:
        score_diff = pd.Series(0, index=out.index)

    out["score_diff"] = score_diff
    out["is_leading"] = (score_diff > 0).astype(int)
    out["is_trailing"] = (score_diff < 0).astype(int)

    # ---- Bonus MoneyPuck features (when available) --------------------------

    # Off-wing shot (puck on the opposite side from shooter handedness)
    out["is_off_wing"] = (
        _to_binary(out[cols["off_wing"]]) if "off_wing" in cols else 0
    )

    # Speed into the shot — higher speed = goalie has less time to set
    if "speed_from_last" in cols:
        out["speed_from_last"] = out[cols["speed_from_last"]].fillna(0).astype(float)
    else:
        out["speed_from_last"] = 0.0

    # Angle change from the previous shot (rebounds / cross-crease plays)
    if "angle_change_rebound" in cols:
        out["angle_change"] = out[cols["angle_change_rebound"]].fillna(0).astype(float)
    else:
        out["angle_change"] = 0.0

    # Shooter position: D / C / L / R
    if "shooter_position" in cols:
        pos = out[cols["shooter_position"]].astype(str).str.upper().fillna("UNKNOWN")
        pos_dummies = pd.get_dummies(pos, prefix="pos").astype(int)
        out = pd.concat([out, pos_dummies], axis=1)

    # ---- Shot type dummies --------------------------------------------------
    if "shot_type" in cols:
        st = out[cols["shot_type"]].astype(str).str.upper().fillna("UNKNOWN")
        st = st.replace({
            "BACK": "BACKHAND", "TIP": "TIP-IN", "WRAP": "WRAP-AROUND",
        })
        dummies = pd.get_dummies(st, prefix="shottype").astype(int)
        out = pd.concat([out, dummies], axis=1)

    # ---- Target -------------------------------------------------------------
    if "goal" in cols:
        out["goal"] = _to_binary(out[cols["goal"]])
    elif "event" in cols:
        out["goal"] = _to_binary(out[cols["event"]], true_values={"GOAL"})

    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of columns the model should consume.

    Only includes columns that actually exist in this DataFrame, so the same
    function works on MoneyPuck (which has off_wing, speed, etc.) and on the
    synthetic sample (which doesn't).
    """
    base = [
        "distance", "angle", "distance_sq", "angle_x_distance", "log_distance",
        "is_rebound", "is_rush",
        "is_even_strength", "is_power_play", "is_short_handed",
        "score_diff", "is_leading", "is_trailing",
        "is_off_wing", "speed_from_last", "angle_change",
    ]
    shottype_cols = sorted(c for c in df.columns if c.startswith("shottype_"))
    pos_cols = sorted(c for c in df.columns if c.startswith("pos_"))
    all_candidates = base + shottype_cols + pos_cols

    present = [c for c in all_candidates if c in df.columns]

    # Drop features that are constant (all zeros / all same value) — they add
    # noise to the model without signal.
    return [c for c in present if df[c].nunique() > 1]
