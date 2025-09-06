from kickbase_api.user_management import get_players_in_squad
from kickbase_api.league_data import get_players_on_market
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np


def live_data_predictions(today_df, model, features):
    """Make live data predictions for today_df using the trained model"""

    # Set features and copy df
    today_df_features = today_df[features]
    today_df_results = today_df.copy()

    # Predict mv_target
    today_df_results["predicted_mv_target"] = np.round(
        model.predict(today_df_features), 2
    )

    # Sort by predicted_mv_target descending
    today_df_results = today_df_results.sort_values(
        "predicted_mv_target", ascending=False
    )

    # Filter date to today or yesterday if before 22:15 (mv update ~22:15)
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    cutoff_time = now.replace(hour=22, minute=15, second=0, microsecond=0)
    date = (now - timedelta(days=1)) if now <= cutoff_time else now
    date = date.date()

    # Drop rows where NaN mv
    today_df_results = today_df_results.dropna(subset=["mv"])

    # Keep only relevant columns
    today_df_results = today_df_results[
        [
            "player_id",
            "first_name",
            "last_name",
            "position",
            "team_name",
            "date",
            "mv_change_1d",
            "mv_trend_1d",
            "mv",
            "predicted_mv_target",
        ]
    ]

    return today_df_results


def join_current_squad(token, league_id, today_df_results):
    """Join predictions with current squad; be robust wrt column names."""

    squad_players = get_players_in_squad(token, league_id)
    squad_df = pd.DataFrame(squad_players.get("it", []))

    # Join: today_df (player_id)  ↔︎  squad_df ("i")
    squad_df = pd.merge(
        today_df_results, squad_df, left_on="player_id", right_on="i", how="inner"
    )

    cols = squad_df.columns

    # mv normalisieren (Merge kann mv_x/mv_y erzeugen)
    if "mv_x" in cols:
        squad_df = squad_df.rename(columns={"mv_x": "mv"})
    elif "mv" not in cols:
        squad_df["mv"] = np.nan

    # Startelf-Wahrscheinlichkeit auf s_11_prob bringen
    if "prob" in cols:
        squad_df = squad_df.rename(columns={"prob": "s_11_prob"})
    elif "s11_prob" in cols:
        squad_df = squad_df.rename(columns={"s11_prob": "s_11_prob"})
    if "s_11_prob" not in squad_df.columns:
        squad_df["s_11_prob"] = np.nan

    # mv_change_1d → mv_change_yesterday
    if "mv_change_1d" in squad_df.columns:
        squad_df = squad_df.rename(columns={"mv_change_1d": "mv_change_yesterday"})
    elif "mv_change_yesterday" not in squad_df.columns:
        squad_df["mv_change_yesterday"] = np.nan

    # Relevante Spalten sicher auswählen
    keep = [
        "first_name",
        "last_name",
        "team_name",
        "mv",
        "mv_change_yesterday",
        "predicted_mv_target",
        "s_11_prob",
    ]
    for c in keep:
        if c not in squad_df.columns:
            squad_df[c] = np.nan
    squad_df = squad_df[keep]

    print("DEBUG squad_df columns:", list(squad_df.columns))
    return squad_df


def join_current_market(token, league_id, today_df_results):
    """Join predictions with current market; be robust wrt column names."""

    players_on_market = get_players_on_market(token, league_id)
    market_df = pd.DataFrame(players_on_market)

    bid_df = pd.merge(
        today_df_results, market_df, left_on="player_id", right_on="id", how="inner"
    )

    # Stunden bis Ablauf
    if "exp" in bid_df.columns:
        bid_df["hours_to_exp"] = np.round(bid_df["exp"] / 3600, 2)
    else:
        bid_df["hours_to_exp"] = np.nan

    # Heute ablaufend?
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    next_22 = now.replace(hour=22, minute=0, second=0, microsecond=0)
    diff = np.round((next_22 - now).total_seconds() / 3600, 2)
    bid_df["expiring_today"] = bid_df["hours_to_exp"] < diff

    # sinnvolle Predictions
    if "predicted_mv_target" in bid_df.columns:
        bid_df = bid_df[bid_df["predicted_mv_target"] > 5000].sort_values(
            "predicted_mv_target", ascending=False
        )

    # Startelf-Wahrscheinlichkeit auf s_11_prob bringen
    if "prob" in bid_df.columns:
        bid_df = bid_df.rename(columns={"prob": "s_11_prob"})
    elif "s11_prob" in bid_df.columns:
        bid_df = bid_df.rename(columns={"s11_prob": "s_11_prob"})
    if "s_11_prob" not in bid_df.columns:
        bid_df["s_11_prob"] = np.nan

    # mv_change_1d → mv_change_yesterday
    if "mv_change_1d" in bid_df.columns:
        bid_df = bid_df.rename(columns={"mv_change_1d": "mv_change_yesterday"})
    elif "mv_change_yesterday" not in bid_df.columns:
        bid_df["mv_change_yesterday"] = np.nan

    keep = [
        "first_name",
        "last_name",
        "team_name",
        "mv",
        "mv_change_yesterday",
        "predicted_mv_target",
        "s_11_prob",
        "hours_to_exp",
        "expiring_today",
    ]
    for c in keep:
        if c not in bid_df.columns:
            bid_df[c] = np.nan
    bid_df = bid_df[keep]

    print("DEBUG bid_df columns:", list(bid_df.columns))
    return bid_df
