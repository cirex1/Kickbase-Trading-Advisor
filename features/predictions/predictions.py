def join_current_squad(token, league_id, today_df_results):
    from kickbase_api.user_management import get_players_in_squad
    import numpy as np
    import pandas as pd

    squad_players = get_players_in_squad(token, league_id)
    squad_df = pd.DataFrame(squad_players.get("it", []))

    # Join: today_df (player_id)  ↔︎  squad_df ("i")
    squad_df = pd.merge(today_df_results, squad_df, left_on="player_id", right_on="i", how="inner")

    # --- Spalten vereinheitlichen/absichern ---
    cols = squad_df.columns

    # Marktwert vereinheitlichen
    if "mv_x" in cols:
        squad_df = squad_df.rename(columns={"mv_x": "mv"})
    elif "mv" not in cols:
        squad_df["mv"] = np.nan

    # Startelf-Proba vereinheitlichen
    if "prob" in cols:
        squad_df = squad_df.rename(columns={"prob": "s_11_prob"})
    elif "s11_prob" in cols:
        squad_df = squad_df.rename(columns={"s11_prob": "s_11_prob"})
    if "s_11_prob" not in squad_df.columns:
        squad_df["s_11_prob"] = np.nan  # falls gar nicht vorhanden

    # mv_change_1d → mv_change_yesterday
    if "mv_change_1d" in squad_df.columns:
        squad_df = squad_df.rename(columns={"mv_change_1d": "mv_change_yesterday"})
    elif "mv_change_yesterday" not in squad_df.columns:
        squad_df["mv_change_yesterday"] = np.nan

    # Relevante Spalten sicher auswählen
    keep = [
        "first_name", "last_name", "team_name", "mv",
        "mv_change_yesterday", "predicted_mv_target", "s_11_prob"
    ]
    for c in keep:
        if c not in squad_df.columns:
            squad_df[c] = np.nan
    squad_df = squad_df[keep]

    # Debug ins Actions-Log
    print("DEBUG squad_df columns:", list(squad_df.columns))
    return squad_df


def join_current_market(token, league_id, today_df_results):
    from kickbase_api.league_data import get_players_on_market
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from zoneinfo import ZoneInfo

    players_on_market = get_players_on_market(token, league_id)
    market_df = pd.DataFrame(players_on_market)

    bid_df = pd.merge(today_df_results, market_df, left_on="player_id", right_on="id", how="inner")

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
        bid_df = bid_df[bid_df["predicted_mv_target"] > 5000].sort_values("predicted_mv_target", ascending=False)

    # Startelf-Proba vereinheitlichen
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
        "first_name", "last_name", "team_name", "mv",
        "mv_change_yesterday", "predicted_mv_target", "s_11_prob",
        "hours_to_exp", "expiring_today"
    ]
    for c in keep:
        if c not in bid_df.columns:
            bid_df[c] = np.nan
    bid_df = bid_df[keep]

    print("DEBUG bid_df columns:", list(bid_df.columns))
    return bid_df
