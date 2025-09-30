import pandas as pd
from src.data_collection.fetch_match_stats import fetch_matchup_stats

SEASONS = ["2019-20","2020-21","2021-22","2022-23","2023-24","2024-25"]
STAT_COLS = ["PTS","FGM","FG3M", "FGA","FTA", "OREB","DREB", "AST", "TOV", "BLK", "STL", "FG_PCT", "FG3_PCT", "FT_PCT"]

def build_matchups(season):
    matchups = fetch_matchup_stats(season)

    games = compute_rolling_features(matchups)

    home = games[games['MATCHUP'].str.contains("vs.")].copy()
    away = games[games['MATCHUP'].str.contains("@")].copy()

    home = home.add_suffix("_home")
    away = away.add_suffix("_visitor")

    games = pd.merge(home, away, left_on="GAME_ID_home", right_on="GAME_ID_visitor")

    games = games.rename(columns={"SEASON_ID_home": "SEASON_ID", "WL_home": "WL"})

    games["WL"] = games["WL"].map({"W": 1, "L": 0})

    clean_games = clean_features(games, 10)

    featured_games = create_features(clean_games, 10)

    return featured_games

def compute_rolling_features(matchups, window = 10):

    games = matchups.sort_values(["TEAM_ID", "GAME_DATE"])

    for col in STAT_COLS:
        games[f"{col}_last{window}"] = (
            games.groupby('TEAM_ID')[col].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    return games

def clean_features(games, window):
    keep_cols = ["MATCHUP_home", "SEASON_ID", "WL", "TEAM_ABBREVIATION_home","TEAM_ABBREVIATION_visitor", "TEAM_ID_home", "TEAM_ID_visitor", "GAME_ID_home","GAME_DATE_home"]


    rolling_cols = [col for col in games.columns if f"_last{window}" in col]
    columns_filtered_games = games[keep_cols + rolling_cols]
    columns_filtered_games = columns_filtered_games.rename(
        columns={"GAME_ID_home": "GAME_ID", "GAME_DATE_home": "GAME_DATE", "MATCHUP_home" : "MATCHUP"}
    )

    return columns_filtered_games

def create_features(games, window):

    games["POS_home"] = 0.96*(games[f"FGA_last{window}_home"] - games[f"OREB_last{window}_home"]
    + games[f"TOV_last{window}_home"] + 0.44 * games[f"FTA_last{window}_home"])

    games["POS_visitor"] = 0.96*(games[f"FGA_last{window}_visitor"] - games[f"OREB_last{window}_visitor"]
    + games[f"TOV_last{window}_visitor"] + 0.44 * games[f"FTA_last{window}_visitor"])

    games["AST/TOV_ratio_diff"] = (games[f"AST_last{window}_home"] / (games[f"TOV_last{window}_home"] + 1e-6)
                                   - games[f"AST_last{window}_visitor"] / (games[f"TOV_last{window}_visitor"] + 1e-6))



    # see here for this formula : https://www.nbastuffer.com/analytics101/possession/
    games["STL%_diff"] = games[f"STL_last{window}_home"] / (games["POS_visitor"] + 1e-6) - games[f"STL_last{window}_visitor"] / (games["POS_home"] + 1e-6)

    games["OREB%_diff"] = games[f"OREB_last{window}_home"] / (games[f"FGA_last{window}_visitor"] + 1e-6) - games[f"OREB_last{window}_visitor"] / (games[f"FGA_last{window}_home"] + 1e-6)

    games["DRB%_diff"] = games[f"DREB_last{window}_home"] / (games[f"FGA_last{window}_visitor"] + 1e-6) - games[f"DREB_last{window}_visitor"] / (games[f"FGA_last{window}_home"] + 1e-6)

    for col in STAT_COLS:
        if col in ["AST","DREB","OREB","TOV","STL"]:
            continue
        games[f"{col}_diff"] = games[f"{col}_last{window}_home"] - games[f"{col}_last{window}_visitor"]
        games = games.drop(columns=[f"{col}_last{window}_home", f"{col}_last{window}_visitor"])

    drop_cols = [
        f"AST_last{window}_home", f"AST_last{window}_visitor",
        f"TOV_last{window}_home", f"TOV_last{window}_visitor",
        f"STL_last{window}_home", f"STL_last{window}_visitor",
        f"OREB_last{window}_home", f"OREB_last{window}_visitor",
        f"DREB_last{window}_home", f"DREB_last{window}_visitor",
    ]

    games = games.drop(columns=drop_cols)
    games = games.dropna()
    return games

if __name__ == "__main__":
    build_matchups("2021-22")

