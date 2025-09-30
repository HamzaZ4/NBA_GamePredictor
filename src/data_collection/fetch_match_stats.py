from nba_api.stats.endpoints import leaguegamefinder


SEASONS = ["2019-20","2020-21","2021-22","2022-23","2023-24","2024-25"]

def fetch_matchup_stats(season):

    matchups = leaguegamefinder.LeagueGameFinder(
        player_or_team_abbreviation="T",
        season_nullable = season,
        league_id_nullable="00",
        season_type_nullable="Regular Season"
    ).get_data_frames()[0]

    matchups = matchups[[
        "SEASON_ID", "GAME_DATE", "GAME_ID",
        "TEAM_ID", "TEAM_ABBREVIATION", "MATCHUP", "WL",
        "PTS", "AST", "TOV", "STL", "BLK", "FG_PCT","FTA", "FG3_PCT", "FT_PCT","FGM","FG3M", "FGA", "OREB","DREB"
    ]]

    return matchups

if __name__ == "__main__":
    for season in SEASONS:
        fetch_matchup_stats(season)