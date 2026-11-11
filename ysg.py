import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

profiles = {
    "Player A": (0.52, 0.39, 8, 5),
    "Player B": (0.48, 0.43, 4, 8),
    "Player C": (0.55, 0.32, 11, 3),
    "Player D": (0.45, 0.36, 6, 9),
    "Player E": (0.50, 0.41, 7, 6),
    "Player F": (0.44, 0.34, 12, 4),
    "Player G": (0.53, 0.45, 5, 5),
    "Player H": (0.47, 0.37, 9, 7),
    "Player I": (0.51, 0.30, 10, 4),
    "Player J": (0.46, 0.42, 3, 10),
    "Player K": (0.49, 0.38, 8, 8),
    "Player L": (0.54, 0.35, 6, 5)
}

records = []

for player, (scoring, three, reb, ast) in profiles.items():
    for match in range(1, 41):
        minutes = np.clip(rng.normal(32, 4), 18, 42)
        fga = rng.poisson(minutes * 0.55)
        three_pa = rng.binomial(fga, 0.38)
        two_pa = fga - three_pa
        three_pm = rng.binomial(three_pa, three)
        two_pm = rng.binomial(two_pa, scoring)
        fgm = two_pm + three_pm
        fta = rng.poisson(minutes * 0.17)
        ftm = rng.binomial(fta, 0.78)

        records.append({
            "Player": player,
            "Match": match,
            "Minutes": round(minutes, 1),
            "Points": 2 * two_pm + 3 * three_pm + ftm,
            "FGM": fgm,
            "FGA": fga,
            "3PM": three_pm,
            "3PA": three_pa,
            "Rebounds": rng.poisson(reb * minutes / 36),
            "Assists": rng.poisson(ast * minutes / 36),
            "Turnovers": rng.poisson(minutes * 0.09),
            "Steals": rng.poisson(minutes * 0.045),
            "Blocks": rng.poisson(minutes * 0.035)
        })

df = pd.DataFrame(records)

df["EFG"] = np.where(
    df["FGA"] > 0,
    (df["FGM"] + 0.5 * df["3PM"]) / df["FGA"] * 100,
    0
)

df["TS"] = np.where(
    (df["FGA"] + 0.44 * df.get("FTA", 0)) > 0,
    df["Points"] / (2 * (df["FGA"] + 0.44 * df.get("FTA", 0))) * 100,
    0
)

summary = df.groupby("Player").agg(
    Games=("Match", "count"),
    Points=("Points", "mean"),
    Peak=("Points", "max"),
    Rebounds=("Rebounds", "mean"),
    Assists=("Assists", "mean"),
    Efficiency=("EFG", "mean"),
    Turnovers=("Turnovers", "mean"),
    Consistency=("Points", "std")
)

summary["CV"] = summary["Consistency"] / summary["Points"]
summary["Peak_Gap"] = summary["Peak"] - summary["Points"]

def percentile(series, higher=True):
    return series.rank(pct=True, ascending=higher) * 100

summary["Rating"] = (
    percentile(summary["Points"]) * 0.30
    + percentile(summary["Efficiency"]) * 0.25
    + percentile(summary["Assists"]) * 0.15
    + percentile(summary["Rebounds"]) * 0.15
    + percentile(summary["CV"], False) * 0.15
)

summary = summary.sort_values("Rating", ascending=False)
summary["Rank"] = np.arange(1, len(summary) + 1)

trends = df.groupby("Player").apply(
    lambda x: x.sort_values("Match").tail(5)["Points"].mean()
    - x.sort_values("Match").iloc[-10:-5]["Points"].mean()
)

def compare_players(p1, p2):
    if p1 not in summary.index or p2 not in summary.index:
        print("Player not found")
        return
    print(summary.loc[[p1, p2]].round(2))

def scouting_report(player):
    if player not in summary.index:
        print("Player not found")
        return
    print(summary.loc[player].round(2))
    print("Recent trend:", trends[player])

def find_players(points=15, assists=4, rating=50):
    return summary[
        (summary["Points"] >= points)
        & (summary["Assists"] >= assists)
        & (summary["Rating"] >= rating)
    ].round(2)

def match_history(player):
    if player not in df["Player"].values:
        print("Player not found")
        return
    return df[df["Player"] == player].tail(10)

print("Match records:", len(df))
print(summary.round(2))

compare_players("Player A", "Player B")
scouting_report("Player A")
print(find_players())
print(match_history("Player A"))

df.to_csv("statscout_match_data.csv", index=False)
summary.to_csv("statscout_rankings.csv")
