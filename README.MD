# NBA Win Predictor

This project is my hands-on way of learning machine learning with **scikit-learn**, while grounding the work in **basketball theory**.  
The task: predict whether the home team wins a game, using stats engineered around three core ideas that matter in basketball outcomes:

1. **Offensive Efficiency** → how well a team converts possessions into points (e.g. shooting percentages, 3-pointers made).  
2. **Ball Control** → avoiding mistakes and maintaining possession (e.g. turnovers, steals).  
3. **Defensive Efficiency** → limiting opponent scoring and dominating rebounds (e.g. defensive rebounds, opponent PPG).

---

## Data

I collect NBA game data from [nba_api](https://github.com/swar/nba_api) and transform it into "matchups" where features represent **stat differences (home – away)** for each game.  

### Features Used (Grouped by Theory)

**1. Offensive Efficiency**
- `FG3M_diff` → 3-pointers made  
- `FG_PCT_diff` → field goal %  
- `FG3_PCT_diff` → 3-point %  
- `FT_PCT_diff` → free throw %  

**2. Ball Control (Possession / Retaining the Ball)**
- `TOV_diff` → turnovers  
- `STL_diff` → steals  

**3. Defensive Efficiency**
- `OREB_diff` → offensive rebounds (extra chances)  
- `DREB_diff` → defensive rebounds (ending possessions)  
- `PPG_visitor_diff` (planned) → points allowed by defense  

**Target:**  
- `WL` → whether the home team won (`1`) or lost (`0`).

---

## Logistic Regression (First Model)

I started with a Logistic Regression model as a simple baseline.  
It’s interpretable, meaning we can directly see how much each feature impacts the probability of winning.
