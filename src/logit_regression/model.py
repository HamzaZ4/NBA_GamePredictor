import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.feature_engineering.build_game_specific_matchups import build_matchups


SEASONS = ["2020-21","2021-22","2022-23","2023-24","2024-25"]

def prepare_training_data(season):

    train1 = build_matchups("2020-21")
    train2 = build_matchups("2021-22")
    train3 = build_matchups("2022-23")
    test1 = build_matchups("2023-24")

    feature_cols = ["FG3M_diff", "FGA_diff", "FTA_diff", "BLK_diff",
                    "FG_PCT_diff", "FG3_PCT_diff", "FT_PCT_diff",
                    "AST/TOV_ratio_diff", "STL%_diff", "OREB%_diff", "DRB%_diff"]

    target_col = ["WL"]
    train = pd.concat([train1,train2,train3], ignore_index=True)
    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test = test1[feature_cols]
    y_test = test1[target_col]

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(max_iter=5000))
    ])

    pipe.fit(X_train, y_train.values.ravel())
    return pipe


def predict(pipe, X_test):
    return pipe.predict(X_test)



if __name__ == "__main__":

    feature_cols = ["FG3M_diff", "FGA_diff", "FTA_diff", "BLK_diff",
                    "FG_PCT_diff", "FG3_PCT_diff", "FT_PCT_diff",
                    "AST/TOV_ratio_diff", "STL%_diff", "OREB%_diff", "DRB%_diff"]

    #prepare training data
    X_train,y_train,X_test,y_test = prepare_training_data(SEASONS[0])

    #trained model ready to predict
    pipe = train_model(X_train,y_train)

    #make predictions
    results = predict(pipe,X_test)


    #ANALYSING THE RESULTS

#naive baseline
    home_rate = y_test.mean()
    print(f"Naive baseline (always home):", home_rate)

# Reading logistics
    coefs = pipe.named_steps["logistic_regression"].coef_[0]
    coef_tbl = pd.DataFrame({
        "feature": feature_cols,
        "coef": coefs
    })

# making the multiplied by std column
    coef_tbl["odds_mult_per_1std"] = np.exp(coef_tbl["coef"])

    # this table represents the
    # 1. Coefficients : which are the weights attributed to each feature
    # However these coefficients are
    print(coef_tbl.sort_values("coef", ascending=False).to_string(index=False))

    res = pd.DataFrame(results)
    match = build_matchups("2022-23")

    res["Matchup"] = match[["MATCHUP"]]
    res["ACTUAL"] = y_test

    print(f"Accuracy Score: {accuracy_score(y_test,results)}")
    print(f"Confusion Matrix: \n{confusion_matrix(y_test,results)}")
    print(f"Classification Report: \n{classification_report(y_test,results)}")




