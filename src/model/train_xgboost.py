import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
from xgboost import XGBClassifier

DATA_PATH = Path("data/features/NIFTY50_final.csv")
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def prepare_features(df):
    drop_cols = ["timestamp", "target", "future_return"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["target"]
    return X, y


if __name__ == "__main__":
    df = load_data()
    X, y = prepare_features(df)

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    print("XGBoost ROC-AUC:", roc_auc_score(y_test, y_proba))

   
    joblib.dump(model, MODEL_PATH / "xgboost_model.pkl")
    print("Saved XGBoost model → models/xgboost_model.pkl")

    
    print("\nProbability summary:")
    print(pd.Series(y_proba).describe())

    print("\nTop 10 probabilities:")
    print(np.sort(y_proba)[-10:])
