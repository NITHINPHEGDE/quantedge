import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import joblib

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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, (y_proba > 0.5).astype(int)))

    
    joblib.dump(model, MODEL_PATH / "logreg_model.pkl")
    joblib.dump(scaler, MODEL_PATH / "scaler.pkl")

    print("Saved baseline model & scaler")
