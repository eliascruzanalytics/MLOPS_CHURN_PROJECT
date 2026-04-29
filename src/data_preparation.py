import pandas as pd

def load_data(path="data/churn.csv"):
    df = pd.read_csv(path)

    # Exemplo simples
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]

    X = pd.get_dummies(X)

    return X, y
