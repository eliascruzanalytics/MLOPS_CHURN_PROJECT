import pandas as pd
import numpy as np
import os

np.random.seed(42)

n = 2000

# Variáveis base
df = pd.DataFrame({
    "customerID": [f"CUST{i}" for i in range(n)],
    "tenure": np.random.randint(1, 72, n),
    "MonthlyCharges": np.random.uniform(20, 150, n),
})

# TotalCharges correlacionado com tenure
df["TotalCharges"] = df["tenure"] * df["MonthlyCharges"] + np.random.normal(0, 200, n)

# Variáveis categóricas
df["Contract"] = np.random.choice(
    ["Month-to-month", "One year", "Two year"],
    p=[0.6, 0.25, 0.15],
    size=n
)

df["InternetService"] = np.random.choice(
    ["DSL", "Fiber optic", "No"],
    p=[0.4, 0.4, 0.2],
    size=n
)

df["PaymentMethod"] = np.random.choice(
    ["Electronic check", "Credit card", "Bank transfer"],
    size=n
)

# Probabilidade de churn (mais realista)
prob = (
    0.4 * (df["tenure"] < 12).astype(int) +
    0.3 * (df["Contract"] == "Month-to-month").astype(int) +
    0.2 * (df["InternetService"] == "Fiber optic").astype(int) +
    0.1 * (df["MonthlyCharges"] > 100).astype(int)
)

# adicionar ruído
prob = prob + np.random.normal(0, 0.1, n)

# normalizar entre 0 e 1
prob = 1 / (1 + np.exp(-prob))

# gerar churn probabilístico
df["Churn"] = np.where(np.random.rand(n) < prob, "Yes", "No")

# salvar
os.makedirs("data", exist_ok=True)
df.to_csv("data/churn.csv", index=False)

print("Dataset realista gerado com sucesso!")
