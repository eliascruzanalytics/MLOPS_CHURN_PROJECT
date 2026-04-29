import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

from data_preparation import load_data

mlflow.set_experiment("churn-advanced")

X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# parâmetros de negócio
ticket_medio = 100
custo_acao = 20

def calcular_metricas(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # impacto financeiro
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()

    ganho = tp * ticket_medio
    custo = (tp + fp) * custo_acao

    lucro = ganho - custo

    return precision, recall, f1, lucro


modelos = {
    "RandomForest": RandomForestClassifier(n_estimators=200),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

melhor_modelo = None
melhor_lucro = -np.inf

for nome, modelo in modelos.items():

    with mlflow.start_run(run_name=nome):

        modelo.fit(X_train, y_train)

        # probabilidade ao invés de classe
        y_prob = modelo.predict_proba(X_test)[:, 1]

        melhor_threshold = 0
        melhor_resultado = None

        # testar múltiplos thresholds
        for threshold in np.arange(0.3, 0.8, 0.05):

            precision, recall, f1, lucro = calcular_metricas(
                y_test, y_prob, threshold
            )

            if melhor_resultado is None or lucro > melhor_resultado["lucro"]:
                melhor_resultado = {
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "lucro": lucro
                }

        # log no MLflow
        mlflow.log_param("model", nome)
        mlflow.log_param("best_threshold", melhor_resultado["threshold"])

        mlflow.log_metric("precision", melhor_resultado["precision"])
        mlflow.log_metric("recall", melhor_resultado["recall"])
        mlflow.log_metric("f1_score", melhor_resultado["f1"])
        mlflow.log_metric("lucro", melhor_resultado["lucro"])

        mlflow.sklearn.log_model(
            modelo,
            "model",
            registered_model_name="churn_model"
        )

        print(f"\nModelo: {nome}")
        print(f"Threshold: {melhor_resultado['threshold']}")
        print(f"Precision: {melhor_resultado['precision']}")
        print(f"Recall: {melhor_resultado['recall']}")
        print(f"Lucro: {melhor_resultado['lucro']}")

        # guardar melhor modelo global
        if melhor_resultado["lucro"] > melhor_lucro:
            melhor_lucro = melhor_resultado["lucro"]
            melhor_modelo = nome

print(f"\n Melhor modelo baseado em lucro: {melhor_modelo}")
