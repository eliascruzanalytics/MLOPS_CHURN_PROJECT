from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# Sempre pega o modelo mais recente
model = mlflow.pyfunc.load_model("models:/churn_model/latest")


# Validação de entrada
class ChurnInput(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str


@app.get("/")
def home():
    return {"status": "API online"}


@app.post("/predict")
def predict(data: ChurnInput):
    try:
        df = pd.DataFrame([data.dict()])
        prediction = model.predict(df)

        return {
            "prediction": int(prediction[0])
        }

    except Exception as e:
        return {
            "error": str(e)
        }
