# src/deployment/fastapi_app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '', 'src')))

from deployment.model_loader import load_model

app = FastAPI()

# Carrega o modelo treinado
model = load_model()

# Defina os campos esperados (baseados nos FEATURES_TO_USE do constants.yaml)
class PredictionInput(BaseModel):
    c: float
    l: float
    periodo_num: float
    a_bin: float
    m: float
    p: float
    b: float
    o_transformed: float
    g_agrup_simples_num: float
    e: float
    j_cluster_agrup: float
    f: float
    d: float
    n: float
    i_len: float
    monto: float

@app.post("/predict")
def predict(input: PredictionInput):
    try:
        # Converte a entrada para DataFrame
        data = pd.DataFrame([input.dict()])
        # Gera a previsão (probabilidade de fraude)
        proba = model.predict_proba(data)[:, 1][0]
        return {"fraude_probability": proba}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
