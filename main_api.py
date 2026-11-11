import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -------------------------------------------------------------
# CARGA DEL PAQUETE SERIALIZADO (se hace una sola vez al iniciar)
# -------------------------------------------------------------
RUTA_MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'modelo_scouting_rf.joblib')
paquete = joblib.load(RUTA_MODELO)

modelo = paquete['modelo']
columnas_entrenamiento = paquete['columnas_X']

# -------------------------------------------------------------
# POSICIONES: valor exacto de la base de datos -> nombre en español
# (el modelo se entrenó con el valor en inglés; el español es solo para mostrar)
# -------------------------------------------------------------
POSICIONES = [
    ("Goalkeeper", "Portero"),
    ("Centre-Back", "Defensa central"),
    ("Right-Back", "Lateral derecho"),
    ("Left-Back", "Lateral izquierdo"),
    ("Defensive Midfield", "Pivote (mediocentro defensivo)"),
    ("Central Midfield", "Mediocentro"),
    ("Right Midfield", "Volante derecho"),
    ("Left Midfield", "Volante izquierdo"),
    ("Attacking Midfield", "Mediapunta"),
    ("Right Winger", "Extremo derecho"),
    ("Left Winger", "Extremo izquierdo"),
    ("Second Striker", "Segundo delantero"),
    ("Centre-Forward", "Delantero centro"),
    ("Sin Posicion", "Sin posición"),
]
POSICIONES_VALIDAS = {valor for valor, _ in POSICIONES}

app = FastAPI(title="API de Scouting Deportivo")

# CORS: permite que index.html (cliente) consuma la API desde el navegador
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------
# ESQUEMAS PYDANTIC (validación de entrada y salida en JSON)
# -------------------------------------------------------------
class JugadorEntrada(BaseModel):
    posicion: Optional[str] = None
    edad: float = Field(..., ge=14, le=50)
    estatura_cm: Optional[float] = Field(None, ge=140, le=225)


class PrediccionSalida(BaseModel):
    prediccion: str
    probabilidad_elite: float


# -------------------------------------------------------------
# ENDPOINTS
# -------------------------------------------------------------
@app.get("/posiciones")
def listar_posiciones():
    """Catálogo de posiciones para el menú del frontend (valor + nombre en español)."""
    return [{"valor": valor, "nombre": nombre} for valor, nombre in POSICIONES]


@app.post("/predict", response_model=PrediccionSalida)
def predict(jugador: JugadorEntrada):
    # 1. Validar la posición contra el catálogo con el que se entrenó el modelo
    posicion = jugador.posicion if jugador.posicion else 'Sin Posicion'
    if posicion not in POSICIONES_VALIDAS:
        raise HTTPException(status_code=422, detail=f"Posición no reconocida: '{posicion}'")

    # 2. Aplicar las MISMAS reglas de imputación aprendidas en entrenamiento
    estatura = jugador.estatura_cm
    if estatura is None:
        estatura = paquete['medianas_estatura'].get(posicion)
    if estatura is None or pd.isna(estatura):
        estatura = paquete['mediana_global_estatura']

    # 3. One-Hot Encoding alineado con las columnas vistas por el modelo
    fila = {'edad': jugador.edad, 'estatura_cm': estatura}
    for col in columnas_entrenamiento:
        if col.startswith('posicion_'):
            fila[col] = int(col == f'posicion_{posicion}')

    X_nuevo = pd.DataFrame([fila]).reindex(columns=columnas_entrenamiento, fill_value=0)

    # 4. Generar inferencia y probabilidad
    clase = modelo.predict(X_nuevo)[0]
    probabilidad = modelo.predict_proba(X_nuevo)[0, 1]

    return PrediccionSalida(
        prediccion='Élite' if clase == 1 else 'Estándar',
        probabilidad_elite=float(np.round(probabilidad * 100, 2))
    )