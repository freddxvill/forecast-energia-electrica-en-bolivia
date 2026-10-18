"""API de pronósticos (FastAPI): sirve los artefactos precalculados de `artifacts/demo/`.

Uso local:
    uv run --group api uvicorn forecast_energia.api.main:app --reload

Endpoints:
    GET /health                                  estado y fecha del último pronóstico
    GET /unidades                                SIN y los 8 departamentos
    GET /forecast?unidad=SIN&fecha=YYYY-MM-DD    cuantiles horarios (por defecto, mañana)
    GET /backtest?unidad=SIN&desde=&hasta=       real vs pronóstico vs PREV (últimos 90 días)
    GET /metrics                                 métricas de test, seguimiento e impacto

No entrena nada: el pronóstico lo genera `pipelines.pronostico` (cron diario) y la API solo lee
parquet/JSON. Fuentes, en orden:
- `FORECAST_DEMO_URL`: URL base de los artefactos (p.ej. raw.githubusercontent.com del repo),
  releída cada 10 minutos. Así el despliegue no hay que rehacerlo cuando el cron actualiza el
  pronóstico.
- `FORECAST_DEMO_DIR` (por defecto `artifacts/demo/`): respaldo local.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
import urllib.request
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from forecast_energia import config

log = logging.getLogger("forecast_energia.api")

PUNTO = "Pronóstico"
UNIDADES = [config.SIN_ID, *config.DEPARTAMENTO_CODIGOS]
TTL_URL_S = 600

app = FastAPI(
    title="Pronóstico de demanda eléctrica de Bolivia",
    description=(
        "Pronóstico probabilístico day-ahead del SIN y sus 8 departamentos, con intervalos "
        "conformal 80/90/95 %. Datos del CNDC."
    ),
    version="1.0",
)


def demo_dir() -> Path:
    return Path(os.environ.get("FORECAST_DEMO_DIR", config.ROOT / "artifacts" / "demo"))


@lru_cache(maxsize=16)
def _descargar(url: str, ventana: int) -> bytes:  # `ventana` (TTL) invalida la caché
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.read()


def _bytes(archivo: str) -> bytes:
    """Contenido de un artefacto: primero la URL remota (si hay), si no el disco local."""
    base = os.environ.get("FORECAST_DEMO_URL", "").rstrip("/")
    if base:
        try:
            return _descargar(f"{base}/{archivo}", int(time.time() // TTL_URL_S))
        except OSError as e:
            log.warning("No se pudo leer %s/%s (%s): se usa la copia local", base, archivo, e)
    path = demo_dir() / archivo
    if not path.exists():
        raise HTTPException(503, f"Todavía no hay {archivo}: corré `poe pronostico`.")
    return path.read_bytes()


def _tabla(nombre: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(_bytes(f"{nombre}.parquet")))


def _unidad(unidad: str) -> str:
    u = unidad.upper().replace("_", " ")
    if u not in UNIDADES:
        raise HTTPException(404, f"Unidad desconocida: {unidad}. Opciones: {UNIDADES}")
    return u


def _registros(df: pd.DataFrame) -> list[dict]:
    """Filas JSON con nombres cortos: punto, lo_80, hi_80, … (NaN → null)."""
    nombres = {PUNTO: "punto"} | {
        f"{PUNTO}-{lado}-{n}": f"{lado}_{n}" for n in config.NIVELES for lado in ("lo", "hi")
    }
    out = df.astype(object).where(df.notna(), None).rename(columns=nombres)
    out["ds"] = df["ds"].dt.strftime("%Y-%m-%dT%H:%M")
    return out.to_dict("records")


@app.get("/health")
def health() -> dict:
    actual = _tabla("pronostico_actual")
    return {
        "status": "ok",
        "corte": str(actual["corte"].iloc[0]),
        "emitido": str(actual["emitido"].iloc[0]),
        "dias": sorted({str(d.date()) for d in actual["ds"].dt.normalize()}),
    }


@app.get("/unidades")
def unidades() -> list[str]:
    return UNIDADES


@app.get("/forecast")
def forecast(
    unidad: Annotated[str, Query(description="SIN o un departamento (p.ej. SANTA CRUZ)")] = (
        config.SIN_ID
    ),
    fecha: Annotated[date | None, Query(description="Día pronosticado; por defecto el último")] = (
        None
    ),
) -> dict:
    u = _unidad(unidad)
    actual = _tabla("pronostico_actual")
    dias = actual["ds"].dt.normalize()
    dia = pd.Timestamp(fecha) if fecha else dias.max()
    filas = actual[(actual["unique_id"] == u) & (dias == dia)]
    if filas.empty:
        disponibles = sorted({str(d.date()) for d in dias})
        raise HTTPException(404, f"No hay pronóstico para {dia.date()}. Hay: {disponibles}")
    cols = ["ds", PUNTO] + [f"{PUNTO}-{lado}-{n}" for n in config.NIVELES for lado in ("lo", "hi")]
    cols += ["PREV"] if u == config.SIN_ID else []
    return {
        "unidad": u,
        "fecha": str(dia.date()),
        "corte": str(filas["corte"].iloc[0]),
        "emitido": str(filas["emitido"].iloc[0]),
        "unidades": "MW (media horaria; ds = inicio de la hora, hora de Bolivia)",
        "horas": _registros(filas[cols]),
    }


@app.get("/backtest")
def backtest(
    unidad: str = config.SIN_ID,
    desde: date | None = None,
    hasta: date | None = None,
) -> dict:
    u = _unidad(unidad)
    bt = _tabla("backtest_reciente")
    filas = bt[bt["unique_id"] == u]
    if desde:
        filas = filas[filas["ds"] >= pd.Timestamp(desde)]
    if hasta:
        filas = filas[filas["ds"] < pd.Timestamp(hasta) + pd.Timedelta(days=1)]
    cols = ["ds", "y", PUNTO, f"{PUNTO}-lo-90", f"{PUNTO}-hi-90"]
    cols += ["PREV"] if u == config.SIN_ID else []
    return {"unidad": u, "horas": _registros(filas[cols])}


@app.get("/metrics")
def metrics() -> dict:
    return json.loads(_bytes("metricas.json").decode("utf-8"))
