"""Métricas de error puntual (las probabilísticas se agregan en la Fase 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _alinear(y: pd.Series, yhat: pd.Series) -> tuple[pd.Series, pd.Series]:
    m = y.notna() & yhat.notna()
    return y[m], yhat[m]


def metricas_puntuales(y: pd.Series, yhat: pd.Series) -> dict[str, float]:
    """MAE, RMSE, MAPE (%), sesgo medio (yhat − y) y n, sobre las horas con ambos datos."""
    y, yhat = _alinear(y, yhat)
    e = yhat - y
    return {
        "MAE": float(e.abs().mean()),
        "RMSE": float(np.sqrt((e**2).mean())),
        "MAPE": float(100 * (e.abs() / y).mean()),
        "sesgo": float(e.mean()),
        "n": int(len(e)),
    }


def tabla_metricas(y: pd.Series, pronosticos: dict[str, pd.Series]) -> pd.DataFrame:
    """Una fila por pronóstico, evaluados sobre las mismas horas (intersección)."""
    comunes = y.notna()
    for p in pronosticos.values():
        comunes &= p.reindex(y.index).notna()
    filas = {
        nombre: metricas_puntuales(y[comunes], p.reindex(y.index)[comunes])
        for nombre, p in pronosticos.items()
    }
    return pd.DataFrame(filas).T


def error_por_grupo(y: pd.Series, yhat: pd.Series, grupo: pd.Series) -> pd.DataFrame:
    """MAE, MAPE y sesgo agregados por un grupo (p.ej. hora del día o mes)."""
    y, yhat = _alinear(y, yhat.reindex(y.index))
    e = yhat - y
    df = pd.DataFrame(
        {"ae": e.abs(), "ape": 100 * e.abs() / y, "e": e, "g": grupo.reindex(y.index)}
    )
    return df.groupby("g").agg(MAE=("ae", "mean"), MAPE=("ape", "mean"), sesgo=("e", "mean"))
