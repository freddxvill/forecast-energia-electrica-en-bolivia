"""Lectura de los datasets procesados (`data/processed`)."""

from __future__ import annotations

import pandas as pd

from forecast_energia import config


def _leer(nombre: str) -> pd.DataFrame:
    path = config.PROCESSED_DIR / f"{nombre}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} no existe: corré `uv run poe data` primero.")
    return pd.read_parquet(path)


def cargar_demanda() -> pd.DataFrame:
    """Demanda horaria (MW) de los 8 departamentos + SIN, formato largo."""
    return _leer("demanda_horaria")


def cargar_generacion() -> pd.DataFrame:
    """Generación horaria (MW) por tecnología, `TOT` y el pronóstico oficial `PREV`."""
    return _leer("generacion_horaria")


def cargar_clima() -> pd.DataFrame:
    """Clima horario por departamento (observado y pronosticado el día anterior `*_d1`)."""
    return _leer("clima_horario")


def a_ancho(df: pd.DataFrame, valores: str = "y") -> pd.DataFrame:
    """Formato largo → ancho: una columna por `unique_id`, índice `ds`."""
    return df.pivot(index="ds", columns="unique_id", values=valores).sort_index()


def serie(df: pd.DataFrame, unique_id: str) -> pd.Series:
    """Extrae una serie como `pd.Series` indexada por `ds`."""
    s = df.loc[df["unique_id"] == unique_id].set_index("ds")["y"].sort_index()
    s.name = unique_id
    return s
