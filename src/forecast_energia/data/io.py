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
    """Clima horario por departamento (observado y pronosticado 48 h antes, `*_d2`)."""
    return _leer("clima_horario")


def cargar_precios_mensuales() -> pd.DataFrame:
    """Precios monómicos mensuales del CNDC (USD/MWh; potencia en USD/kW-mes)."""
    return _leer("precios_mensuales")


def cargar_costo_marginal_diario() -> pd.DataFrame:
    """Costo marginal medio diario del CNDC (USD/MWh)."""
    return _leer("costo_marginal_diario")


def a_ancho(df: pd.DataFrame, valores: str = "y") -> pd.DataFrame:
    """Formato largo → ancho: una columna por `unique_id`, índice `ds`."""
    return df.pivot(index="ds", columns="unique_id", values=valores).sort_index()


def serie(df: pd.DataFrame, unique_id: str) -> pd.Series:
    """Extrae una serie como `pd.Series` indexada por `ds`."""
    s = df.loc[df["unique_id"] == unique_id].set_index("ds")["y"].sort_index()
    s.name = unique_id
    return s


def panel_nixtla(df: pd.DataFrame, unique_ids: list[str] | None = None) -> pd.DataFrame:
    """Panel `unique_id, ds, y` sin NaN, como lo exigen los modelos de Nixtla.

    Los pocos huecos que quedan (12 h en todo el periodo) se interpolan en el tiempo. Para
    evaluar se usan los valores observados originales (ver
    `evaluation.reemplazar_y_observado`), así que la interpolación no premia a ningún modelo.
    """
    d = df[["unique_id", "ds", "y"]]
    if unique_ids is not None:
        d = d[d["unique_id"].isin(unique_ids)]
    partes = []
    for uid, g in d.groupby("unique_id", sort=True):
        s = g.set_index("ds")["y"].sort_index()
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h", name="ds"))
        s = s.interpolate(method="time").ffill().bfill()
        partes.append(pd.DataFrame({"unique_id": uid, "ds": s.index, "y": s.to_numpy()}))
    return pd.concat(partes, ignore_index=True)
