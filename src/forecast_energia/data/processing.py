"""Limpieza de series de 15 min y agregación a frecuencia horaria (formato Nixtla)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_energia import config

PASO = pd.Timedelta(minutes=15)


def completar_grilla(df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    """Reindexa cada serie a una grilla regular: los huecos quedan como NaN explícitos."""
    out = []
    for uid, g in df.groupby("unique_id", sort=True):
        idx = pd.date_range(g["ds"].min(), g["ds"].max(), freq=freq, name="ds")
        s = g.set_index("ds")["y"].reindex(idx)
        out.append(pd.DataFrame({"unique_id": uid, "ds": idx, "y": s.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def _largo_de_racha_nan(y: pd.Series) -> pd.Series:
    """Largo de la racha de NaN consecutivos a la que pertenece cada posición (0 si no es NaN)."""
    es_nan = y.isna()
    grupo = (es_nan != es_nan.shift()).cumsum()
    largo = es_nan.groupby(grupo).transform("sum")
    return largo.where(es_nan, 0).astype(int)


def interpolar_huecos_cortos(df: pd.DataFrame, max_pasos: int = 4) -> pd.DataFrame:
    """Interpola linealmente solo los huecos de hasta `max_pasos` valores (4 × 15 min = 1 h).

    Los huecos más largos quedan como NaN, porque inventarlos sesgaría los modelos. Agrega
    la columna booleana `imputado`.
    """
    partes = []
    for _, g in df.groupby("unique_id", sort=False):
        g = g.sort_values("ds").copy()
        racha = _largo_de_racha_nan(g["y"])
        interp = g["y"].interpolate(method="linear", limit_area="inside")
        rellenar = (racha > 0) & (racha <= max_pasos)
        g["imputado"] = rellenar & interp.notna()
        g["y"] = g["y"].where(~g["imputado"], interp)
        partes.append(g)
    return pd.concat(partes, ignore_index=True)


def a_horario(df: pd.DataFrame, min_validos: int = 3, closed: str = "right") -> pd.DataFrame:
    """Promedia las mediciones de 15 min de cada hora; `ds` marca el inicio de la hora.

    Con `closed="right"`, la hora 00:00 promedia 00:15, 00:30, 00:45 y 01:00, igual que la
    hora 1 del CNDC (hora final). La hora queda NaN si tiene menos de `min_validos` datos.
    """
    partes = []
    for uid, g in df.groupby("unique_id", sort=True):
        s = g.set_index("ds")["y"]
        r = s.resample("h", closed=closed, label="left")
        media = r.mean().where(r.count() >= min_validos)
        partes.append(pd.DataFrame({"unique_id": uid, "ds": media.index, "y": media.to_numpy()}))
    return pd.concat(partes, ignore_index=True)


def escalon_horario(df: pd.DataFrame) -> pd.DataFrame:
    """Valor horario de una serie escalonada (el pronóstico PREV del CNDC).

    El escalón de la hora hh dura de hh:00 a hh:45 o, en algunos meses (nov-2024 a
    mar-2025), de hh:15 a hh+1:00. En ambos casos contiene hh:30, así que se toma ese valor.
    """
    medio = df[df["ds"].dt.minute == 30]
    return pd.DataFrame(
        {
            "unique_id": medio["unique_id"].to_numpy(),
            "ds": medio["ds"].dt.floor("h").to_numpy(),
            "y": medio["y"].to_numpy(),
        }
    )


def construir_sin(dem_h: pd.DataFrame, departamentos: list[str] | None = None) -> pd.DataFrame:
    """Demanda del SIN = suma de departamentos (NaN si falta alguno en esa hora)."""
    departamentos = departamentos or config.DEPARTAMENTO_CODIGOS
    ancho = dem_h[dem_h["unique_id"].isin(departamentos)].pivot(
        index="ds", columns="unique_id", values="y"
    )
    ancho = ancho.reindex(columns=departamentos)
    sin = ancho.sum(axis=1, min_count=len(departamentos))
    return pd.DataFrame({"unique_id": config.SIN_ID, "ds": sin.index, "y": sin.to_numpy()})


def resumen_calidad(df: pd.DataFrame) -> pd.DataFrame:
    """% de faltantes y rango temporal por serie."""
    g = df.groupby("unique_id")
    res = pd.DataFrame(
        {
            "inicio": g["ds"].min(),
            "fin": g["ds"].max(),
            "n": g.size(),
            "pct_faltante": g["y"].apply(lambda y: 100 * y.isna().mean()).round(3),
        }
    )
    if "imputado" in df:
        res["pct_imputado"] = g["imputado"].mean().mul(100).round(3)
    return res


def ratio(a: pd.Series, b: pd.Series) -> float:
    """Cociente de sumas sobre las posiciones donde ambas series tienen dato."""
    m = a.notna() & b.notna()
    return float(np.round(a[m].sum() / b[m].sum(), 4))
