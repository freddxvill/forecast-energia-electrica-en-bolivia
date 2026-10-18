"""Variables exógenas futuras (conocidas al momento del corte): calendario y clima pronosticado.

Todas las columnas que agrega `agregar_exogenas` se pueden calcular para el día D en el corte
D−2 23:00:
- El calendario se conoce siempre.
- El clima usa solo `*_d2`, el pronóstico hecho 48 h antes de cada hora. Las corridas
  meteorológicas que lo generan se inician en o antes del corte.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_energia import config
from forecast_energia.data.calendar import features_calendario

# Umbral de temperatura de confort a partir del cual sube el uso de aire acondicionado (°C).
TEMP_CONFORT = 24.0

CALENDARIO = ["hora", "dia_semana", "es_fin_de_semana", "es_feriado", "pre_feriado", "post_feriado"]
CICLICAS = ["hora_sin", "hora_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"]
CLIMA = ["temp", "humedad", "radiacion", "nubosidad", "precipitacion", "cdd"]

_VARIABLES_CLIMA = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "humedad",
    "shortwave_radiation": "radiacion",
    "cloud_cover": "nubosidad",
    "precipitation": "precipitacion",
}


def pesos_departamentales(dem: pd.DataFrame) -> pd.Series:
    """Participación media de cada departamento en el SIN (para ponderar su clima)."""
    deptos = dem[dem["unique_id"].isin(config.DEPARTAMENTO_CODIGOS)]
    media = deptos.groupby("unique_id")["y"].mean()
    return media / media.sum()


def clima_pronosticado(clima: pd.DataFrame, pesos: pd.Series) -> pd.DataFrame:
    """Clima `*_d2` por unique_id: cada departamento con su capital y el SIN ponderado por consumo.

    La columna `cdd` (cooling degree hours, max(T − 24 °C, 0)) captura la no linealidad del
    aire acondicionado. Para el SIN se pondera la CDD de cada capital, no la temperatura media.
    """
    sufijo = config.CLIMA_SUFIJO_PRONOSTICO
    cols = {f"{v}{sufijo}": nombre for v, nombre in _VARIABLES_CLIMA.items()}
    deptos = clima[["unique_id", "ds", *cols]].rename(columns=cols)
    deptos["cdd"] = (deptos["temp"] - TEMP_CONFORT).clip(lower=0)

    w = deptos["unique_id"].map(pesos)
    ponderado = deptos[CLIMA].mul(w, axis=0).assign(ds=deptos["ds"]).groupby("ds").sum()
    sin = ponderado.reset_index().assign(unique_id=config.SIN_ID)
    return pd.concat([deptos, sin[deptos.columns]], ignore_index=True)


def ciclicas(ds: pd.Series) -> pd.DataFrame:
    """Codificación seno/coseno de hora, día de semana y día del año (para redes neuronales)."""
    hora = ds.dt.hour + ds.dt.minute / 60
    return pd.DataFrame(
        {
            "hora_sin": np.sin(2 * np.pi * hora / 24),
            "hora_cos": np.cos(2 * np.pi * hora / 24),
            "dow_sin": np.sin(2 * np.pi * ds.dt.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * ds.dt.dayofweek / 7),
            "doy_sin": np.sin(2 * np.pi * ds.dt.dayofyear / 365.25),
            "doy_cos": np.cos(2 * np.pi * ds.dt.dayofyear / 365.25),
        },
        index=ds.index,
    )


def agregar_exogenas(panel: pd.DataFrame, clima: pd.DataFrame, pesos: pd.Series) -> pd.DataFrame:
    """Agrega al panel `unique_id, ds, y` el calendario, las cíclicas y el clima pronosticado.

    El panel puede extenderse hacia el futuro (filas con `y` NaN) para predecir: las exógenas
    se calculan igual.
    """
    out = panel.copy()
    cal = features_calendario(out["ds"], out["unique_id"])[CALENDARIO]
    out = pd.concat([out, cal, ciclicas(out["ds"])], axis=1)
    out = out.merge(clima_pronosticado(clima, pesos), on=["unique_id", "ds"], how="left")
    faltan = out[CLIMA].isna().any(axis=1)
    if faltan.any():
        # Open-Meteo puede no cubrir las primeras/últimas horas: se completa por cercanía.
        out[CLIMA] = out.groupby("unique_id")[CLIMA].transform(lambda c: c.ffill().bfill())
    return out


def futuro(ultimo_ds: pd.Timestamp, unique_ids: list[str], horas: int) -> pd.DataFrame:
    """Esqueleto `unique_id, ds` de las `horas` siguientes a `ultimo_ds` (para predecir)."""
    ds = pd.date_range(ultimo_ds + pd.Timedelta(hours=1), periods=horas, freq="h")
    return pd.DataFrame([(u, d) for u in unique_ids for d in ds], columns=["unique_id", "ds"])
