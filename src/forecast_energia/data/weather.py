"""Clima horario de las capitales departamentales desde la Previous Runs API de Open-Meteo.

Para cada variable se guardan dos versiones:
- `<var>`: el mejor valor disponible (análisis o pronóstico más reciente). Sirve para el EDA.
- `<var>_d2`: el pronóstico hecho 48 h antes de esa hora. Es la versión **sin leakage** que
  se usa como exógena futura (ver `config.CLIMA_DIAS_ANTELACION`).
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from forecast_energia import config

log = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
def _fetch(lat: float, lon: float, start: date, end: date) -> dict:
    n = config.CLIMA_DIAS_ANTELACION
    hourly = config.CLIMA_VARIABLES + [f"{v}_previous_day{n}" for v in config.CLIMA_VARIABLES]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": config.TZ,
    }
    resp = requests.get(config.OPEN_METEO_PREVIOUS_RUNS_URL, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def parse_respuesta(data: dict, unique_id: str) -> pd.DataFrame:
    """Convierte la respuesta de Open-Meteo a un DataFrame con una fila por hora."""
    df = pd.DataFrame(data["hourly"])
    df["ds"] = pd.to_datetime(df.pop("time"))
    n, sufijo = config.CLIMA_DIAS_ANTELACION, config.CLIMA_SUFIJO_PRONOSTICO
    df = df.rename(columns={f"{v}_previous_day{n}": f"{v}{sufijo}" for v in config.CLIMA_VARIABLES})
    df.insert(0, "unique_id", unique_id)
    return df


def descargar_clima(start: date, end: date) -> pd.DataFrame:
    """Clima horario (hora local de Bolivia) de las 8 capitales, una fila por depto y hora."""
    frames = []
    for dep in config.DEPARTAMENTOS:
        log.info("Clima %s (%s)", dep.codigo, dep.capital)
        frames.append(parse_respuesta(_fetch(dep.lat, dep.lon, start, end), dep.codigo))
    return pd.concat(frames, ignore_index=True)
