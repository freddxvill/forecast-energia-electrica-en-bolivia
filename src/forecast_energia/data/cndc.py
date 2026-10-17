"""Cliente de la API pública del CNDC con caché mensual en Parquet.

Cada endpoint diario (`rt/demanda`, `rt/generacion`) devuelve, por serie, 96 valores
instantáneos de potencia (MW) registrados por SCADA cada 15 minutos. El valor `i`
corresponde a `fecha + (i + 1) * 15 min`, es decir, de 00:15 a 24:00. Lo validamos
contra la curva horaria oficial de `rt/curva-dia`: el índice 3 coincide con la hora 01:00.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from forecast_energia import config

log = logging.getLogger(__name__)

PASO = pd.Timedelta(minutes=15)
VALORES_POR_DIA = 96
COLUMNAS = ["unique_id", "ds", "y"]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "forecast-energia (proyecto de ciencia de datos; uso académico)"
    return s


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20), reraise=True)
def fetch_dia(session: requests.Session, endpoint: str, fecha: date) -> list[dict]:
    """Descarga el JSON crudo de un endpoint para una fecha. Lista vacía si no hay datos."""
    url = f"{config.CNDC_API_BASE}/{endpoint}"
    resp = session.get(url, params={"fecha": fecha.isoformat()}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"Respuesta inesperada de {endpoint} para {fecha}: {str(data)[:200]}")
    return data


def parse_dia(raw: list[dict], fecha: date) -> pd.DataFrame:
    """Convierte la respuesta de un día al formato largo `unique_id, ds, y` (MW).

    El centinela `-1` se convierte en NaN.
    """
    inicio = pd.Timestamp(fecha)
    ds = inicio + PASO * np.arange(1, VALORES_POR_DIA + 1)
    frames = []
    for serie in raw:
        valores = np.asarray(serie["valores"], dtype="float64")
        if len(valores) != VALORES_POR_DIA:
            log.warning("%s %s: %d valores (se esperaban 96)", serie["codigo"], fecha, len(valores))
            valores = np.resize(np.append(valores, [np.nan] * VALORES_POR_DIA), VALORES_POR_DIA)
        valores[valores == config.CNDC_MISSING] = np.nan
        frames.append(pd.DataFrame({"unique_id": serie["codigo"], "ds": ds, "y": valores}))
    if not frames:
        return pd.DataFrame(
            {
                c: pd.Series(dtype=t)
                for c, t in zip(COLUMNAS, ["str", "datetime64[ns]", "float64"], strict=True)
            }
        )
    return pd.concat(frames, ignore_index=True)


def _cache_dir(endpoint: str) -> Path:
    return config.RAW_DIR / endpoint.replace("/", "_")


def _cache_path(endpoint: str, mes: str) -> Path:
    return _cache_dir(endpoint) / f"{mes}.parquet"


def leer_cache(endpoint: str) -> pd.DataFrame:
    """Lee todos los meses cacheados de un endpoint."""
    archivos = sorted(_cache_dir(endpoint).glob("*.parquet"))
    if not archivos:
        return pd.DataFrame(columns=COLUMNAS)
    return pd.concat([pd.read_parquet(a) for a in archivos], ignore_index=True)


def _dias_cacheados(endpoint: str) -> set[date]:
    df = leer_cache(endpoint)
    if df.empty:
        return set()
    # Un valor a las 24:00 pertenece al día anterior.
    dias = (pd.to_datetime(df["ds"]) - PASO).dt.date
    return set(dias.unique())


def _rango(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def descargar(endpoint: str, start: date, end: date) -> pd.DataFrame:
    """Descarga los días faltantes de `[start, end]` y actualiza la caché mensual.

    Siempre se vuelven a descargar los últimos `CNDC_REFETCH_LAST_DAYS` días, porque el
    CNDC publica los datos más recientes incompletos.
    """
    cacheados = _dias_cacheados(endpoint)
    recientes = {end - timedelta(days=i) for i in range(config.CNDC_REFETCH_LAST_DAYS)}
    pendientes = [d for d in _rango(start, end) if d not in cacheados or d in recientes]
    log.info("%s: %d días en caché, %d por descargar", endpoint, len(cacheados), len(pendientes))

    nuevos: list[pd.DataFrame] = []
    sin_datos: list[date] = []
    session = _session()

    def tarea(fecha: date) -> tuple[date, pd.DataFrame]:
        raw = fetch_dia(session, endpoint, fecha)
        time.sleep(config.CNDC_REQUEST_PAUSE_S)
        return fecha, parse_dia(raw, fecha)

    with ThreadPoolExecutor(max_workers=config.CNDC_MAX_WORKERS) as pool:
        futuros = [pool.submit(tarea, d) for d in pendientes]
        for i, fut in enumerate(as_completed(futuros), 1):
            fecha, df = fut.result()
            if df.empty:
                sin_datos.append(fecha)
            else:
                nuevos.append(df)
            if i % 50 == 0 or i == len(futuros):
                log.info("%s: %d/%d días descargados", endpoint, i, len(futuros))

    if sin_datos:
        log.warning(
            "%s: %d días sin datos (p.ej. %s)", endpoint, len(sin_datos), sorted(sin_datos)[:5]
        )
    if nuevos:
        _escribir_cache(endpoint, pd.concat(nuevos, ignore_index=True))
    return leer_cache(endpoint)


def _escribir_cache(endpoint: str, nuevos: pd.DataFrame) -> None:
    """Combina con la caché existente (los datos nuevos tienen prioridad) y guarda por mes."""
    _cache_dir(endpoint).mkdir(parents=True, exist_ok=True)
    nuevos = nuevos.assign(mes=(nuevos["ds"] - PASO).dt.strftime("%Y-%m"))
    for mes, grupo in nuevos.groupby("mes"):
        path = _cache_path(endpoint, mes)
        grupo = grupo.drop(columns="mes")
        if path.exists():
            grupo = pd.concat([pd.read_parquet(path), grupo], ignore_index=True)
        grupo = (
            grupo.drop_duplicates(subset=["unique_id", "ds"], keep="last")
            .sort_values(["unique_id", "ds"])
            .reset_index(drop=True)
        )
        grupo.to_parquet(path, index=False)
