"""Configuración central: rutas, fuentes de datos y metadatos de departamentos."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TZ = "America/La_Paz"  # UTC-4 todo el año, sin horario de verano

# --- CNDC (Comité Nacional de Despacho de Carga) -----------------------------------
CNDC_API_BASE = "https://www.cndc.bo/wp-json/cndc/v1"
# Primer día con datos de 15 min en la API pública (verificado empíricamente).
CNDC_START_DATE = date(2024, 3, 1)
# Valor centinela que usa la API para datos faltantes.
CNDC_MISSING = -1
# Pausa entre requests por worker, para no sobrecargar el servidor del CNDC.
CNDC_REQUEST_PAUSE_S = 0.3
CNDC_MAX_WORKERS = 4
# Los últimos días pueden venir incompletos: se vuelven a descargar en cada refresh.
CNDC_REFETCH_LAST_DAYS = 3

# Endpoints usados. Cada uno devuelve una lista de {"codigo", "fecha", "valores"[96]}.
ENDPOINT_DEMANDA = "rt/demanda"
ENDPOINT_GENERACION = "rt/generacion"

# Series de generación del endpoint rt/generacion.
GENERACION_SERIES = ["TOT", "TERMO", "HIDRO", "SOLAR", "EOL", "BAGAZO", "RENO", "PREV"]
# PREV = pronóstico oficial del CNDC (predespacho), benchmark a batir.
SERIE_PRONOSTICO_CNDC = "PREV"
# Pronóstico del CNDC para Santa Cruz que aparece en rt/demanda (no es demanda real).
SERIE_PRONOSTICO_SCZ = "Prev.SCZ"

# Desde esta fecha Σ demanda departamental = generación TOT (antes ≈ 0.965 × TOT, sin
# pérdidas). Ver processing.ajustar_a_tot.
CNDC_CAMBIO_DEFINICION_DEMANDA = "2026-05-01"

SIN_ID = "SIN"  # Sistema Interconectado Nacional = suma de departamentos


@dataclass(frozen=True)
class Departamento:
    codigo: str  # código usado por la API del CNDC
    capital: str
    lat: float
    lon: float
    subdiv: str  # código de subdivisión en la librería `holidays`


DEPARTAMENTOS: tuple[Departamento, ...] = (
    Departamento("SANTA CRUZ", "Santa Cruz de la Sierra", -17.7833, -63.1821, "S"),
    Departamento("LA PAZ", "La Paz", -16.4955, -68.1336, "L"),
    Departamento("COCHABAMBA", "Cochabamba", -17.3895, -66.1568, "C"),
    Departamento("POTOSI", "Potosí", -19.5836, -65.7531, "P"),
    Departamento("ORURO", "Oruro", -17.9647, -67.1060, "O"),
    Departamento("TARIJA", "Tarija", -21.5355, -64.7296, "T"),
    Departamento("CHUQUISACA", "Sucre", -19.0196, -65.2619, "H"),
    Departamento("BENI", "Trinidad", -14.8333, -64.9000, "B"),
)
DEPARTAMENTO_CODIGOS = [d.codigo for d in DEPARTAMENTOS]

# --- Open-Meteo -----------------------------------------------------------------------
# La Previous Runs API entrega, para cada hora, el valor más reciente y el pronóstico
# emitido N días antes (`<var>_previous_day1`). Esto permite entrenar con el clima que
# realmente se conocía al pronosticar el día siguiente, sin leakage.
OPEN_METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
CLIMA_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "shortwave_radiation",
]
