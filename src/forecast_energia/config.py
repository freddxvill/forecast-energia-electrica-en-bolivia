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

# Precios (para valorizar el impacto, Fase 7). `dashboard/precios` devuelve los precios
# monómicos de los 12 meses que terminan en el mes pedido; `cm-diario`, el costo marginal medio
# diario de los últimos N días (solo hay ~200 días publicados, con fechas "dd/mm" sin año).
ENDPOINT_PRECIOS = "dashboard/precios"
ENDPOINT_CM_DIARIO = "cm-diario"

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
# La Previous Runs API entrega, para cada hora, el valor más reciente y el pronóstico hecho
# N días antes de esa hora (`<var>_previous_dayN`, lead fijo de N × 24 h). Para el día D el
# corte es D−2 23:00, así que solo `_previous_day2` (corridas iniciadas ≤ el corte) está libre
# de leakage: `_previous_day1` usaría corridas del día D−1, posteriores al corte.
CLIMA_DIAS_ANTELACION = 2
CLIMA_SUFIJO_PRONOSTICO = f"_d{CLIMA_DIAS_ANTELACION}"
OPEN_METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
CLIMA_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "shortwave_radiation",
]

# --- Protocolo de evaluación (Fase 3) ----------------------------------------------------
# Pronóstico day-ahead comparable con el predespacho del CNDC, que se publica con 1–2 días de
# anticipación: para pronosticar el día D solo se usan datos hasta las 23:00 de D−2 (el
# "corte"). El modelo pronostica 48 h desde el corte y se evalúan las horas 25–48, que son
# las del día D.
HORIZONTE_H = 48
HORAS_DIA_OBJETIVO = (25, 48)  # horizontes (inclusive) que caen en el día D
# Fechas fijas para que los resultados sean reproducibles aunque lleguen datos nuevos.
CALIB_INICIO = "2026-01-16"  # ~3 meses de calibración conformal
TEST_INICIO = "2026-04-17"  # 6 meses de test (183 días)
# Último día del test reportado. Los días posteriores que lleguen con `poe data` forman el
# periodo "prospectivo": datos que nadie miró al diseñar los modelos (validación honesta).
TEST_FIN = "2026-10-16"
NIVELES = (80, 90, 95)  # niveles de los intervalos de predicción (%)
ESTACIONALIDAD_MASE = 168  # naive semanal como escala de MASE

# --- Impacto de negocio (Fase 7) ---------------------------------------------------------
# Supuestos para valorizar los desvíos evitados. Los precios del mercado (energía, potencia,
# costo marginal) salen de la API del CNDC; lo de abajo no está en la API y se documenta con
# su fuente. Cada valor tiene un rango (bajo/central/alto) para el análisis de sensibilidad.
SUPUESTOS_IMPACTO: dict = {
    "heat_rate_mmbtu_mwh": {
        "bajo": 9.0,
        "central": 10.5,
        "alto": 12.0,
        "descripcion": "Consumo específico de la unidad térmica que ajusta el desvío",
        "fuente": (
            "Rango típico de turbinas a gas de ciclo abierto (el parque marginal del SIN). "
            "Chequeo con datos del CNDC: costo marginal mediano / precio regulado del gas "
            "(lo calcula pipelines.impacto)."
        ),
    },
    "precio_gas_oportunidad_usd_mmbtu": {
        "bajo": 5.0,
        "central": 6.5,
        "alto": 8.0,
        "descripcion": "Costo de oportunidad del gas: precio de exportación a Brasil",
        "fuente": (
            "Contratos de YPFB con compradores brasileños indexados al Brent, ~6–7 USD/MMBtu "
            "(OilPrice.com, 2025); históricamente 4–6 USD/MMBtu."
        ),
    },
    "precio_gas_regulado_usd_mpc": {
        "central": 1.30,
        "descripcion": "Precio regulado del gas para generación termoeléctrica",
        "fuente": "DS 26037 (2000), vigente; ~1.3 USD/MMBtu (1 MPC ≈ 1.03 MMBtu).",
    },
    "mmbtu_por_mpc": {
        "central": 1.03,
        "descripcion": "Poder calorífico del gas natural boliviano",
        "fuente": "Valor típico del gas natural seco (~1 030 BTU/pie³).",
    },
    "nivel_reserva": {
        "central": 95,
        "descripcion": "% de los desvíos al alza (demanda > pronóstico) que cubre la reserva",
        "fuente": "Criterio del análisis (cuantil 95 unilateral = límite superior del IC 90 %).",
    },
}
