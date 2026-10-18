"""Variables de calendario y feriados de Bolivia (nacionales y departamentales)."""

from __future__ import annotations

import holidays
import pandas as pd

from forecast_energia import config

# Días cívicos que la librería `holidays` lista como feriados pero no suspenden actividades: el 17
# de octubre la demanda del SIN no cae (−1.0 % en 2024, +3.7 % en 2025, contra −7 % a −25 % en los
# feriados reales). Marcarlo como feriado hacía que los modelos pronosticaran una caída inexistente.
DIAS_CIVICOS = ("Día de la Dignidad Nacional",)


def feriados(anios: list[int], subdiv: str | None = None) -> holidays.HolidayBase:
    """Feriados de Bolivia; con `subdiv` se agrega el feriado departamental."""
    cal = holidays.Bolivia(years=anios, subdiv=subdiv)
    for nombre in DIAS_CIVICOS:
        for fecha in [f for f, n in cal.items() if n == nombre]:
            del cal[fecha]
    return cal


def _fechas(cal: holidays.HolidayBase) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(cal.keys()))


def features_calendario(ds: pd.Series, unique_id: pd.Series | None = None) -> pd.DataFrame:
    """Variables de calendario conocidas a futuro, para un vector de timestamps horarios.

    Si se pasa `unique_id` (código de departamento), `es_feriado` también incluye el
    feriado departamental. Para el SIN solo cuentan los feriados nacionales.
    """
    ds = pd.to_datetime(ds)
    anios = sorted({*ds.dt.year, ds.dt.year.max() + 1})
    dias = ds.dt.normalize()
    un_dia = pd.Timedelta(days=1)

    nacional = _fechas(feriados(anios))
    es_feriado = dias.isin(nacional)
    if unique_id is not None:
        for dep in config.DEPARTAMENTOS:
            propios = _fechas(feriados(anios, dep.subdiv)).difference(nacional)
            es_feriado |= (unique_id.to_numpy() == dep.codigo) & dias.isin(propios)

    ayer = (dias - un_dia).isin(nacional)
    manana = (dias + un_dia).isin(nacional)
    return pd.DataFrame(
        {
            "hora": ds.dt.hour.astype("int8"),
            "dia_semana": ds.dt.dayofweek.astype("int8"),
            "mes": ds.dt.month.astype("int8"),
            "dia_anio": ds.dt.dayofyear.astype("int16"),
            "es_fin_de_semana": (ds.dt.dayofweek >= 5).astype("int8"),
            "es_feriado": es_feriado.astype("int8"),
            "post_feriado": ayer.astype("int8"),
            "pre_feriado": manana.astype("int8"),
        },
        index=ds.index,
    )
