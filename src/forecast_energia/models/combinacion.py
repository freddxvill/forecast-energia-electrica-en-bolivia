"""Corrección de sesgo por hora y combinación con el pronóstico oficial del CNDC (PREV).

Hallazgo de la Fase 5: buena parte del error de PREV es un **sesgo sistemático por hora**
(subestima la rampa de las 18 h y sobrestima de noche). Restarle la mediana de su error en
cada hora, calculada con los últimos 91 días conocidos en el corte (hasta D−2), baja su MAE en
test de 54.8 a 42.7 MW. Promediarlo con el ensamble propio, que usa otra información (clima,
lags), da el mejor pronóstico. Se validó en la parte final de la calibración y en el test.
"""

from __future__ import annotations

import pandas as pd

from forecast_energia.conformal import RETRASO_DIAS

NOMBRE = "Combinado"
VENTANA_DIAS = 91
MIN_DIAS = 30


def corregir_sesgo_por_hora(
    cv: pd.DataFrame,
    modelo: str,
    nombre: str | None = None,
    ventana_dias: int = VENTANA_DIAS,
    min_dias: int = MIN_DIAS,
) -> pd.DataFrame:
    """ŷ + mediana(y − ŷ) de la misma hora en los `ventana_dias` días hasta D−2.

    El backtest tiene una fila por día y hora, así que dentro de cada hora `shift(2)` excluye
    los días D−1 y D, que todavía no se observaron en el corte.
    """
    nombre = nombre or f"{modelo} corregido"
    d = pd.DataFrame(
        {
            "uid": cv["unique_id"],
            "dia": cv["ds"].dt.normalize(),
            "hora": cv["ds"].dt.hour,
            "e": cv["y"] - cv[modelo],
        }
    ).sort_values("dia")
    sesgo = d.groupby(["uid", "hora"])["e"].transform(
        lambda e: e.shift(RETRASO_DIAS).rolling(ventana_dias, min_periods=min_dias).median()
    )
    return cv.assign(**{nombre: cv[modelo] + sesgo.reindex(cv.index)})


def sesgo_reciente(
    y: pd.Series,
    yhat: pd.Series,
    hasta: pd.Timestamp,
    ventana_dias: int = VENTANA_DIAS,
    min_dias: int = MIN_DIAS,
) -> pd.Series:
    """Mediana de y − ŷ por hora del día en los `ventana_dias` días que terminan en `hasta`.

    Es la corrección de `corregir_sesgo_por_hora` para producción: con el corte al final del
    día `hasta`, se aplica a los días siguientes. Series indexadas por `ds`. Las horas con menos
    de `min_dias` errores quedan NaN.
    """
    e = (y - yhat).dropna()
    dia = e.index.normalize()
    hasta = pd.Timestamp(hasta).normalize()
    e = e[(dia <= hasta) & (dia > hasta - pd.Timedelta(days=ventana_dias))]
    g = e.groupby(e.index.hour)
    return g.median().where(g.size() >= min_dias).reindex(range(24))


def combinar(cv: pd.DataFrame, miembros: list[str], nombre: str = NOMBRE) -> pd.DataFrame:
    """Media simple de los miembros (NaN si falta alguno)."""
    return cv.assign(**{nombre: cv[miembros].mean(axis=1, skipna=False)})
