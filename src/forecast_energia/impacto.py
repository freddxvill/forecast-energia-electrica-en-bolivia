"""Impacto de negocio del pronóstico: desvíos evitados, su valor económico y la reserva.

Todas las funciones son puras y reciben series horarias en MW. Como cada fila es una hora, un
error de `e` MW equivale a `e` MWh de energía mal programada.

Las cifras económicas son **órdenes de magnitud con supuestos explícitos**
(`config.SUPUESTOS_IMPACTO`), no un costo contable: suponen que cada MWh de desvío se
resuelve en tiempo real con generación térmica de ajuste, valorizada a la tarifa regulada o
al costo de oportunidad del gas que esa generación quema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DIAS_ANIO = 365
MESES_ANIO = 12
KW_POR_MW = 1000


def desvio_evitado(y: pd.Series, base: pd.Series, modelo: pd.Series) -> pd.Series:
    """MWh de desvío evitados en cada hora: |y − base| − |y − modelo| (negativo si empeora)."""
    return (y - base).abs() - (y - modelo).abs()


def anualizar(total: float, dias: int) -> float:
    return total * DIAS_ANIO / dias


def precio_por_hora(ds: pd.Series, mensuales: pd.DataFrame, columna: str) -> pd.Series:
    """Precio del mes de cada hora. Los meses aún no publicados usan el último disponible."""
    mes = pd.to_datetime(ds).dt.to_period("M").dt.to_timestamp()
    tabla = mensuales.set_index("mes")[columna].sort_index()
    meses = pd.DatetimeIndex(sorted(set(mes) | set(tabla.index)))
    tabla = tabla.reindex(meses).ffill()
    return pd.Series(tabla.reindex(mes).to_numpy(), index=ds.index)


def valor_desvios(mwh: pd.Series, precio_usd_mwh: pd.Series | float) -> float:
    """USD de los MWh evitados, hora a hora, al precio indicado."""
    return float((mwh * precio_usd_mwh).sum())


def costo_combustible_usd_mwh(heat_rate_mmbtu_mwh: float, precio_gas_usd_mmbtu: float) -> float:
    """Costo del gas para producir 1 MWh térmico."""
    return heat_rate_mmbtu_mwh * precio_gas_usd_mmbtu


def heat_rate_implicito(costo_marginal_usd_mwh: float, precio_gas_usd_mmbtu: float) -> float:
    """Consumo específico que explicaría el costo marginal solo con combustible (cota superior).

    El costo marginal del CNDC es el costo variable de la unidad marginal (combustible + O&M),
    así que dividirlo por el precio regulado del gas acota el heat rate por arriba.
    """
    return costo_marginal_usd_mwh / precio_gas_usd_mmbtu


def reserva_fija(y_calib: pd.Series, yhat_calib: pd.Series, nivel: float) -> float:
    """MW constantes que cubren el `nivel` % de los desvíos al alza (y − ŷ) en calibración."""
    e = (y_calib - yhat_calib).dropna().to_numpy()
    return float(np.quantile(e, nivel / 100))


def reserva_fija_con_cobertura(y: pd.Series, yhat: pd.Series, cobertura_pct: float) -> float:
    """MW constantes que logran `cobertura_pct` sobre los mismos (y, ŷ) que se evalúan.

    Es un "oráculo" (mira los datos que evalúa): sirve como cota conservadora para comparar
    reglas a igual cobertura, sin premiar a la que cubre menos.
    """
    return float(np.quantile((y - yhat).dropna().to_numpy(), cobertura_pct / 100))


def resumen_reserva(y: pd.Series, yhat: pd.Series, limite_superior: pd.Series) -> dict:
    """Reserva al alza requerida (límite superior − pronóstico) y la cobertura que logra."""
    m = y.notna() & yhat.notna() & limite_superior.notna()
    reserva = (limite_superior - yhat)[m]
    return {
        "reserva_media_MW": float(reserva.mean()),
        "reserva_max_MW": float(reserva.max()),
        "cobertura_pct": float(100 * (y[m] <= limite_superior[m]).mean()),
        "n": int(m.sum()),
    }


def valor_reserva_usd_anio(mw: float, precio_potencia_usd_kw_mes: float) -> float:
    """Valor anual de `mw` de capacidad al precio de potencia (USD/kW-mes)."""
    return mw * KW_POR_MW * precio_potencia_usd_kw_mes * MESES_ANIO
