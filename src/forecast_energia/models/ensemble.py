"""Ensamble: promedio de los k mejores modelos según el MAE del periodo de **calibración**.

La selección nunca mira el test. Elegir los miembros con los resultados del test inflaría
el desempeño reportado (*data snooping*).
"""

from __future__ import annotations

import pandas as pd

from forecast_energia import evaluation as E

NOMBRE = "Ensamble"


def seleccionar(cv: pd.DataFrame, candidatos: list[str], k: int = 2) -> list[str]:
    """Los `k` candidatos con menor MAE en el periodo de calibración."""
    calib = cv[cv["periodo"] == "calibracion"]
    mae = E.evaluar(calib, candidatos)["MAE"].sort_values()
    return list(mae.index[:k])


def agregar(cv: pd.DataFrame, miembros: list[str]) -> pd.DataFrame:
    """Agrega la columna `Ensamble` (media simple de los miembros)."""
    return cv.assign(**{NOMBRE: cv[miembros].mean(axis=1, skipna=False)})
