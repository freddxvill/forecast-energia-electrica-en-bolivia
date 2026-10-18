"""Baselines: naive estacionales válidos para el horizonte 25–48 h y el pronóstico oficial PREV."""

from __future__ import annotations

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from forecast_energia import config
from forecast_energia.evaluation import Particion

NOMBRE_PREV = "PREV"


def backtest(panel: pd.DataFrame, particion: Particion) -> pd.DataFrame:
    """Naive semanal (y(t−168)) y naive de 2 días (y(t−48)).

    Un naive de 24 h no es válido: con el corte en D−2, y(t−24) todavía no se observó.
    """
    sf = StatsForecast(
        models=[
            SeasonalNaive(season_length=168, alias="Naive168"),
            SeasonalNaive(season_length=48, alias="Naive48"),
        ],
        freq="h",
    )
    return sf.cross_validation(
        df=panel[["unique_id", "ds", "y"]],
        h=config.HORIZONTE_H,
        step_size=24,
        n_windows=particion.n_ventanas(),
    )


def agregar_prev(cv: pd.DataFrame, generacion: pd.DataFrame) -> pd.DataFrame:
    """Agrega el pronóstico oficial del CNDC como columna `PREV` (solo aplica al SIN)."""
    prev = generacion.loc[generacion["unique_id"] == config.SERIE_PRONOSTICO_CNDC, ["ds", "y"]]
    prev = prev.rename(columns={"y": NOMBRE_PREV}).assign(unique_id=config.SIN_ID)
    return cv.merge(prev, on=["unique_id", "ds"], how="left")
