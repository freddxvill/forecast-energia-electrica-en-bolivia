"""Modelo estadístico: MSTL (estacionalidad diaria y semanal) con tendencia AutoETS."""

from __future__ import annotations

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoETS

from forecast_energia import config
from forecast_energia.evaluation import Particion

REFIT_CADA = 7  # ventanas (días) entre reentrenamientos
# Solo las últimas 8 semanas: MSTL sobre ~2 años de datos horarios tarda mucho (más de 30 min
# en el backtest) y una ventana corta además sigue mejor el nivel reciente.
VENTANA_ENTRENAMIENTO_H = 8 * 7 * 24


def modelos() -> list:
    return [MSTL(season_length=[24, 168], trend_forecaster=AutoETS(model="ZZN"), alias="MSTL")]


def backtest(panel: pd.DataFrame, particion: Particion) -> pd.DataFrame:
    """MSTL no usa exógenas: se le pasa solo `unique_id, ds, y`."""
    sf = StatsForecast(models=modelos(), freq="h")
    return sf.cross_validation(
        df=panel[["unique_id", "ds", "y"]],
        h=config.HORIZONTE_H,
        step_size=24,
        n_windows=particion.n_ventanas(),
        refit=REFIT_CADA,
        input_size=VENTANA_ENTRENAMIENTO_H,
    )
