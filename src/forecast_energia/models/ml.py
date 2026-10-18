"""LightGBM con MLForecast sobre la demanda normalizada por su nivel reciente.

- **Lags ≥ 48 h** (el horizonte máximo): la predicción nunca usa valores pronosticados como
  entrada, así que un único modelo equivale a una estrategia directa, sin errores recursivos.
- **Target normalizado** `y / nivel`, con `nivel` = media de `y` en la semana que termina en
  t−48 (conocida en el corte para todo h ≤ 48). Los árboles no extrapolan: sobre `y` cruda
  subestimaban cuando la demanda superaba los niveles históricos (sesgo de −57 MW en
  calibración). Sobre el cociente aprenden la forma del perfil y el nivel lo aporta `nivel`.
- **Sin día del año:** con ~2 años de historia, los árboles memorizaban "marzo = baja demanda"
  de 2024–2025. La estacionalidad anual la explican el clima y los lags.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd

from forecast_energia import config, features
from forecast_energia.evaluation import Particion

NOMBRE = "LightGBM"
REFIT_CADA = 7
LAGS = [48, 72, 96, 120, 144, 168, 336, 504]
EXOGENAS = features.CALENDARIO + features.CLIMA
VENTANA_NIVEL_H = 168

PARAMS = dict(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)


def crear(**params) -> MLForecast:
    return MLForecast(
        models={NOMBRE: lgb.LGBMRegressor(**(PARAMS | params))},
        freq="h",
        lags=LAGS,
        lag_transforms={
            48: [RollingMean(window_size=24), RollingMean(window_size=168), RollingStd(24)],
            168: [RollingMean(window_size=168)],
        },
    )


def nivel(panel: pd.DataFrame) -> pd.Series:
    """Media de `y` en [t−48−167, t−48]: solo usa datos observados ≤ t−48 (≤ el corte)."""
    return panel.groupby("unique_id")["y"].transform(
        lambda y: y.rolling(VENTANA_NIVEL_H).mean().shift(config.HORIZONTE_H)
    )


def pronosticar(panel: pd.DataFrame, futuro: pd.DataFrame) -> pd.DataFrame:
    """Entrena con toda la historia de `panel` y pronostica las filas de `futuro` (≤ 48 h).

    Ambos deben traer las exógenas de `features.agregar_exogenas`; `futuro` sin `y`. El nivel
    de las horas futuras usa solo datos observados (está desplazado 48 h).
    """
    todo = pd.concat([panel, futuro], ignore_index=True)
    todo = todo.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    todo["nivel"] = nivel(todo)
    es_futuro = todo["y"].isna()
    hist = todo[~es_futuro].dropna(subset=["nivel"])
    fut = todo[es_futuro]
    mlf = crear().fit(
        hist.assign(y=hist["y"] / hist["nivel"])[["unique_id", "ds", "y", *EXOGENAS]],
        static_features=[],
    )
    h = int(fut.groupby("unique_id").size().max())
    pred = mlf.predict(h=h, X_df=fut[["unique_id", "ds", *EXOGENAS]])
    pred = pred.merge(fut[["unique_id", "ds", "nivel"]], on=["unique_id", "ds"], how="inner")
    pred[NOMBRE] = pred[NOMBRE] * pred.pop("nivel")
    return pred


def backtest(panel: pd.DataFrame, particion: Particion) -> pd.DataFrame:
    """`panel` debe traer las exógenas de `features.agregar_exogenas`."""
    p = panel.assign(nivel=nivel(panel)).dropna(subset=["nivel"]).reset_index(drop=True)
    normalizado = p.assign(y=p["y"] / p["nivel"])
    cv = crear().cross_validation(
        df=normalizado[["unique_id", "ds", "y", *EXOGENAS]],
        h=config.HORIZONTE_H,
        step_size=24,
        n_windows=particion.n_ventanas(),
        refit=REFIT_CADA,
        static_features=[],  # todas las exógenas son dinámicas (conocidas a futuro)
    )
    cv = cv.merge(p[["unique_id", "ds", "nivel"]], on=["unique_id", "ds"], how="left")
    cv[NOMBRE] = cv[NOMBRE] * cv.pop("nivel")
    return cv  # `y` queda normalizada: el pipeline la reemplaza por el valor observado
