"""Modelos neuronales (NeuralForecast) con pérdida cuantílica: NHITS y TFT.

Ambos usan exógenas futuras (calendario cíclico, feriados y clima pronosticado) y `MQLoss`,
que entrega directamente los cuantiles de los intervalos 80/90/95. Son redes pequeñas que
corren en CPU o en GPU si hay una disponible (p.ej. en Kaggle; se puede forzar con la variable
de entorno `FORECAST_ACCELERATOR`). Se reentrenan cada `REFIT_CADA` ventanas.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import NHITS, TFT

from forecast_energia import config, features
from forecast_energia.evaluation import Particion

REFIT_CADA = 91  # ≈ al inicio de la calibración, del test y a mitad del test
EXOGENAS = (
    features.CICLICAS + ["es_fin_de_semana", "es_feriado", "pre_feriado", "post_feriado"]
) + features.CLIMA

_TRAINER = dict(
    accelerator=os.environ.get("FORECAST_ACCELERATOR", "auto"),
    # Una sola GPU: con varias (p.ej. T4×2 en Kaggle) Lightning usa DDP, que relanza el script
    # en otro proceso y falla al predecir una única serie.
    devices=1,
    enable_progress_bar=False,
    enable_model_summary=False,
    logger=False,
    enable_checkpointing=False,
)


def _silenciar_logs() -> None:
    for nombre in ("pytorch_lightning", "lightning.pytorch", "lightning_fabric"):
        logging.getLogger(nombre).setLevel(logging.ERROR)


def nhits(max_steps: int = 1500) -> NHITS:
    return NHITS(
        h=config.HORIZONTE_H,
        input_size=7 * 24,
        futr_exog_list=EXOGENAS,
        loss=MQLoss(level=list(config.NIVELES)),
        n_pool_kernel_size=[8, 4, 1],
        n_freq_downsample=[24, 12, 1],
        mlp_units=[[256, 256]] * 3,
        dropout_prob_theta=0.1,
        scaler_type="robust",
        learning_rate=1e-3,
        max_steps=max_steps,
        batch_size=32,
        windows_batch_size=256,
        random_seed=1,
        alias="NHITS",
        **_TRAINER,
    )


def tft(max_steps: int = 800) -> TFT:
    return TFT(
        h=config.HORIZONTE_H,
        input_size=7 * 24,
        futr_exog_list=EXOGENAS,
        hidden_size=32,
        n_head=4,
        dropout=0.1,
        loss=MQLoss(level=list(config.NIVELES)),
        scaler_type="robust",
        learning_rate=1e-3,
        max_steps=max_steps,
        batch_size=32,
        windows_batch_size=128,
        random_seed=1,
        alias="TFT",
        **_TRAINER,
    )


def _renombrar_mediana(cv: pd.DataFrame) -> pd.DataFrame:
    """MQLoss devuelve `<modelo>-median`: se renombra a `<modelo>` (pronóstico puntual)."""
    return cv.rename(columns={c: c.removesuffix("-median") for c in cv.columns})


def backtest(panel: pd.DataFrame, particion: Particion, modelos: list) -> pd.DataFrame:
    """Backtest de los modelos dados (p.ej. `[nhits()]`) con las exógenas de `features`."""
    _silenciar_logs()
    nf = NeuralForecast(models=modelos, freq="h")
    cv = nf.cross_validation(
        df=panel[["unique_id", "ds", "y", *EXOGENAS]],
        n_windows=particion.n_ventanas(),
        step_size=24,
        refit=REFIT_CADA,
    )
    return _renombrar_mediana(cv)
