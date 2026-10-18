"""Backtest day-ahead de todos los modelos y tabla comparativa en el periodo de test.

Uso:
    uv run python -m forecast_energia.pipelines.backtest                    # todos los modelos
    uv run python -m forecast_energia.pipelines.backtest --modelos lgbm nhits
    uv run python -m forecast_energia.pipelines.backtest --solo-reporte     # reusa artifacts/

Cada familia de modelos se guarda en `artifacts/backtest/<unique_id>/<familia>.parquet`, así
se puede reejecutar una sola sin repetir las demás (TFT tarda bastante en CPU). El reporte
combina todo lo disponible en `artifacts/backtest/<unique_id>.parquet` y
`reports/metricas_test_<unique_id>.csv`.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from forecast_energia import config, features
from forecast_energia import evaluation as E
from forecast_energia.data import io
from forecast_energia.models import baselines, ensemble, ml, neural, stats

log = logging.getLogger("forecast_energia.backtest")

ARTIFACTS = config.ROOT / "artifacts" / "backtest"
REPORTS = config.ROOT / "reports"
CLAVES = ["unique_id", "ds", "cutoff"]
# Modelos aprendidos que pueden formar parte del ensamble (los naive y PREV no).
CANDIDATOS_ENSAMBLE = ["MSTL", ml.NOMBRE, "NHITS", "TFT"]

FAMILIAS: dict[str, Callable[[pd.DataFrame, E.Particion], pd.DataFrame]] = {
    "naive": baselines.backtest,
    "mstl": stats.backtest,
    "lgbm": ml.backtest,
    "nhits": lambda p, part: neural.backtest(p, part, [neural.nhits()]),
    "tft": lambda p, part: neural.backtest(p, part, [neural.tft()]),
}


def preparar_panel(unique_id: str) -> tuple[pd.DataFrame, pd.DataFrame, E.Particion]:
    """Panel con exógenas recortado a la partición, y las observaciones originales."""
    dem = io.cargar_demanda()
    observado = dem.loc[dem["unique_id"] == unique_id, ["unique_id", "ds", "y"]]
    panel = io.panel_nixtla(observado)
    particion = E.particion_por_defecto(panel["ds"].max())
    panel = E.recortar_para_backtest(panel, particion)
    panel = features.agregar_exogenas(panel, io.cargar_clima(), features.pesos_departamentales(dem))
    return panel, observado, particion


def _ruta(unique_id: str, familia: str) -> Path:
    return ARTIFACTS / unique_id.replace(" ", "_") / f"{familia}.parquet"


def correr(unique_id: str, familias: list[str]) -> None:
    panel, observado, particion = preparar_panel(unique_id)
    log.info(
        "%s: %d ventanas (calibración %d días, test %d días hasta %s)",
        unique_id,
        particion.n_ventanas(),
        particion.dias_calibracion,
        particion.dias_test,
        particion.fin.date(),
    )
    for familia in familias:
        t0 = time.time()
        cv = FAMILIAS[familia](panel, particion)
        cv = E.reemplazar_y_observado(E.filtrar_dia_objetivo(cv), observado)
        ruta = _ruta(unique_id, familia)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        cv.to_parquet(ruta, index=False)
        log.info(
            "%s/%s: %.0f s, %d filas → %s", unique_id, familia, time.time() - t0, len(cv), ruta
        )


def combinar(unique_id: str) -> pd.DataFrame:
    """Une las familias disponibles en un solo DataFrame ancho (una columna por modelo)."""
    partes = [pd.read_parquet(p) for p in sorted(_ruta(unique_id, "x").parent.glob("*.parquet"))]
    if not partes:
        raise FileNotFoundError(f"No hay backtests para {unique_id}: corré el pipeline primero.")
    base = partes[0]
    for p in partes[1:]:
        nuevas = [c for c in p.columns if c not in base.columns]
        base = base.merge(p[CLAVES + nuevas], on=CLAVES, how="inner")
    if unique_id == config.SIN_ID:
        base = baselines.agregar_prev(base, io.cargar_generacion())
    particion = E.particion_por_defecto(base["ds"].max())
    base["periodo"] = particion.etiqueta(base["ds"])
    return base


def modelos_en(cv: pd.DataFrame) -> list[str]:
    excluir = set(CLAVES) | {"y", "horizonte", "periodo"}
    return [c for c in cv.columns if c not in excluir and "-lo-" not in c and "-hi-" not in c]


def reporte(unique_id: str) -> pd.DataFrame:
    cv = combinar(unique_id)
    candidatos = [m for m in CANDIDATOS_ENSAMBLE if m in cv]
    if len(candidatos) >= 2:
        miembros = ensemble.seleccionar(cv, candidatos)
        cv = ensemble.agregar(cv, miembros)
        log.info("Ensamble = media de %s (mejores en calibración)", miembros)
    salida = ARTIFACTS / f"{unique_id.replace(' ', '_')}.parquet"
    cv.to_parquet(salida, index=False)

    dem = io.cargar_demanda()
    y_hist = io.serie(dem, unique_id)
    particion = E.particion_por_defecto(cv["ds"].max())
    escala = E.escala_mase(y_hist[y_hist.index < particion.calib_inicio])

    test = cv[cv["periodo"] == "test"]
    modelos = modelos_en(cv)
    tabla = E.evaluar(test, modelos, escala=escala).sort_values("MAE")
    calib = E.evaluar(cv[cv["periodo"] == "calibracion"], modelos)
    tabla.insert(0, "MAE_calib", calib["MAE"])

    referencia = baselines.NOMBRE_PREV if baselines.NOMBRE_PREV in modelos else "Naive168"
    t = test.set_index("ds")
    dm = {m: E.diebold_mariano(t["y"], t[m], t[referencia]) for m in modelos if m != referencia}
    tabla[f"DM_vs_{referencia}"] = pd.Series({m: r["estadistico"] for m, r in dm.items()})
    tabla[f"p_vs_{referencia}"] = pd.Series({m: r["p_valor"] for m, r in dm.items()})

    REPORTS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(REPORTS / f"metricas_test_{unique_id.replace(' ', '_')}.csv")
    return tabla


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--unique-id", default=config.SIN_ID)
    parser.add_argument("--modelos", nargs="+", choices=list(FAMILIAS), default=list(FAMILIAS))
    parser.add_argument("--solo-reporte", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if not args.solo_reporte:
        correr(args.unique_id, args.modelos)
    tabla = reporte(args.unique_id)
    columnas = [
        "MAE_calib",
        "MAE",
        "RMSE",
        "MAPE",
        "MASE",
        "sesgo",
        "cobertura_90",
        "ancho_90",
        "CRPS",
    ]
    columnas += [c for c in tabla.columns if c.startswith(("DM_", "p_"))]
    with pd.option_context("display.width", 200, "display.float_format", "{:,.3f}".format):
        print(tabla.reindex(columns=[c for c in columnas if c in tabla]).to_string())


if __name__ == "__main__":
    main()
