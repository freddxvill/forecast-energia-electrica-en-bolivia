"""Pronóstico jerárquico SIN → 8 departamentos: reconciliación, intervalos y comparación.

Uso:
    uv run python -m forecast_energia.pipelines.backtest --conjunto JERARQUIA --modelos naive lgbm
    (TFT de la jerarquía en Kaggle → artifacts/backtest/JERARQUIA/tft.parquet)
    uv run python -m forecast_energia.pipelines.jerarquia

Pronósticos base:
- departamentos: Ensamble = media de LightGBM y TFT globales (entrenados sobre las 9 series);
- SIN: Combinado = media del Ensamble y PREV corregido por hora (lo mejor de la Fase 5).

Luego se reconcilian con BottomUp, TopDown, MinT-WLS y MinT-shrink. El método se elige por el
MASE medio de las 9 series en calibración, y se le agregan intervalos conformal + ACI por serie.

Salidas:
    artifacts/jerarquia.parquet                 pronósticos reconciliados + intervalos
    reports/metricas_jerarquia.csv              MAPE de test por serie y método
    reports/figures/jerarquia_mape_por_serie.png
"""

from __future__ import annotations

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecast_energia import config, viz
from forecast_energia import conformal as C
from forecast_energia import evaluation as E
from forecast_energia import hierarchy as H
from forecast_energia.data import io
from forecast_energia.models import baselines, combinacion, ensemble, ml

log = logging.getLogger("forecast_energia.jerarquia")

ARTIFACTS = config.ROOT / "artifacts"
REPORTS = config.ROOT / "reports"
METODOS = ["Base", "BottomUp", "TopDown", "MinT-WLS", "MinT-shrink"]
MIEMBROS = [ml.NOMBRE, "TFT"]


def bases(cv: pd.DataFrame) -> pd.DataFrame:
    """Agrega el Ensamble (todas las series) y el Combinado con PREV corregido (solo SIN)."""
    cv = ensemble.agregar(cv, MIEMBROS)
    sin = cv[cv["unique_id"] == config.SIN_ID]
    sin = combinacion.corregir_sesgo_por_hora(sin, baselines.NOMBRE_PREV)
    sin = combinacion.combinar(sin, [ensemble.NOMBRE, f"{baselines.NOMBRE_PREV} corregido"])
    return cv.merge(
        sin[["unique_id", "ds", combinacion.NOMBRE]], on=["unique_id", "ds"], how="left"
    )


def escalas_mase(observado: pd.DataFrame, antes_de: pd.Timestamp) -> pd.Series:
    """MAE del naive 168 h in-sample por serie (previo a la calibración)."""
    hist = observado[observado["ds"] < antes_de]
    return hist.groupby("unique_id")["y"].apply(E.escala_mase)


def tabla_por_serie(
    rec: pd.DataFrame, metrica: str, escalas: pd.Series | None = None
) -> pd.DataFrame:
    """Métrica por serie (filas) y método (columnas), sobre las horas con todos los métodos."""
    d = rec.dropna(subset=METODOS + ["y"])
    filas = {}
    for uid, g in d.groupby("unique_id"):
        escala = None if escalas is None else escalas[uid]
        filas[uid] = {m: E.metricas_puntuales(g["y"], g[m], escala)[metrica] for m in METODOS}
    orden = [config.SIN_ID, *config.DEPARTAMENTO_CODIGOS]
    return pd.DataFrame(filas).T.reindex(orden)


def intervalos_por_serie(rec: pd.DataFrame, metodo: str) -> pd.DataFrame:
    """Conformal rodante por hora + ACI aplicado a cada serie reconciliada por separado."""
    partes = []
    for _, g in rec.groupby("unique_id", sort=False):
        partes.append(C.aci(g.reset_index(drop=True), metodo, "Jerárquico + ACI", ventana_dias=91))
    return pd.concat(partes, ignore_index=True)


def figura_mape(tabla: pd.DataFrame, metodo: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = np.arange(len(tabla))
    ax.barh(
        y + 0.2, tabla["Base"], height=0.38, color=viz.GRIS_CONTEXTO, label="Base (sin reconciliar)"
    )
    ax.barh(
        y - 0.2,
        tabla[metodo],
        height=0.38,
        color=viz.CATEGORICA[0],
        label=f"Reconciliado ({metodo})",
    )
    for i, (b, r) in enumerate(zip(tabla["Base"], tabla[metodo], strict=True)):
        ax.annotate(
            f"{r:.1f} %",
            (r, i - 0.2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=viz.TINTA_2,
        )
        ax.annotate(
            f"{b:.1f} %",
            (b, i + 0.2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=viz.TINTA_SUAVE,
        )
    ax.set_yticks(y, [u.title() if u != config.SIN_ID else "SIN (total)" for u in tabla.index])
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set(
        title="MAPE de test por serie: pronóstico base vs reconciliado",
        xlabel="MAPE (%)",
        xlim=(0, tabla[["Base", metodo]].max().max() * 1.18),
    )
    ax.legend(loc="upper right")
    viz.guardar(fig, "jerarquia_mape_por_serie")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    cv = bases(pd.read_parquet(ARTIFACTS / "backtest" / "JERARQUIA.parquet"))
    base = {config.SIN_ID: combinacion.NOMBRE} | {
        d: ensemble.NOMBRE for d in config.DEPARTAMENTO_CODIGOS
    }
    rec = H.reconciliar_backtest(cv, base, config.SIN_ID, config.DEPARTAMENTO_CODIGOS)
    particion = E.particion_por_defecto(rec["ds"].max())
    rec["periodo"] = particion.etiqueta(rec["ds"])

    escalas = escalas_mase(io.cargar_demanda(), particion.calib_inicio)
    calib = tabla_por_serie(rec[rec["periodo"] == "calibracion"], "MASE", escalas)
    elegido = calib.mean().drop("Base").idxmin()
    log.info("MASE medio en calibración: %s → elegido %s", calib.mean().round(3).to_dict(), elegido)

    test = rec[rec["periodo"] == "test"]
    mape = tabla_por_serie(test, "MAPE")
    mase = tabla_por_serie(test, "MASE", escalas)
    mape.loc["Media deptos"] = mape.loc[config.DEPARTAMENTO_CODIGOS].mean()
    mape.loc["MASE medio (9 series)"] = mase.mean()

    rec = intervalos_por_serie(rec, elegido)
    cobertura = (
        rec[(rec["periodo"] == "test") & rec["Jerárquico + ACI-lo-90"].notna()]
        .groupby("unique_id")
        .apply(
            lambda g: pd.Series(E.metricas_probabilisticas(g, "Jerárquico + ACI")),
            include_groups=False,
        )[["cobertura_80", "cobertura_90", "cobertura_95", "ancho_90"]]
        .reindex([config.SIN_ID, *config.DEPARTAMENTO_CODIGOS])
    )

    rec.to_parquet(ARTIFACTS / "jerarquia.parquet", index=False)
    REPORTS.mkdir(parents=True, exist_ok=True)
    mape.to_csv(REPORTS / "metricas_jerarquia.csv")
    cobertura.to_csv(REPORTS / "cobertura_jerarquia.csv")
    viz.estilo()
    figura_mape(mape.loc[[config.SIN_ID, *config.DEPARTAMENTO_CODIGOS]], elegido)

    with pd.option_context("display.width", 200, "display.float_format", "{:,.2f}".format):
        print(f"Método elegido por calibración: {elegido}\n")
        print("MAPE (%) en test por serie y método:")
        print(mape.to_string())
        print("\nIntervalos (test) del método elegido:")
        print(cobertura.to_string())


if __name__ == "__main__":
    main()
