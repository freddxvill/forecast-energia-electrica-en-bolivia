"""Intervalos conformal sobre el backtest y comparación de métodos en el periodo de test.

Uso:
    uv run python -m forecast_energia.pipelines.intervalos

Lee `artifacts/backtest/SIN.parquet` (generado por `pipelines.backtest`) y escribe:
    artifacts/intervalos_SIN.parquet           pronósticos + intervalos de todos los métodos
    reports/metricas_intervalos_SIN.csv        cobertura, ancho, Winkler y CRPS por método
    reports/figures/cp_cobertura_por_hora.png  cobertura condicional por hora (90 %)
    reports/figures/cp_semana_ejemplo.png      semana de test con bandas 80/95 %
"""

from __future__ import annotations

import argparse
import logging

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from forecast_energia import config, viz
from forecast_energia import conformal as C
from forecast_energia import evaluation as E
from forecast_energia.models import baselines, combinacion, ensemble

log = logging.getLogger("forecast_energia.intervalos")

ARTIFACTS = config.ROOT / "artifacts"
REPORTS = config.ROOT / "reports"
VENTANA_DIAS = 91
GAMMA_ACI = 0.05
# Método de producción: pronóstico combinado (ensamble propio + PREV corregido por hora) con
# conformal rodante por hora + ACI. El respaldo, si el CNDC aún no publicó PREV, es el
# ensamble solo con el mismo esquema de intervalos.
METODO_PRODUCCION = "Combinado + ACI"
METODO_RESPALDO = "Ensamble + ACI"

ETIQUETAS = {
    "TFT": "TFT (cuantiles crudos)",
    "TFT + CQR": "TFT + CQR rodante",
    "Ensamble + CP estático": "Ensamble + CP estático",
    "Ensamble + CP sin hora": "Ensamble + CP rodante (sin Mondrian)",
    "Ensamble + CP rodante": "Ensamble + CP rodante",
    "Ensamble + ACI": "Ensamble + CP rodante + ACI",
    "PREV + CP": "PREV + CP rodante",
    "Combinado + ACI": "Combinado (Ensamble + PREV corregido) + ACI",
}


def construir(cv: pd.DataFrame) -> pd.DataFrame:
    ens = ensemble.NOMBRE
    cv = C.split_conformal(cv, ens, "Ensamble + CP estático", ventana_dias=None)
    cv = C.split_conformal(
        cv, ens, "Ensamble + CP sin hora", ventana_dias=VENTANA_DIAS, por_hora=False
    )
    cv = C.split_conformal(cv, ens, "Ensamble + CP rodante", ventana_dias=VENTANA_DIAS)
    cv = C.aci(cv, ens, "Ensamble + ACI", ventana_dias=VENTANA_DIAS, gamma=GAMMA_ACI)
    cv = C.cqr(cv, "TFT", "TFT + CQR", ventana_dias=VENTANA_DIAS)
    if baselines.NOMBRE_PREV in cv:
        prev = baselines.NOMBRE_PREV
        cv = C.split_conformal(cv, prev, "PREV + CP", ventana_dias=VENTANA_DIAS)
        cv = combinacion.corregir_sesgo_por_hora(cv, prev)
        cv = combinacion.combinar(cv, [ens, f"{prev} corregido"])
        cv = C.aci(
            cv, combinacion.NOMBRE, "Combinado + ACI", ventana_dias=VENTANA_DIAS, gamma=GAMMA_ACI
        )
    return cv


def comparar(cv: pd.DataFrame) -> pd.DataFrame:
    metodos = [m for m in ETIQUETAS if f"{m}-lo-90" in cv]
    test = cv[cv["periodo"] == "test"]
    for m in metodos:  # todas las métricas sobre las mismas horas
        test = test[test[f"{m}-lo-90"].notna()]
    tabla = E.evaluar(test, metodos)
    hora = test["ds"].dt.hour
    mes = test["ds"].dt.to_period("M")
    for m in metodos:
        por_hora = E.cobertura_por_grupo(test, m, 90, hora)
        por_mes = E.cobertura_por_grupo(test, m, 90, mes)
        tabla.loc[m, "cob90_hora_min"] = por_hora.min()
        tabla.loc[m, "cob90_hora_max"] = por_hora.max()
        tabla.loc[m, "cob90_mes_min"] = por_mes.min()
    tabla.index = [ETIQUETAS[m] for m in tabla.index]
    return tabla


def figura_cobertura_por_hora(cv: pd.DataFrame) -> None:
    test = cv[(cv["periodo"] == "test") & cv[f"{METODO_PRODUCCION}-lo-90"].notna()]
    hora = test["ds"].dt.hour
    series = {
        "TFT (cuantiles crudos)": ("TFT", viz.CATEGORICA[1]),
        "CP rodante sin Mondrian": ("Ensamble + CP sin hora", viz.CATEGORICA[2]),
        "Combinado + CP por hora + ACI": (METODO_PRODUCCION, viz.CATEGORICA[0]),
    }
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(90, color=viz.TINTA, lw=1)
    ax.annotate(
        "nominal 90 %",
        (23.3, 90),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=9,
        color=viz.TINTA_2,
        ha="right",
    )
    for etiqueta, (m, color) in series.items():
        cob = E.cobertura_por_grupo(test, m, 90, hora)
        ax.plot(cob.index, cob.values, color=color, marker="o", ms=4, label=etiqueta)
    ax.set(
        title="Cobertura del intervalo 90 % por hora del día (test)",
        xlabel="Hora",
        ylabel="% de horas dentro del intervalo",
        xticks=range(0, 24, 3),
        ylim=(40, 102),
    )
    ax.legend(loc="lower left")
    viz.guardar(fig, "cp_cobertura_por_hora")
    plt.close(fig)


def figura_semana(cv: pd.DataFrame, metodo: str = METODO_PRODUCCION) -> None:
    test = cv[cv["periodo"] == "test"].set_index("ds").sort_index()
    semana = test.loc["2026-09-21":"2026-09-27 23:00"]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(
        semana.index,
        semana[f"{metodo}-lo-95"],
        semana[f"{metodo}-hi-95"],
        color=viz.CATEGORICA[0],
        alpha=0.15,
        lw=0,
        label="Intervalo 95 %",
    )
    ax.fill_between(
        semana.index,
        semana[f"{metodo}-lo-80"],
        semana[f"{metodo}-hi-80"],
        color=viz.CATEGORICA[0],
        alpha=0.30,
        lw=0,
        label="Intervalo 80 %",
    )
    ax.plot(semana.index, semana[metodo], color=viz.CATEGORICA[0], lw=1.6, label="Pronóstico")
    ax.plot(semana.index, semana["y"], color=viz.TINTA, lw=2, label="Real")
    ax.set(
        title="Pronóstico day-ahead del SIN con intervalos conformal (semana de test)", ylabel="MW"
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    ax.legend(loc="upper left", ncols=4)
    viz.guardar(fig, "cp_semana_ejemplo")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--unique-id", default=config.SIN_ID)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    uid = args.unique_id.replace(" ", "_")
    cv = construir(pd.read_parquet(ARTIFACTS / "backtest" / f"{uid}.parquet"))
    cv.to_parquet(ARTIFACTS / f"intervalos_{uid}.parquet", index=False)
    tabla = comparar(cv)
    REPORTS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(REPORTS / f"metricas_intervalos_{uid}.csv")

    viz.estilo()
    figura_cobertura_por_hora(cv)
    figura_semana(cv)

    cols = [
        "MAE",
        "cobertura_80",
        "cobertura_90",
        "cobertura_95",
        "ancho_90",
        "winkler_90",
        "CRPS",
        "cob90_hora_min",
        "cob90_hora_max",
        "cob90_mes_min",
    ]
    with pd.option_context("display.width", 250, "display.float_format", "{:,.1f}".format):
        print(tabla[cols].to_string())


if __name__ == "__main__":
    main()
