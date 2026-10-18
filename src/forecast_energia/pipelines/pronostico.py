"""Pronóstico de producción: mañana (y hoy) para el SIN y los 8 departamentos, con intervalos.

Uso:
    uv run python -m forecast_energia.pipelines.pronostico --inicializar   # una vez: histórico
    uv run python -m forecast_energia.pipelines.pronostico --entrenar-tft  # semanal: checkpoint
    uv run python -m forecast_energia.pipelines.pronostico                 # diario

Es el mismo esquema que se evaluó en el backtest jerárquico (Fase 6), con el corte en el último
día completo observado (D−2 23:00 para el día D = mañana):
1. LightGBM global (9 series), reentrenado en cada corrida (~1 min en CPU). El modelo nuevo
   reemplaza al anterior en `artifacts/modelos/lgbm_electric/`.
2. TFT global desde el checkpoint de `artifacts/modelos/tft_electric/` (inferencia en CPU),
   reentrenado cada semana. Si no hay checkpoint, el ensamble es solo LightGBM.

Los modelos se publican en la rama `modelos` y los artefactos de serving en `data-inference` (ver
`.github/scripts/rama-artefactos.sh`).
3. SIN: media del ensamble y el predespacho del CNDC (`precmp`) corregido por hora. Si el
   CNDC todavía no lo publicó, solo el ensamble.
4. Reconciliación MinT-WLS y conformal por hora + ACI por serie, con los errores de los
   pronósticos emitidos (`artifacts/historico_pronosticos.parquet`, que arranca con el backtest).

Salidas (`artifacts/demo/`, versionadas para la API y la app):
    pronostico_actual.parquet   días posteriores al corte: punto + intervalos 80/90/95 (+ PREV)
    backtest_reciente.parquet   últimos 90 días observados: real vs pronóstico vs PREV
    metricas.json               métricas de test (Fases 5–7) y del seguimiento reciente
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date

import numpy as np
import pandas as pd

from forecast_energia import config, features
from forecast_energia import conformal as C
from forecast_energia import evaluation as E
from forecast_energia import hierarchy as H
from forecast_energia.data import cndc, io
from forecast_energia.models import baselines, combinacion, ensemble, ml

log = logging.getLogger("forecast_energia.pronostico")

ARTIFACTS = config.ROOT / "artifacts"
REPORTS = config.ROOT / "reports"
DEMO = ARTIFACTS / "demo"
HISTORICO = ARTIFACTS / "historico_pronosticos.parquet"
CHECKPOINT_TFT = ARTIFACTS / "modelos" / "tft_electric"
MODELO_LGBM = ARTIFACTS / "modelos" / "lgbm_electric"
SERIES = [config.SIN_ID, *config.DEPARTAMENTO_CODIGOS]
PREV = baselines.NOMBRE_PREV
PUNTO = "Pronóstico"
DIAS_BACKTEST_RECIENTE = 90
COLUMNAS_HISTORICO = ["unique_id", "ds", "cutoff", "Base", ml.NOMBRE, "TFT", PREV]


# --- Datos ------------------------------------------------------------------------------------


def corte_actual(observado: pd.DataFrame) -> pd.Timestamp:
    """Última hora 23:00 con dato en todas las series (el último día completo)."""
    ultimo = observado.dropna(subset=["y"]).groupby("unique_id")["ds"].max().min()
    return (
        ultimo.normalize() + pd.Timedelta(hours=23)
        if ultimo.hour == 23
        else (ultimo.normalize() - pd.Timedelta(hours=1))
    )


def preparar(corte: pd.Timestamp | None = None):
    """Panel con exógenas hasta el corte y las 48 h siguientes (también con exógenas)."""
    dem = io.cargar_demanda()
    observado = dem.loc[dem["unique_id"].isin(SERIES), ["unique_id", "ds", "y"]]
    corte = corte or corte_actual(observado)
    panel = io.panel_nixtla(observado[observado["ds"] <= corte])
    fut = features.futuro(corte, SERIES, config.HORIZONTE_H)
    clima = io.cargar_clima()
    if clima["ds"].max() < fut["ds"].max():
        log.warning("El clima llega hasta %s: se completa por cercanía", clima["ds"].max())
    todo = features.agregar_exogenas(
        pd.concat([panel, fut], ignore_index=True), clima, features.pesos_departamentales(dem)
    )
    es_futuro = todo["y"].isna()
    return (
        todo[~es_futuro].reset_index(drop=True),
        todo[es_futuro].reset_index(drop=True),
        (
            observado,
            corte,
        ),
    )


# --- Modelos ----------------------------------------------------------------------------------


def entrenar_tft(corte: pd.Timestamp | None = None) -> None:
    from forecast_energia.models import neural  # torch solo hace falta para el TFT

    panel, _, (_, corte) = preparar(corte)
    t0 = time.time()
    nf = neural.entrenar(panel, [neural.tft()])
    CHECKPOINT_TFT.parent.mkdir(parents=True, exist_ok=True)
    neural.guardar(nf, CHECKPOINT_TFT)
    (CHECKPOINT_TFT / "entrenado_hasta.txt").write_text(str(corte), encoding="utf-8")
    log.info("TFT entrenado hasta %s en %.0f s → %s", corte, time.time() - t0, CHECKPOINT_TFT)


def pronosticos_base(panel: pd.DataFrame, fut: pd.DataFrame) -> pd.DataFrame:
    """LightGBM (reentrenado y guardado) + TFT (si hay checkpoint) y su ensamble, 48 h."""
    pred, mlf = ml.pronosticar(panel, fut)
    ml.guardar(mlf, MODELO_LGBM)
    (MODELO_LGBM / "entrenado_hasta.txt").write_text(str(panel["ds"].max()), encoding="utf-8")
    log.info("LightGBM entrenado hasta %s → %s", panel["ds"].max(), MODELO_LGBM)
    if (CHECKPOINT_TFT / "configuration.json").exists():
        from forecast_energia.models import neural

        tft = neural.pronosticar(neural.cargar(CHECKPOINT_TFT), panel, fut)
        pred = pred.merge(tft[["unique_id", "ds", "TFT"]], on=["unique_id", "ds"], how="left")
        miembros = [ml.NOMBRE, "TFT"]
    else:
        log.warning("Sin checkpoint del TFT (%s): el ensamble es solo LightGBM", CHECKPOINT_TFT)
        pred["TFT"] = np.nan
        miembros = [ml.NOMBRE]
    return ensemble.agregar(pred, miembros)


def agregar_prev(pred: pd.DataFrame, observado: pd.DataFrame, corte: pd.Timestamp) -> pd.DataFrame:
    """Predespacho del CNDC corregido por hora y Base = media con el ensamble (solo SIN)."""
    dias = sorted({d.date() for d in pred["ds"].dt.normalize()})
    try:
        prev = cndc.prev_predespacho(dias)
    except Exception as e:  # la API puede fallar: el pronóstico propio sigue siendo válido
        log.warning("No se pudo descargar el predespacho: %s", e)
        prev = pd.Series(dtype="float64", name=PREV)
    gen = io.cargar_generacion()
    hist_prev = gen[gen["unique_id"] == config.SERIE_PRONOSTICO_CNDC].set_index("ds")["y"]
    y_sin = observado[observado["unique_id"] == config.SIN_ID].set_index("ds")["y"]
    sesgo = combinacion.sesgo_reciente(y_sin, hist_prev, hasta=corte)

    es_sin = pred["unique_id"] == config.SIN_ID
    pred[PREV] = np.where(es_sin, pred["ds"].map(prev), np.nan)
    pred[f"{PREV} corregido"] = pred[PREV] + pred["ds"].dt.hour.map(sesgo)
    combinado = pred[[ensemble.NOMBRE, f"{PREV} corregido"]].mean(axis=1, skipna=False)
    pred["Base"] = np.where(es_sin & combinado.notna(), combinado, pred[ensemble.NOMBRE])
    faltan = es_sin & pred[PREV].isna()
    if faltan.any():
        log.warning("Sin predespacho para %d horas del SIN: se usa el ensamble", faltan.sum())
    return pred


# --- Histórico, reconciliación e intervalos ----------------------------------------------------


def inicializar_historico() -> pd.DataFrame:
    """Histórico de pronósticos base desde el backtest jerárquico (calibración + test)."""
    from forecast_energia.pipelines import backtest, jerarquia

    cv = jerarquia.bases(backtest.combinar("JERARQUIA"))
    cv = cv[cv["periodo"] != "train"]
    cv["Base"] = np.where(
        cv["unique_id"] == config.SIN_ID, cv[combinacion.NOMBRE], cv[ensemble.NOMBRE]
    )
    hist = cv.reindex(columns=COLUMNAS_HISTORICO).sort_values(["unique_id", "ds"])
    hist.to_parquet(HISTORICO, index=False)
    log.info(
        "Histórico inicializado: %d filas (%s → %s)", len(hist), hist["ds"].min(), hist["ds"].max()
    )
    return hist


def actualizar_historico(hist: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Agrega las horas nuevas. Las ya emitidas no se pisan: queda el pronóstico de mayor
    anticipación (el day-ahead), que es el que se evalúa."""
    nuevas = pred.reindex(columns=COLUMNAS_HISTORICO)
    clave = ["unique_id", "ds"]
    ya = nuevas.set_index(clave).index.isin(hist.set_index(clave).index)
    out = pd.concat([hist, nuevas[~ya]], ignore_index=True)
    return out.sort_values(clave).reset_index(drop=True)


def reconciliar_con_intervalos(hist: pd.DataFrame, observado: pd.DataFrame) -> pd.DataFrame:
    """MinT-WLS día a día sobre el histórico y conformal por hora + ACI por serie."""
    cv = hist.merge(observado, on=["unique_id", "ds"], how="left")
    rec = H.reconciliar_backtest(
        cv, {u: "Base" for u in SERIES}, config.SIN_ID, config.DEPARTAMENTO_CODIGOS
    )
    rec = rec[["unique_id", "ds", "y", "Base", "MinT-WLS"]].rename(columns={"MinT-WLS": PUNTO})
    rec["periodo"] = E.particion_por_defecto(rec["ds"].max()).etiqueta(rec["ds"])
    partes = [
        C.aci(g.reset_index(drop=True), PUNTO, PUNTO, ventana_dias=91)
        for _, g in rec.groupby("unique_id", sort=False)
    ]
    rec = pd.concat(partes, ignore_index=True)
    return rec.merge(hist[["unique_id", "ds", PREV]], on=["unique_id", "ds"], how="left")


# --- Salidas de la demo ------------------------------------------------------------------------


def _metricas_recientes(rec: pd.DataFrame, dias: int) -> dict:
    """Seguimiento del SIN en los últimos `dias` días observados: pronóstico vs PREV."""
    sin = rec[(rec["unique_id"] == config.SIN_ID)].dropna(subset=["y", PUNTO, PREV])
    sin = sin[sin["ds"] > sin["ds"].max() - pd.Timedelta(days=dias)]
    if sin.empty:
        return {}
    m = E.metricas_puntuales(sin["y"], sin[PUNTO])
    p = E.metricas_puntuales(sin["y"], sin[PREV])
    prob = E.metricas_probabilisticas(sin.dropna(subset=[f"{PUNTO}-lo-90"]), PUNTO)
    return {
        "desde": str(sin["ds"].min().date()),
        "hasta": str(sin["ds"].max().date()),
        "MAE": m["MAE"],
        "MAPE": m["MAPE"],
        "MAE_PREV": p["MAE"],
        "MAPE_PREV": p["MAPE"],
        **{f"cobertura_{n}": prob[f"cobertura_{n}"] for n in config.NIVELES},
    }


def _tabla(csv: str) -> list[dict]:
    path = REPORTS / csv
    return (
        pd.read_csv(path).rename(columns={"Unnamed: 0": "nombre"}).to_dict("records")
        if (path.exists())
        else []
    )


def guardar_demo(rec: pd.DataFrame, pred: pd.DataFrame, corte: pd.Timestamp) -> None:
    DEMO.mkdir(parents=True, exist_ok=True)
    cols = ["unique_id", "ds", PUNTO] + [
        f"{PUNTO}-{lado}-{n}" for n in config.NIVELES for lado in ("lo", "hi")
    ]
    actual = rec[rec["ds"] > corte][cols + [PREV]]
    actual = actual.merge(
        pred[["unique_id", "ds", f"{PREV} corregido", ml.NOMBRE, "TFT"]],
        on=["unique_id", "ds"],
        how="left",
    )
    actual = actual.assign(corte=corte, emitido=pd.Timestamp.now(tz=config.TZ).tz_localize(None))
    actual.to_parquet(DEMO / "pronostico_actual.parquet", index=False)

    obs = rec[rec["y"].notna()]
    reciente = obs[obs["ds"] > obs["ds"].max() - pd.Timedelta(days=DIAS_BACKTEST_RECIENTE)]
    reciente[["y", *cols, PREV, "periodo"]].to_parquet(
        DEMO / "backtest_reciente.parquet", index=False
    )

    impacto = REPORTS / "impacto.json"
    metricas = {
        "corte": str(corte),
        "emitido": str(actual["emitido"].iloc[0]),
        "usa_tft": bool(actual["TFT"].notna().any()),
        "usa_prev": bool(actual.loc[actual["unique_id"] == config.SIN_ID, PREV].notna().all()),
        "seguimiento_30_dias": _metricas_recientes(rec, 30),
        "test_intervalos_SIN": _tabla("metricas_intervalos_SIN.csv"),
        "test_jerarquia_mape": _tabla("metricas_jerarquia.csv"),
        "test_jerarquia_cobertura": _tabla("cobertura_jerarquia.csv"),
        "impacto": json.loads(impacto.read_text(encoding="utf-8")) if impacto.exists() else {},
    }
    (DEMO / "metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def pronosticar(corte: pd.Timestamp | None = None) -> pd.DataFrame:
    t0 = time.time()
    panel, fut, (observado, corte) = preparar(corte)
    log.info("Corte %s → pronóstico de %s a %s", corte, fut["ds"].min(), fut["ds"].max())
    pred = agregar_prev(pronosticos_base(panel, fut), observado, corte)
    pred["cutoff"] = corte
    hist = pd.read_parquet(HISTORICO) if HISTORICO.exists() else inicializar_historico()
    hist = actualizar_historico(hist, pred)
    hist.to_parquet(HISTORICO, index=False)
    rec = reconciliar_con_intervalos(hist, observado)
    guardar_demo(rec, pred, corte)
    log.info("Listo en %.0f s → %s", time.time() - t0, DEMO)
    return rec[rec["ds"] > corte]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--inicializar", action="store_true", help="recrea el histórico")
    parser.add_argument("--entrenar-tft", action="store_true", help="reentrena el TFT")
    parser.add_argument("--corte", type=date.fromisoformat, help="día del corte (pruebas)")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    corte = pd.Timestamp(args.corte) + pd.Timedelta(hours=23) if args.corte else None
    if args.inicializar:
        inicializar_historico()
    if args.entrenar_tft:
        entrenar_tft(corte)
        return
    out = pronosticar(corte)
    sin = out[out["unique_id"] == config.SIN_ID].set_index("ds")
    cols = [PUNTO, f"{PUNTO}-lo-90", f"{PUNTO}-hi-90", PREV]
    with pd.option_context("display.float_format", "{:,.0f}".format, "display.width", 150):
        print(sin[cols].to_string())


if __name__ == "__main__":
    main()
