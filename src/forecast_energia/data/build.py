"""Pipeline de datos: API CNDC + Open-Meteo → `data/processed/*.parquet`.

Uso:
    uv run python -m forecast_energia.data.build                 # incremental hasta ayer
    uv run python -m forecast_energia.data.build --start 2025-01-01 --end 2025-01-31
    uv run python -m forecast_energia.data.build --offline       # solo reprocesa la caché

Salidas (formato largo de Nixtla: `unique_id, ds, y`, con `ds` = inicio de la hora local):
    demanda_horaria.parquet     8 departamentos + SIN (MW): `y` conciliado con TOT,
                                `y_medido` tal como lo publica el CNDC, `imputado`
    generacion_horaria.parquet  TOT, TERMO, HIDRO, SOLAR, EOL, BAGAZO, RENO, PREV (MW)
    clima_horario.parquet       clima observado y pronosticado 48 h antes (`*_d2`)
    precios_mensuales.parquet   precios monómicos del CNDC (energía, potencia, transporte)
    costo_marginal_diario.parquet  costo marginal medio diario (se acumula entre descargas)
    calidad.json                faltantes, imputación y chequeos de coherencia
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta

import pandas as pd

from forecast_energia import config
from forecast_energia.data import cndc, processing, weather

log = logging.getLogger("forecast_energia.build")


def procesar_demanda(raw15: pd.DataFrame, tot: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Demanda horaria conciliada con la generación TOT (ver `processing.ajustar_a_tot`).

    El SIN resultante es igual a TOT: tiene una definición constante en todo el periodo y es la
    misma magnitud que pronostica el CNDC (PREV).
    """
    dem15 = raw15[raw15["unique_id"].isin(config.DEPARTAMENTO_CODIGOS)]
    dem15 = processing.interpolar_huecos_cortos(processing.completar_grilla(dem15))
    medido = processing.a_horario(dem15).pivot(index="ds", columns="unique_id", values="y")
    medido = medido.reindex(columns=config.DEPARTAMENTO_CODIGOS)
    tot = tot.reindex(medido.index)

    anomalo = processing.detectar_anomalias_departamentales(medido, tot)
    imputado = anomalo | medido.isna()
    ajustado = processing.ajustar_a_tot(
        processing.imputar_por_participacion(medido, tot, anomalo), tot
    )

    largo = pd.concat(
        {
            "y": ajustado.stack(future_stack=True),
            "y_medido": medido.stack(future_stack=True),
            "imputado": imputado.stack(future_stack=True),
        },
        axis=1,
    ).reset_index()
    sin = pd.DataFrame(
        {
            "unique_id": config.SIN_ID,
            "ds": ajustado.index,
            "y": ajustado.sum(axis=1, min_count=ajustado.shape[1]).to_numpy(),
            "y_medido": medido.sum(axis=1, min_count=medido.shape[1]).to_numpy(),
            "imputado": imputado.any(axis=1).to_numpy(),
        }
    )
    return pd.concat([largo, sin], ignore_index=True), dem15


def procesar_generacion(raw15: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gen15 = raw15[raw15["unique_id"].isin(config.GENERACION_SERIES)]
    gen15 = processing.interpolar_huecos_cortos(processing.completar_grilla(gen15))
    medido = gen15[gen15["unique_id"] != config.SERIE_PRONOSTICO_CNDC]
    prev = gen15[gen15["unique_id"] == config.SERIE_PRONOSTICO_CNDC]
    gen_h = pd.concat(
        [processing.a_horario(medido), processing.escalon_horario(prev)],
        ignore_index=True,
    )
    return gen_h, gen15


def guardar_precios(mensuales: pd.DataFrame, cm: pd.DataFrame) -> None:
    """Guarda los precios, acumulando con lo ya descargado."""
    for nombre, df, clave in [
        ("precios_mensuales", mensuales, "mes"),
        ("costo_marginal_diario", cm, "fecha"),
    ]:
        path = config.PROCESSED_DIR / f"{nombre}.parquet"
        existente = pd.read_parquet(path) if path.exists() else None
        cndc.acumular(df, existente, clave).to_parquet(path, index=False)


def _a_mb(path) -> float:
    return round(path.stat().st_size / 1e6, 3)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start", type=date.fromisoformat, default=config.CNDC_START_DATE)
    parser.add_argument("--end", type=date.fromisoformat, default=date.today() - timedelta(days=1))
    parser.add_argument("--offline", action="store_true", help="no descarga; reprocesa la caché")
    parser.add_argument("--sin-clima", action="store_true", help="omite la descarga de clima")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.offline:
        raw_dem = cndc.leer_cache(config.ENDPOINT_DEMANDA)
        raw_gen = cndc.leer_cache(config.ENDPOINT_GENERACION)
    else:
        raw_dem = cndc.descargar(config.ENDPOINT_DEMANDA, args.start, args.end)
        raw_gen = cndc.descargar(config.ENDPOINT_GENERACION, args.start, args.end)

    gen_h, gen15 = procesar_generacion(raw_gen)
    tot = gen_h[gen_h["unique_id"] == "TOT"].set_index("ds")["y"]
    dem_h, dem15 = procesar_demanda(raw_dem, tot)

    salidas = {
        "demanda_horaria": dem_h,
        "generacion_horaria": gen_h,
    }

    if not args.sin_clima:
        inicio = dem_h["ds"].min().date()
        # Incluye mañana: el pronóstico del día siguiente ya está disponible.
        fin = max(dem_h["ds"].max().date(), date.today()) + timedelta(days=1)
        salidas["clima_horario"] = weather.descargar_clima(inicio, fin)

    for nombre, df in salidas.items():
        df.sort_values(["unique_id", "ds"]).to_parquet(
            config.PROCESSED_DIR / f"{nombre}.parquet", index=False
        )

    if not args.offline:
        guardar_precios(*cndc.descargar_precios(args.start))

    sin_df = dem_h[dem_h["unique_id"] == config.SIN_ID].set_index("ds")
    sin, sin_medido = sin_df["y"], sin_df["y_medido"]
    tot = tot.reindex(sin.index)
    cambio = sin.index >= config.CNDC_CAMBIO_DEFINICION_DEMANDA
    prev = (
        gen_h[gen_h["unique_id"] == config.SERIE_PRONOSTICO_CNDC]
        .set_index("ds")["y"]
        .reindex(sin.index)
    )
    calidad = {
        "generado": pd.Timestamp.now().isoformat(timespec="seconds"),
        "rango": [str(dem_h["ds"].min()), str(dem_h["ds"].max())],
        "demanda_15min": processing.resumen_calidad(dem15).astype(str).to_dict(orient="index"),
        "generacion_15min": processing.resumen_calidad(gen15).astype(str).to_dict(orient="index"),
        "demanda_horaria": processing.resumen_calidad(dem_h).astype(str).to_dict(orient="index"),
        "chequeos": {
            "ratio_sin_medido_vs_tot_antes_cambio": processing.ratio(
                sin_medido[~cambio], tot[~cambio]
            ),
            "ratio_sin_medido_vs_tot_desde_cambio": processing.ratio(
                sin_medido[cambio], tot[cambio]
            ),
            "ratio_sin_conciliado_vs_tot": processing.ratio(sin, tot),
            "pct_horas_depto_imputadas": round(
                100 * float(dem_h.loc[dem_h["unique_id"] != config.SIN_ID, "imputado"].mean()), 3
            ),
            "ratio_prev_vs_generacion_tot": processing.ratio(prev, tot),
        },
        "tamanos_mb": {p.name: _a_mb(p) for p in sorted(config.PROCESSED_DIR.glob("*.parquet"))},
    }
    (config.PROCESSED_DIR / "calidad.json").write_text(
        json.dumps(calidad, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("Rango: %s → %s", *calidad["rango"])
    log.info("Chequeos: %s", calidad["chequeos"])
    log.info("Tamaños (MB): %s", calidad["tamanos_mb"])


if __name__ == "__main__":
    main()
