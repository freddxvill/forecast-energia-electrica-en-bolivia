"""Impacto de negocio del pronóstico de producción frente al pronóstico oficial del CNDC (PREV).

Uso:
    uv run python -m forecast_energia.pipelines.impacto

Lee `artifacts/intervalos_SIN.parquet` (de `pipelines.intervalos`) y los precios del CNDC
(`data/processed/precios_mensuales.parquet`, `costo_marginal_diario.parquet`) y escribe:
    reports/impacto.json                       todos los KPIs + supuestos
    reports/impacto.md                         tablas listas para el README
    reports/figures/impacto_reserva_por_hora.png  reserva al alza: regla fija vs conformal

KPIs (periodo de test; si hay días posteriores a `config.TEST_FIN`, también el prospectivo):
1. Precisión: MAE y MWh de desvío evitados por año.
2. Valor económico de esos MWh: tarifa regulada y costo de oportunidad del gas.
3. Reserva al alza para cubrir el 95 % de los desvíos: regla fija sobre PREV vs conformal.
4. Confiabilidad de los intervalos: cobertura global, por hora y por mes.
"""

from __future__ import annotations

import argparse
import json
import logging

import matplotlib.pyplot as plt
import pandas as pd

from forecast_energia import config, viz
from forecast_energia import evaluation as E
from forecast_energia import impacto as I
from forecast_energia.data import io
from forecast_energia.models import baselines, combinacion, ensemble
from forecast_energia.pipelines.intervalos import METODO_PRODUCCION

log = logging.getLogger("forecast_energia.impacto")

ARTIFACTS = config.ROOT / "artifacts"
REPORTS = config.ROOT / "reports"
PREV = baselines.NOMBRE_PREV
PREV_CP = "PREV + CP"
# Pronósticos puntuales, de menor a mayor sofisticación: muestra de dónde sale la mejora.
PUNTUALES = {
    "PREV (CNDC)": PREV,
    "PREV corregido por hora": f"{PREV} corregido",
    "Ensamble (LightGBM + TFT)": ensemble.NOMBRE,
    "Combinado (Ensamble + PREV corregido)": combinacion.NOMBRE,
}
S = config.SUPUESTOS_IMPACTO
REGLA_FIJA = "PREV + regla fija"
REGLA_ORACULO = "PREV + regla fija oráculo (misma cobertura)"
RESERVA_PRODUCCION = "Combinado + conformal por hora + ACI"


def _nivel_intervalo() -> int:
    """El límite superior del intervalo central (2·nivel − 100) % es el cuantil unilateral."""
    return 2 * S["nivel_reserva"]["central"] - 100


def precision(df: pd.DataFrame, dias: int) -> list[dict]:
    t = df.set_index("ds")
    filas = []
    for etiqueta, col in PUNTUALES.items():
        m = E.metricas_puntuales(t["y"], t[col])
        ev = I.desvio_evitado(t["y"], t[PREV], t[col])
        fila = {
            "pronostico": etiqueta,
            "MWh_error_anio": I.anualizar(float((t["y"] - t[col]).abs().sum()), dias),
            "MAE_MW": m["MAE"],
            "MAPE_pct": m["MAPE"],
            "delta_MAE_MW": m["MAE"] - E.metricas_puntuales(t["y"], t[PREV])["MAE"],
            "MWh_evitados": float(ev.sum()),
            "MWh_evitados_anio": I.anualizar(float(ev.sum()), dias),
        }
        fila["delta_MAE_pct"] = 100 * fila["delta_MAE_MW"] / (fila["MAE_MW"] - fila["delta_MAE_MW"])
        if col != PREV:
            fila["p_DM_vs_PREV"] = E.diebold_mariano(t["y"], t[col], t[PREV])["p_valor"]
        filas.append(fila)
    return filas


def economico(test: pd.DataFrame, dias: int) -> dict:
    ev = I.desvio_evitado(test["y"], test[PREV], test[combinacion.NOMBRE])
    mwh_anio = I.anualizar(float(ev.sum()), dias)
    mensuales = io.cargar_precios_mensuales()
    p_energia = I.precio_por_hora(test["ds"], mensuales, "precio_energia")
    regulado = I.valor_desvios(ev, p_energia)

    cm = io.cargar_costo_marginal_diario().set_index("fecha")["costo_marginal"]
    cm_hora = test["ds"].dt.normalize().map(cm)
    con_cm = cm_hora.notna()
    precio_cm = I.valor_desvios(ev[con_cm], cm_hora[con_cm]) / float(ev[con_cm].sum())

    gas_regulado = S["precio_gas_regulado_usd_mpc"]["central"] / S["mmbtu_por_mpc"]["central"]
    hr, pg = S["heat_rate_mmbtu_mwh"], S["precio_gas_oportunidad_usd_mmbtu"]
    escenarios = {}
    for k in ("bajo", "central", "alto"):
        costo = I.costo_combustible_usd_mwh(hr[k], pg[k])
        escenarios[k] = {
            "heat_rate_mmbtu_mwh": hr[k],
            "precio_gas_usd_mmbtu": pg[k],
            "costo_usd_mwh": costo,
            "usd_anio": mwh_anio * costo,
        }
    return {
        "MWh_evitados_anio": mwh_anio,
        "tarifa_regulada": {
            "precio_energia_medio_usd_mwh": regulado / float(ev.sum()),
            "usd_anio": I.anualizar(regulado, dias),
            "fuente": "CNDC dashboard/precios: precio de energía mensual (monómico)",
        },
        "costo_marginal": {
            "precio_medio_usd_mwh": precio_cm,
            "usd_anio": mwh_anio * precio_cm,
            "pct_horas_con_dato": float(100 * con_cm.mean()),
            "fuente": "CNDC cm-diario: costo marginal medio diario",
        },
        "costo_gas": escenarios,
        "chequeo_heat_rate": {
            "costo_marginal_mediano_usd_mwh": float(cm_hora.median()),
            "precio_gas_regulado_usd_mmbtu": gas_regulado,
            "heat_rate_implicito_max": I.heat_rate_implicito(float(cm_hora.median()), gas_regulado),
        },
    }


def reserva(calib: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    nivel = S["nivel_reserva"]["central"]
    hi = _nivel_intervalo()
    r_fija = I.reserva_fija(calib["y"], calib[PREV], nivel)
    limites = {
        REGLA_FIJA: (PREV, test[PREV] + r_fija),
        "PREV + conformal por hora": (PREV, test[f"{PREV_CP}-hi-{hi}"]),
        RESERVA_PRODUCCION: (combinacion.NOMBRE, test[f"{METODO_PRODUCCION}-hi-{hi}"]),
    }
    reglas = {k: I.resumen_reserva(test["y"], test[c], lim) for k, (c, lim) in limites.items()}
    # Cota conservadora: la regla fija calibrada en el propio test con la misma cobertura que
    # logra el método de producción (le da ventaja a la regla fija).
    r_oraculo = I.reserva_fija_con_cobertura(
        test["y"], test[PREV], reglas[RESERVA_PRODUCCION]["cobertura_pct"]
    )
    limites[REGLA_ORACULO] = (PREV, test[PREV] + r_oraculo)
    reglas[REGLA_ORACULO] = I.resumen_reserva(test["y"], test[PREV], limites[REGLA_ORACULO][1])

    p_potencia = float(
        I.precio_por_hora(test["ds"], io.cargar_precios_mensuales(), "precio_potencia").mean()
    )
    prod = reglas[RESERVA_PRODUCCION]
    for base in (REGLA_FIJA, REGLA_ORACULO):
        ahorro = reglas[base]["reserva_media_MW"] - prod["reserva_media_MW"]
        prod[f"ahorro_MW_vs_{base}"] = ahorro
        prod[f"ahorro_usd_anio_vs_{base}"] = I.valor_reserva_usd_anio(ahorro, p_potencia)

    hora = test["ds"].dt.hour
    por_hora = pd.DataFrame(
        {k: (lim - test[c]).groupby(hora).mean() for k, (c, lim) in limites.items()}
    )
    return {
        "nivel_pct": nivel,
        "precio_potencia_usd_kw_mes": p_potencia,
        "reglas": reglas,
    }, por_hora


def confiabilidad(test: pd.DataFrame) -> dict:
    prob = E.metricas_probabilisticas(test, METODO_PRODUCCION)
    out = {f"cobertura_{n}": prob[f"cobertura_{n}"] for n in config.NIVELES}
    out["ancho_90_MW"] = prob["ancho_90"]
    hora = E.cobertura_por_grupo(test, METODO_PRODUCCION, 90, test["ds"].dt.hour)
    mes = E.cobertura_por_grupo(test, METODO_PRODUCCION, 90, test["ds"].dt.to_period("M"))
    out |= {
        "cobertura_90_peor_hora": float(hora.min()),
        "peor_hora": int(hora.idxmin()),
        "cobertura_90_peor_mes": float(mes.min()),
        "peor_mes": str(mes.idxmin()),
        "dentro_de_2pp": bool(abs(out["cobertura_90"] - 90) <= 2),
    }
    return out


def _filtrar(cv: pd.DataFrame, periodo: str) -> pd.DataFrame:
    cols = ["y", *PUNTUALES.values(), f"{METODO_PRODUCCION}-hi-90", f"{PREV_CP}-hi-90"]
    return cv[cv["periodo"] == periodo].dropna(subset=cols).reset_index(drop=True)


def calcular(cv: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    particion = E.particion_por_defecto(cv["ds"].max())
    test = _filtrar(cv, "test")
    calib = cv[cv["periodo"] == "calibracion"].dropna(subset=["y", PREV])
    res_reserva, por_hora = reserva(calib, test)
    out = {
        "periodo": {
            "test": [str(particion.test_inicio.date()), str(particion.fin_test.date())],
            "dias_test": particion.dias_test,
            "horas_evaluadas": len(test),
            "metodo_produccion": METODO_PRODUCCION,
            "referencia": "PREV (predespacho oficial del CNDC)",
        },
        "precision": precision(test, particion.dias_test),
        "economico": economico(test, particion.dias_test),
        "reserva": res_reserva,
        "confiabilidad": confiabilidad(test),
        "prospectivo": None,
        "supuestos": S,
    }
    prosp = _filtrar(cv, "prospectivo")
    if particion.dias_prospectivo and len(prosp):
        out["prospectivo"] = {
            "periodo": [str(prosp["ds"].min().date()), str(prosp["ds"].max().date())],
            "dias": int(prosp["ds"].dt.normalize().nunique()),
            "precision": precision(prosp, int(prosp["ds"].dt.normalize().nunique())),
            "confiabilidad": confiabilidad(prosp),
        }
    return out, por_hora


def _usd_m(x: float) -> str:
    return f"{x / 1e6:,.1f} M USD"


def markdown(r: dict) -> str:
    p = r["periodo"]
    prec = {f["pronostico"]: f for f in r["precision"]}
    base, comb = prec["PREV (CNDC)"], prec["Combinado (Ensamble + PREV corregido)"]
    eco, res, conf = r["economico"], r["reserva"], r["confiabilidad"]
    lineas = [
        f"Periodo de test: {p['test'][0]} → {p['test'][1]} ({p['dias_test']} días, "
        f"{p['horas_evaluadas']:,} horas). Referencia: {p['referencia']}.",
        "",
        "| KPI | PREV (CNDC) | Este proyecto | Mejora |",
        "|---|---|---|---|",
        f"| Error medio (MAE) | {base['MAE_MW']:.1f} MW | "
        f"{comb['MAE_MW']:.1f} MW | **{comb['delta_MAE_pct']:+.0f} %** |",
        f"| Energía mal programada (Σ\\|error\\|) | {base['MWh_error_anio'] / 1e3:,.0f} GWh/año | "
        f"{comb['MWh_error_anio'] / 1e3:,.0f} GWh/año | "
        f"**{-comb['MWh_evitados_anio'] / 1e3:,.0f} GWh/año** |",
    ]
    fija, cp = res["reglas"][REGLA_FIJA], res["reglas"][RESERVA_PRODUCCION]
    lineas += [
        f"| Reserva al alza para cubrir el {res['nivel_pct']} % de los desvíos | "
        f"{fija['reserva_media_MW']:.0f} MW fijos (cubre {fija['cobertura_pct']:.1f} %) | "
        f"{cp['reserva_media_MW']:.0f} MW de media, variable por hora (cubre "
        f"{cp['cobertura_pct']:.1f} %) | **{-cp[f'ahorro_MW_vs_{REGLA_FIJA}']:+.0f} MW** "
        f"({-cp[f'ahorro_MW_vs_{REGLA_ORACULO}']:+.0f} MW a igual cobertura) |",
        f"| Cobertura del intervalo 90 % | sin intervalos | {conf['cobertura_90']:.1f} % "
        f"(peor hora {conf['cobertura_90_peor_hora']:.0f} %, peor mes "
        f"{conf['cobertura_90_peor_mes']:.0f} %) | nominal ±2 pp: "
        f"{'sí' if conf['dentro_de_2pp'] else 'no'} |",
        "",
        "**Valor económico estimado (orden de magnitud, USD/año):**",
        "",
        "| Escenario | USD/MWh | Desvíos evitados | Reserva liberada |",
        "|---|---|---|---|",
        f"| Tarifa regulada (precio de energía del CNDC) | "
        f"{eco['tarifa_regulada']['precio_energia_medio_usd_mwh']:.1f} | "
        f"{_usd_m(eco['tarifa_regulada']['usd_anio'])} | "
        f"{_usd_m(cp[f'ahorro_usd_anio_vs_{REGLA_FIJA}'])} "
        f"({_usd_m(cp[f'ahorro_usd_anio_vs_{REGLA_ORACULO}'])} a igual cobertura; "
        f"{res['precio_potencia_usd_kw_mes']:.1f} USD/kW-mes) |",
    ]
    for k, e in eco["costo_gas"].items():
        lineas.append(
            f"| Costo de oportunidad del gas ({k}: {e['heat_rate_mmbtu_mwh']:g} MMBtu/MWh × "
            f"{e['precio_gas_usd_mmbtu']:g} USD/MMBtu) | {e['costo_usd_mwh']:.0f} | "
            f"{_usd_m(e['usd_anio'])} | — |"
        )
    lineas += [
        "",
        "**De dónde sale la mejora** (MAE de test y MWh evitados frente a PREV):",
        "",
        "| Pronóstico | MAE (MW) | MAPE | Δ vs PREV | GWh/año evitados | DM p-valor |",
        "|---|---|---|---|---|---|",
    ]
    for f in r["precision"]:
        es_base = "p_DM_vs_PREV" not in f
        pval = "—" if es_base else f"{f['p_DM_vs_PREV']:.3f}"
        delta = "—" if es_base else f"{f['delta_MAE_pct']:+.0f} %"
        gwh = "—" if es_base else f"{f['MWh_evitados_anio'] / 1e3:,.0f}"
        lineas.append(
            f"| {f['pronostico']} | {f['MAE_MW']:.1f} | {f['MAPE_pct']:.2f} % | "
            f"{delta} | {gwh} | {pval} |"
        )
    lineas += [
        "",
        "**Reserva al alza** (MW sobre el pronóstico para cubrir el "
        f"{res['nivel_pct']} % de las horas con demanda mayor a la pronosticada):",
        "",
        "| Regla | Reserva media (MW) | Reserva máx. (MW) | Cobertura en test |",
        "|---|---|---|---|",
    ]
    for k, v in res["reglas"].items():
        lineas.append(
            f"| {k} | {v['reserva_media_MW']:.0f} | {v['reserva_max_MW']:.0f} | "
            f"{v['cobertura_pct']:.1f} % |"
        )
    hr, cm = eco["chequeo_heat_rate"], eco["costo_marginal"]
    lineas += [
        "",
        "**Supuestos y fuentes:**",
        "",
        "| Supuesto | Valor (bajo / central / alto) | Fuente |",
        "|---|---|---|",
    ]
    for v in r["supuestos"].values():
        valores = " / ".join(f"{v[k]:g}" for k in ("bajo", "central", "alto") if k in v)
        lineas.append(f"| {v['descripcion']} | {valores} | {v['fuente']} |")
    lineas += [
        "",
        f"- Chequeo del heat rate: el costo marginal mediano del test "
        f"({hr['costo_marginal_mediano_usd_mwh']:.1f} USD/MWh) dividido por el precio regulado "
        f"del gas ({hr['precio_gas_regulado_usd_mmbtu']:.2f} USD/MMBtu) acota el consumo "
        f"específico de la unidad marginal en ≤ {hr['heat_rate_implicito_max']:.1f} MMBtu/MWh "
        "(incluye O&M), coherente con el rango supuesto.",
        f"- Con el costo marginal diario del CNDC ({cm['precio_medio_usd_mwh']:.1f} USD/MWh "
        f"ponderado; dato en el {cm['pct_horas_con_dato']:.0f} % de las horas) los desvíos "
        f"evitados valen {_usd_m(cm['usd_anio'])}/año, similar a la tarifa regulada.",
        "- Anualizado como total del test × 365 / días de test. El test cubre abril–octubre "
        "(estación seca); no incluye el verano lluvioso.",
        "- La reserva liberada se valoriza al precio de potencia (pago por capacidad firme) como "
        "referencia del costo de mantener capacidad disponible.",
        "- Valorizar cada MWh de desvío supone que se cubre en tiempo real con generación "
        "térmica de ajuste. No es un ahorro contable del CNDC: es la escala del beneficio.",
    ]
    if r["prospectivo"]:
        pr = r["prospectivo"]
        c = {f["pronostico"]: f for f in pr["precision"]}
        lineas += [
            "",
            f"**Validación prospectiva** ({pr['periodo'][0]} → {pr['periodo'][1]}, "
            f"{pr['dias']} días que no se miraron al diseñar los modelos): MAE PREV "
            f"{c['PREV (CNDC)']['MAE_MW']:.1f} MW vs Combinado "
            f"{c['Combinado (Ensamble + PREV corregido)']['MAE_MW']:.1f} MW; cobertura 90 % "
            f"{pr['confiabilidad']['cobertura_90']:.1f} %.",
        ]
    return "\n".join(lineas) + "\n"


def figura_reserva(por_hora: pd.DataFrame) -> None:
    series = {
        REGLA_FIJA: ("PREV + regla fija (calibración)", viz.COLOR_PREV, None),
        RESERVA_PRODUCCION: ("Combinado + conformal por hora", viz.COLOR_MODELO, "o"),
    }
    fig, ax = plt.subplots(figsize=(9, 4))
    for col, (etiqueta, color, marcador) in series.items():
        s = por_hora[col]
        ax.plot(s.index, s.values, color=color, lw=2, marker=marcador, ms=4, label=etiqueta)
        ax.annotate(
            f"{s.mean():.0f} MW de media",
            (s.index[-1], s.iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=viz.TINTA_2,
        )
    ax.set(
        title="Reserva al alza para cubrir el 95 % de los desvíos, por hora (test)",
        xlabel="Hora",
        ylabel="MW sobre el pronóstico",
        xticks=range(0, 24, 3),
        xlim=(-0.5, 27),
        ylim=(0, 135),
    )
    ax.legend(loc="lower right")
    viz.guardar(fig, "impacto_reserva_por_hora")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cv = pd.read_parquet(ARTIFACTS / f"intervalos_{config.SIN_ID}.parquet")
    resultado, por_hora = calcular(cv)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "impacto.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = markdown(resultado)
    (REPORTS / "impacto.md").write_text(md, encoding="utf-8")
    viz.estilo()
    figura_reserva(por_hora)
    print(md)


if __name__ == "__main__":
    main()
