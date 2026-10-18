"""Demo pública: pronóstico probabilístico de la demanda eléctrica de Bolivia.

Uso local:
    uv run --group app streamlit run app/streamlit_app.py

Fuentes, en orden: la API (`API_URL` en los secrets de Streamlit o en el entorno); si no
responde (p.ej. el Space de Hugging Face está dormido), los artefactos que el cron diario
publica en la rama `demo-data` (`DEMO_URL`); y si tampoco, la copia de `artifacts/demo/` del
repo. La demo nunca queda en blanco. Es autocontenida (no importa `forecast_energia`) para que
Streamlit Cloud la instale solo con `app/requirements.txt`.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
DEMO = RAIZ / "artifacts" / "demo"
FIGURAS = RAIZ / "reports" / "figures"
REPO_URL = "https://github.com/freddxvill/forecast-energia-electrica-en-bolivia"
DEMO_URL_DEFECTO = "https://raw.githubusercontent.com/freddxvill/forecast-energia-electrica-en-bolivia/demo-data/artifacts/demo"
PUNTO = "Pronóstico"
NIVELES = (80, 90, 95)
UNIDADES = [
    "SIN",
    "SANTA CRUZ",
    "LA PAZ",
    "COCHABAMBA",
    "POTOSI",
    "ORURO",
    "TARIJA",
    "CHUQUISACA",
    "BENI",
]
# Paleta del proyecto (validada para daltonismo, ver src/forecast_energia/viz.py).
AZUL, NARANJA, TINTA, TINTA_2 = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
DIAS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

st.set_page_config(page_title="Demanda eléctrica de Bolivia", page_icon="⚡", layout="wide")


# --- Datos: API, rama demo-data o copia local ---------------------------------------------------


def _ajuste(nombre: str, defecto: str = "") -> str:
    try:
        valor = st.secrets.get(nombre)
    except Exception:  # sin secrets.toml
        valor = None
    return (valor or os.environ.get(nombre) or defecto).rstrip("/")


def _archivo(nombre: str) -> tuple[bytes, str]:
    """Artefacto desde la rama `demo-data` (actualizada a diario) o, si falla, el local."""
    try:
        r = requests.get(f"{_ajuste('DEMO_URL', DEMO_URL_DEFECTO)}/{nombre}", timeout=8)
        r.raise_for_status()
        return r.content, "GitHub (demo-data)"
    except requests.RequestException:
        return (DEMO / nombre).read_bytes(), "copia local"


def _parquet(nombre: str) -> tuple[pd.DataFrame, str]:
    contenido, fuente = _archivo(nombre)
    return pd.read_parquet(io.BytesIO(contenido)), fuente


def _get(ruta: str, **params) -> dict | None:
    url = _ajuste("API_URL")
    if not url:
        return None
    try:
        r = requests.get(f"{url}{ruta}", params=params, timeout=6)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def _desde_api(horas: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(horas)
    df["ds"] = pd.to_datetime(df["ds"])
    nombres = {"punto": PUNTO} | {
        f"{lado}_{n}": f"{PUNTO}-{lado}-{n}" for n in NIVELES for lado in ("lo", "hi")
    }
    return df.rename(columns=nombres)


@st.cache_data(ttl=600, show_spinner=False)
def cargar_pronostico(unidad: str) -> tuple[pd.DataFrame, str]:
    """Pronóstico de los días posteriores al corte para una unidad, y la fuente usada."""
    salud = _get("/health")
    if salud:
        partes = [_get("/forecast", unidad=unidad, fecha=d) for d in salud["dias"]]
        if all(partes):
            df = pd.concat([_desde_api(p["horas"]) for p in partes], ignore_index=True)
            return df.assign(corte=partes[0]["corte"], emitido=partes[0]["emitido"]), "API"
    df, fuente = _parquet("pronostico_actual.parquet")
    return df[df["unique_id"] == unidad].reset_index(drop=True), fuente


@st.cache_data(ttl=600, show_spinner=False)
def cargar_backtest(unidad: str) -> pd.DataFrame:
    respuesta = _get("/backtest", unidad=unidad)
    if respuesta:
        return _desde_api(respuesta["horas"])
    df, _ = _parquet("backtest_reciente.parquet")
    return df[df["unique_id"] == unidad].reset_index(drop=True)


@st.cache_data(ttl=600, show_spinner=False)
def cargar_metricas() -> dict:
    return _get("/metrics") or json.loads(_archivo("metricas.json")[0].decode("utf-8"))


# --- Gráficos ---------------------------------------------------------------------------------


def _banda(fig: go.Figure, df: pd.DataFrame, nivel: int, opacidad: float) -> None:
    fig.add_trace(
        go.Scatter(
            x=pd.concat([df["ds"], df["ds"][::-1]]),
            y=pd.concat([df[f"{PUNTO}-hi-{nivel}"], df[f"{PUNTO}-lo-{nivel}"][::-1]]),
            fill="toself",
            fillcolor=f"rgba(42,120,214,{opacidad})",
            line={"width": 0},
            hoverinfo="skip",
            name=f"Intervalo {nivel} %",
        )
    )


def grafico(df: pd.DataFrame, niveles: tuple[int, ...], real: bool = False) -> go.Figure:
    fig = go.Figure()
    for nivel, opacidad in zip(niveles, (0.15, 0.30), strict=False):
        _banda(fig, df, nivel, opacidad)
    if "PREV" in df and df["PREV"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df["ds"],
                y=df["PREV"],
                name="Oficial CNDC (PREV)",
                line={"color": NARANJA, "width": 2, "dash": "dot"},
            )
        )
    fig.add_trace(
        go.Scatter(x=df["ds"], y=df[PUNTO], name="Pronóstico", line={"color": AZUL, "width": 2})
    )
    if real:
        fig.add_trace(
            go.Scatter(x=df["ds"], y=df["y"], name="Real", line={"color": TINTA, "width": 2})
        )
    fig.update_layout(
        height=420,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        hovermode="x unified",
        yaxis_title="MW",
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    fig.update_traces(hovertemplate="%{y:,.0f} MW")
    return fig


def nombre(unidad: str) -> str:
    return "SIN (total nacional)" if unidad == "SIN" else unidad.title()


def _fecha_es(ts: pd.Timestamp) -> str:
    return f"{DIAS_ES[ts.dayofweek]} {ts.day:02d}/{ts.month:02d}/{ts.year}"


# --- Pestañas ---------------------------------------------------------------------------------


def pestaña_pronostico() -> None:
    c1, c2 = st.columns([1, 2])
    unidad = c1.selectbox("Sistema o departamento", UNIDADES, format_func=nombre)
    df, fuente = cargar_pronostico(unidad)
    dias = sorted(df["ds"].dt.normalize().unique())
    dia = c2.radio(
        "Día",
        dias,
        index=len(dias) - 1,
        format_func=lambda d: ("Mañana: " if d == dias[-1] else "Hoy: ") + _fecha_es(d),
        horizontal=True,
    )
    d = df[df["ds"].dt.normalize() == dia]
    pico = d.loc[d[PUNTO].idxmax()]
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Pico pronosticado ({pico['ds']:%H:%M})", f"{pico[PUNTO]:,.0f} MW")
    k2.metric("Energía del día", f"{d[PUNTO].sum() / 1e3:,.1f} GWh")
    ancho = (d[f"{PUNTO}-hi-90"] - d[f"{PUNTO}-lo-90"]).mean()
    k3.metric(
        "Incertidumbre (IC 90 %)",
        f"± {ancho / 2:,.0f} MW",
        help="Mitad del ancho medio del intervalo del 90 %.",
    )
    st.plotly_chart(grafico(d, (95, 80)), use_container_width=True)
    st.caption(
        f"Corte de datos: {pd.Timestamp(df['corte'].iloc[0]):%d/%m %H:%M} · emitido "
        f"{pd.Timestamp(df['emitido'].iloc[0]):%d/%m %H:%M} (hora de Bolivia) · fuente: {fuente}. "
        "Bandas: intervalos conformal del 80 % y 95 %."
    )
    with st.expander("Ver tabla"):
        cols = ["ds", PUNTO] + [f"{PUNTO}-{lado}-{n}" for n in NIVELES for lado in ("lo", "hi")]
        cols += ["PREV"] if unidad == "SIN" else []
        st.dataframe(d[cols].round(0), hide_index=True, use_container_width=True)


def pestaña_backtest() -> None:
    unidad = st.selectbox("Serie", UNIDADES, format_func=nombre, key="bt_unidad")
    bt = cargar_backtest(unidad).dropna(subset=["y"])
    fin = bt["ds"].max().normalize()
    desde, hasta = st.slider(
        "Rango",
        min_value=bt["ds"].min().date(),
        max_value=fin.date(),
        value=((fin - pd.Timedelta(days=6)).date(), fin.date()),
        format="DD/MM",
    )
    sel = bt[
        (bt["ds"] >= pd.Timestamp(desde)) & (bt["ds"] < pd.Timestamp(hasta) + pd.Timedelta(days=1))
    ]
    e = (sel[PUNTO] - sel["y"]).abs()
    dentro = (sel["y"] >= sel[f"{PUNTO}-lo-90"]) & (sel["y"] <= sel[f"{PUNTO}-hi-90"])
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Error medio (MAPE {100 * (e / sel['y']).mean():.2f} %)", f"{e.mean():,.1f} MW")
    if unidad == "SIN" and sel["PREV"].notna().any():
        e_prev = (sel["PREV"] - sel["y"]).abs()
        mejora = 100 * (e.mean() / e_prev.mean() - 1)
        k2.metric(
            "MAE oficial CNDC",
            f"{e_prev.mean():,.1f} MW",
            f"{mejora:+.0f} % con este modelo",
            "inverse",
        )
    k3.metric("Cobertura del IC 90 % (nominal 90 %)", f"{100 * dentro.mean():.0f} %")
    st.plotly_chart(grafico(sel, (90,), real=True), use_container_width=True)
    st.caption(
        "Pronósticos day-ahead emitidos con datos hasta D−2 (los mismos que publicaría el "
        "sistema ese día; sin mirar el futuro)."
    )


def pestaña_incertidumbre(m: dict) -> None:
    st.markdown(
        "Los intervalos se calibran con **conformal prediction** por hora del día y se ajustan "
        "con **ACI** cuando la cobertura se desvía. Así valen hora a hora, no solo en promedio."
    )
    if (FIGURAS / "cp_cobertura_por_hora.png").exists():
        st.image(str(FIGURAS / "cp_cobertura_por_hora.png"))
    cob = pd.DataFrame(m.get("test_jerarquia_cobertura", []))
    if not cob.empty:
        cob = cob.rename(columns={"unique_id": "Serie", "nombre": "Serie"})
        st.markdown("**Cobertura en test (2026-04-17 → 2026-10-16) por serie**")
        st.dataframe(cob.round(1), hide_index=True, use_container_width=True)
    seg = m.get("seguimiento_30_dias") or {}
    if seg:
        st.markdown(
            f"**Últimos 30 días** ({seg['desde']} → {seg['hasta']}): MAE {seg['MAE']:.1f} MW "
            f"vs {seg['MAE_PREV']:.1f} MW del CNDC; cobertura 80/90/95 = "
            f"{seg['cobertura_80']:.0f} / {seg['cobertura_90']:.0f} / {seg['cobertura_95']:.0f} %."
        )


def pestaña_impacto(m: dict) -> None:
    imp = m.get("impacto") or {}
    if not imp:
        st.info("Todavía no hay métricas de impacto.")
        return
    prec = {f["pronostico"]: f for f in imp["precision"]}
    base = prec["PREV (CNDC)"]
    comb = prec["Combinado (Ensamble + PREV corregido)"]
    res = imp["reserva"]["reglas"]
    prod = res["Combinado + conformal por hora + ACI"]
    eco = imp["economico"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        f"Error vs oficial ({base['MAE_MW']:.1f} → {comb['MAE_MW']:.1f} MW)",
        f"{comb['delta_MAE_pct']:+.0f} %",
    )
    k2.metric("Energía mal programada evitada", f"{comb['MWh_evitados_anio'] / 1e3:,.0f} GWh/año")
    k3.metric(
        "Reserva al alza (95 % de los desvíos)",
        f"{-prod['ahorro_MW_vs_PREV + regla fija']:+.0f} MW",
        help="Frente a una reserva fija sobre el pronóstico oficial.",
    )
    bajo, alto = eco["tarifa_regulada"]["usd_anio"], eco["costo_gas"]["alto"]["usd_anio"]
    k4.metric(
        "Valor estimado",
        f"{bajo / 1e6:.0f}–{alto / 1e6:.0f} M USD/año",
        help="De la tarifa regulada al costo de oportunidad del gas exportable.",
    )
    if (FIGURAS / "impacto_reserva_por_hora.png").exists():
        st.image(str(FIGURAS / "impacto_reserva_por_hora.png"))
    st.caption(
        "Órdenes de magnitud con supuestos explícitos (heat rate, precio de exportación del gas, "
        "precio de potencia): ver reports/impacto.md en el repositorio."
    )


CNDC = "[CNDC](https://www.cndc.bo)"
OPEN_METEO = "[Open-Meteo Previous Runs](https://open-meteo.com/en/docs/previous-runs-api)"
# (fuente, qué se usa, resolución, desde)
DATOS = [
    (f"{CNDC} `rt/demanda`", "Demanda (MW) de los 8 departamentos", "15 min", "2024-03-01"),
    (
        f"{CNDC} `rt/generacion`",
        "Generación por tecnología y pronóstico oficial `PREV`",
        "15 min",
        "2024-03-01",
    ),
    (f"{CNDC} `precmp`", "Predespacho horario de mañana (`PREV` en producción)", "1 h", "~30 días"),
    (
        f"{CNDC} `dashboard/precios`, `cm-diario`",
        "Precios de energía y potencia; costo marginal diario",
        "mes / día",
        "2024-03",
    ),
    (OPEN_METEO, "Clima de las 8 capitales pronosticado 48 h antes", "1 h", "2024-03-01"),
    ("[`holidays`](https://pypi.org/project/holidays/)", "Feriados de Bolivia", "día", "—"),
]
TABLA_DATOS = "\n".join(
    ["| Fuente | Qué se usa | Resolución | Desde |", "|---|---|---|---|"]
    + [f"| {' | '.join(fila)} |" for fila in DATOS]
)

FUENTES = f"""
**Datos**

{TABLA_DATOS}

- La API del CNDC (`https://www.cndc.bo/wp-json/cndc/v1`) es pública y sin clave, pero **no está
  documentada oficialmente**: los endpoints y su convención horaria se validaron a mano (por
  ejemplo, contra `rt/curva-dia`). El cron diario descarga los datos nuevos cada mañana.
- Open-Meteo: [datos del tiempo por Open-Meteo.com](https://open-meteo.com/), licencia
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Se usa el pronóstico hecho 48 h antes
  de cada hora porque es lo único que se conocía en el momento del corte (D−2).
- `holidays`: se excluye el 17 de octubre (Día de la Dignidad Nacional), que la librería marca como
  feriado pero en el que la demanda no cae.
- Supuestos del impacto: precio regulado del gas para generación (DS 26037), precio de exportación
  del gas a Brasil (~6–7 USD/MMBtu, OilPrice.com 2025) y heat rate típico de turbinas de ciclo
  abierto, acotado con el costo marginal del CNDC.

**Metodología, en cinco pasos**

1. **Limpieza:** los `-1` de la API son faltantes; se interpolan huecos de hasta 1 h y los
   departamentos se concilian con la generación total (la suma cuadra con el SIN).
2. **Pronóstico del día D con datos hasta D−2 23:00**, la misma anticipación que el predespacho.
3. **Modelos:** LightGBM y TFT globales sobre las 9 series; el SIN se combina con el pronóstico
   oficial corregido por su sesgo por hora.
4. **Coherencia:** reconciliación MinT para que la suma de los departamentos sea el SIN.
5. **Incertidumbre:** conformal prediction por hora del día + ACI, con los errores de los últimos
   91 días.

Más detalle: [glosario de métodos]({REPO_URL}/blob/main/docs/glosario.md) y
[código y reproducción]({REPO_URL}).
"""


def pestaña_fuentes() -> None:
    st.markdown(FUENTES)


def main() -> None:
    st.title("⚡ Demanda eléctrica de Bolivia: pronóstico probabilístico")
    st.markdown(
        "Pronóstico **day-ahead** de la demanda horaria del Sistema Interconectado Nacional y sus "
        "8 departamentos, con **intervalos con garantía estadística**. Datos reales del CNDC; "
        "**−30 % de error frente al pronóstico oficial** en 6 meses de test."
    )
    m = cargar_metricas()
    tabs = st.tabs(["Pronóstico", "Backtest", "Incertidumbre", "Impacto", "Fuentes y metodología"])
    with tabs[0]:
        pestaña_pronostico()
    with tabs[1]:
        pestaña_backtest()
    with tabs[2]:
        pestaña_incertidumbre(m)
    with tabs[3]:
        pestaña_impacto(m)
    with tabs[4]:
        pestaña_fuentes()


main()
