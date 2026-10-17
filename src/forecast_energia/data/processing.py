"""Limpieza de series de 15 min y agregación a frecuencia horaria (formato Nixtla)."""

from __future__ import annotations

import numpy as np
import pandas as pd

PASO = pd.Timedelta(minutes=15)


def completar_grilla(df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    """Reindexa cada serie a una grilla regular: los huecos quedan como NaN explícitos."""
    out = []
    for uid, g in df.groupby("unique_id", sort=True):
        idx = pd.date_range(g["ds"].min(), g["ds"].max(), freq=freq, name="ds")
        s = g.set_index("ds")["y"].reindex(idx)
        out.append(pd.DataFrame({"unique_id": uid, "ds": idx, "y": s.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def _largo_de_racha_nan(y: pd.Series) -> pd.Series:
    """Largo de la racha de NaN consecutivos a la que pertenece cada posición (0 si no es NaN)."""
    es_nan = y.isna()
    grupo = (es_nan != es_nan.shift()).cumsum()
    largo = es_nan.groupby(grupo).transform("sum")
    return largo.where(es_nan, 0).astype(int)


def interpolar_huecos_cortos(df: pd.DataFrame, max_pasos: int = 4) -> pd.DataFrame:
    """Interpola linealmente solo los huecos de hasta `max_pasos` valores (4 × 15 min = 1 h).

    Los huecos más largos quedan como NaN, porque inventarlos sesgaría los modelos. Agrega
    la columna booleana `imputado`.
    """
    partes = []
    for _, g in df.groupby("unique_id", sort=False):
        g = g.sort_values("ds").copy()
        racha = _largo_de_racha_nan(g["y"])
        interp = g["y"].interpolate(method="linear", limit_area="inside")
        rellenar = (racha > 0) & (racha <= max_pasos)
        g["imputado"] = rellenar & interp.notna()
        g["y"] = g["y"].where(~g["imputado"], interp)
        partes.append(g)
    return pd.concat(partes, ignore_index=True)


def a_horario(df: pd.DataFrame, min_validos: int = 3, closed: str = "right") -> pd.DataFrame:
    """Promedia las mediciones de 15 min de cada hora; `ds` marca el inicio de la hora.

    Con `closed="right"`, la hora 00:00 promedia 00:15, 00:30, 00:45 y 01:00, igual que la
    hora 1 del CNDC (hora final). La hora queda NaN si tiene menos de `min_validos` datos.
    """
    partes = []
    for uid, g in df.groupby("unique_id", sort=True):
        s = g.set_index("ds")["y"]
        r = s.resample("h", closed=closed, label="left")
        media = r.mean().where(r.count() >= min_validos)
        partes.append(pd.DataFrame({"unique_id": uid, "ds": media.index, "y": media.to_numpy()}))
    return pd.concat(partes, ignore_index=True)


def escalon_horario(df: pd.DataFrame) -> pd.DataFrame:
    """Valor horario de una serie escalonada (el pronóstico PREV del CNDC).

    El escalón de la hora hh dura de hh:00 a hh:45 o, en algunos meses (nov-2024 a
    mar-2025), de hh:15 a hh+1:00. En ambos casos contiene hh:30, así que se toma ese valor.
    """
    medio = df[df["ds"].dt.minute == 30]
    return pd.DataFrame(
        {
            "unique_id": medio["unique_id"].to_numpy(),
            "ds": medio["ds"].dt.floor("h").to_numpy(),
            "y": medio["y"].to_numpy(),
        }
    )


def mediana_misma_hora(ancho: pd.DataFrame, dias: int = 14, min_dias: int = 5) -> pd.DataFrame:
    """Mediana de los `dias` días previos a la misma hora (sin incluir el valor actual)."""
    return ancho.groupby(ancho.index.hour).transform(
        lambda x: x.shift(1).rolling(dias, min_periods=min_dias).median()
    )


def detectar_anomalias_departamentales(
    dep: pd.DataFrame,
    tot: pd.Series,
    tol_ratio: float = 0.05,
    min_frac: float = 0.4,
    max_frac: float = 2.5,
) -> pd.DataFrame:
    """Marca (True) las horas departamentales no confiables, con `TOT` como referencia externa.

    - Toda la hora se marca si Σ deptos / TOT se aleja más de `tol_ratio` de su nivel
      local (mediana móvil de 7 días). Así se detectan fallas de medición que afectan a varios
      departamentos a la vez, como el periodo oct–nov 2024.
    - Un departamento se marca si su valor cae fuera de [`min_frac`, `max_frac`] × su
      mediana de la misma hora en los 14 días previos.
    """
    tot = tot.reindex(dep.index)
    ratio = dep.sum(axis=1, min_count=dep.shape[1]) / tot
    nivel = ratio.rolling(7 * 24, center=True, min_periods=24).median()
    hora_mala = (ratio / nivel - 1).abs() > tol_ratio

    ref = mediana_misma_hora(dep)
    rel = dep / ref
    mascara = (rel < min_frac) | (rel > max_frac)
    return mascara.apply(lambda col: col | hora_mala)


def imputar_por_participacion(
    dep: pd.DataFrame, tot: pd.Series, mascara: pd.DataFrame
) -> pd.DataFrame:
    """Reemplaza los valores marcados o faltantes por participación típica × TOT.

    La participación de referencia es la mediana de la misma hora en los 14 días previos,
    calculada solo con datos válidos.
    """
    tot = tot.reindex(dep.index)
    invalido = mascara | dep.isna()
    participacion = (dep / tot.to_numpy()[:, None]).mask(invalido)
    ref = mediana_misma_hora(participacion).ffill().bfill()
    estimado = ref.mul(tot, axis=0)
    return dep.mask(invalido, estimado)


def ajustar_a_tot(dep: pd.DataFrame, tot: pd.Series) -> pd.DataFrame:
    """Reescala los departamentos para que su suma sea exactamente `TOT`.

    Hasta el 2026-04-30 el CNDC publicaba la demanda departamental sin pérdidas
    (Σ deptos ≈ 0.965 × TOT). Desde el 2026-05-01 Σ deptos = TOT. Al repartir la diferencia
    en proporción al consumo, las series quedan con una definición constante (demanda +
    pérdidas asignadas) y la jerarquía SIN = Σ deptos cierra exactamente.
    """
    factor = tot.reindex(dep.index) / dep.sum(axis=1, min_count=dep.shape[1])
    return dep.mul(factor, axis=0)


def resumen_calidad(df: pd.DataFrame) -> pd.DataFrame:
    """% de faltantes y rango temporal por serie."""
    g = df.groupby("unique_id")
    res = pd.DataFrame(
        {
            "inicio": g["ds"].min(),
            "fin": g["ds"].max(),
            "n": g.size(),
            "pct_faltante": g["y"].apply(lambda y: 100 * y.isna().mean()).round(3),
        }
    )
    if "imputado" in df:
        res["pct_imputado"] = g["imputado"].mean().mul(100).round(3)
    return res


def ratio(a: pd.Series, b: pd.Series) -> float:
    """Cociente de sumas sobre las posiciones donde ambas series tienen dato."""
    m = a.notna() & b.notna()
    return float(np.round(a[m].sum() / b[m].sum(), 4))
