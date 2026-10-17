"""Análisis exploratorio reutilizable: perfiles, feriados, clima, mix de generación y benchmark."""

from __future__ import annotations

import pandas as pd

from forecast_energia import config
from forecast_energia.data.calendar import feriados

TECNOLOGIAS = {"HIDRO": "Hidro", "TERMO": "Térmica", "EOL": "Eólica", "SOLAR": "Solar"}
OTRAS = ["BAGAZO", "RENO"]


def tipo_de_dia(ds: pd.DatetimeIndex) -> pd.Series:
    """'Laborable', 'Sábado' o 'Domingo/feriado' (feriados nacionales)."""
    dias = ds.normalize()
    nacional = pd.DatetimeIndex(sorted(feriados(sorted(set(ds.year))).keys()))
    tipo = pd.Series("Laborable", index=ds)
    tipo[ds.dayofweek == 5] = "Sábado"
    tipo[(ds.dayofweek == 6) | dias.isin(nacional)] = "Domingo/feriado"
    return tipo


def perfil_horario(s: pd.Series) -> pd.DataFrame:
    """Demanda media por hora (filas) y tipo de día (columnas)."""
    tipo = tipo_de_dia(s.index)
    return s.groupby([s.index.hour, tipo.to_numpy()]).mean().unstack()


def heatmap_hora_mes(s: pd.Series) -> pd.DataFrame:
    """Demanda media por hora del día (filas) y mes calendario (columnas)."""
    return s.groupby([s.index.hour, s.index.to_period("M")]).mean().unstack()


def diaria(s: pd.Series, min_horas: int = 22) -> pd.DataFrame:
    """Media y máximo diarios (días con al menos `min_horas` horas válidas)."""
    g = s.groupby(s.index.normalize())
    df = pd.DataFrame({"media": g.mean(), "max": g.max(), "n": g.count()})
    return df[df["n"] >= min_horas].drop(columns="n")


def efecto_feriados(s: pd.Series) -> pd.DataFrame:
    """Demanda de cada feriado nacional vs el mismo día de semana una semana antes y después.

    `pct` < 0 indica que el feriado consume menos que un día normal equivalente.
    """
    d = diaria(s)["media"]
    cal = feriados(sorted(set(d.index.year)))
    filas = []
    for fecha, nombre in sorted(cal.items()):
        f = pd.Timestamp(fecha)
        refs = [f - pd.Timedelta(days=7), f + pd.Timedelta(days=7)]
        refs = [r for r in refs if r in d.index and r.date() not in cal]
        if f in d.index and refs:
            base = d[refs].mean()
            filas.append({"fecha": f, "feriado": nombre, "pct": 100 * (d[f] / base - 1)})
    return pd.DataFrame(filas)


def sensibilidad_temperatura(dem: pd.Series, temp: pd.Series, ventana: int = 28) -> pd.DataFrame:
    """Anomalías diarias de demanda (%) y de temperatura (°C) respecto de su media móvil.

    Al restar la media móvil centrada se quitan la tendencia y la estacionalidad lenta, y
    queda la respuesta de corto plazo de la demanda al clima. Solo se usan días laborables,
    para no mezclar el efecto del fin de semana.
    """
    d = diaria(dem)["media"]
    t = temp.groupby(temp.index.normalize()).mean().reindex(d.index)
    laborable = tipo_de_dia(d.index).to_numpy() == "Laborable"
    d_base = d[laborable].rolling(ventana, center=True, min_periods=ventana // 2).mean()
    t_base = t[laborable].rolling(ventana, center=True, min_periods=ventana // 2).mean()
    out = pd.DataFrame(
        {
            "demanda_anom_pct": 100 * (d[laborable] / d_base - 1),
            "temp_anom": t[laborable] - t_base,
            "temp": t[laborable],
        }
    )
    return out.dropna()


def correlacion_clima_por_depto(dem: pd.DataFrame, clima: pd.DataFrame) -> pd.DataFrame:
    """Correlación anomalía de demanda vs anomalía de temperatura, por departamento."""
    filas = []
    for dep in config.DEPARTAMENTOS:
        s = dem.loc[dem["unique_id"] == dep.codigo].set_index("ds")["y"]
        t = clima.loc[clima["unique_id"] == dep.codigo].set_index("ds")["temperature_2m"]
        sens = sensibilidad_temperatura(s, t)
        pendiente = sens["demanda_anom_pct"].cov(sens["temp_anom"]) / sens["temp_anom"].var()
        filas.append(
            {
                "departamento": dep.codigo,
                "temp_media": t.mean(),
                "corr": sens["demanda_anom_pct"].corr(sens["temp_anom"]),
                "pct_por_grado": pendiente,
            }
        )
    return pd.DataFrame(filas).set_index("departamento")


def mix_mensual(gen: pd.DataFrame) -> pd.DataFrame:
    """Energía mensual (GWh) por tecnología; BAGAZO y otras renovables menores van a 'Otras'."""
    ancho = gen.pivot(index="ds", columns="unique_id", values="y")
    tec = ancho[list(TECNOLOGIAS)].rename(columns=TECNOLOGIAS)
    tec["Otras"] = ancho.reindex(columns=OTRAS).sum(axis=1, min_count=1)
    # MW medios por hora × 1 h = MWh; / 1000 = GWh
    return tec.resample("MS").sum(min_count=600) / 1000
