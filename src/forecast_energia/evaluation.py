"""Protocolo de evaluación day-ahead: backtest rolling-origin, métricas y significancia.

Convenciones (las de Nixtla):
- Los pronósticos de un backtest vienen en formato largo, con columnas `unique_id, ds, cutoff, y`
  y una columna por modelo. Los intervalos se nombran `<modelo>-lo-<nivel>` / `<modelo>-hi-<nivel>`.
- `cutoff` es el último timestamp observado. Para el día D, `cutoff` = D−2 23:00 y se
  evalúan los horizontes 25–48 (ver `config.HORAS_DIA_OBJETIVO`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from forecast_energia import config

# --- Protocolo -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Particion:
    """Periodos del backtest: calibración (para conformal) y test (para reportar)."""

    calib_inicio: pd.Timestamp
    test_inicio: pd.Timestamp
    fin: pd.Timestamp  # último día evaluado (inclusive)

    @property
    def dias_calibracion(self) -> int:
        return (self.test_inicio - self.calib_inicio).days

    @property
    def dias_test(self) -> int:
        return (self.fin - self.test_inicio).days + 1

    def n_ventanas(self, desde: pd.Timestamp | None = None) -> int:
        """Ventanas diarias de backtest desde `desde` (por defecto, desde la calibración)."""
        desde = self.calib_inicio if desde is None else desde
        return (self.fin - desde).days + 1

    def etiqueta(self, ds: pd.Series) -> pd.Series:
        """'calibracion' | 'test' | 'train' según la fecha de cada pronóstico."""
        dia = pd.to_datetime(ds).dt.normalize()
        out = pd.Series("train", index=ds.index)
        out[dia >= self.calib_inicio] = "calibracion"
        out[dia >= self.test_inicio] = "test"
        return out


def particion_por_defecto(ultimo_ds: pd.Timestamp) -> Particion:
    """Partición con las fechas de `config`. El test llega hasta el último día completo."""
    ultimo = pd.Timestamp(ultimo_ds)
    fin = ultimo.normalize() if ultimo.hour == 23 else ultimo.normalize() - pd.Timedelta(days=1)
    return Particion(pd.Timestamp(config.CALIB_INICIO), pd.Timestamp(config.TEST_INICIO), fin)


def recortar_para_backtest(df: pd.DataFrame, particion: Particion) -> pd.DataFrame:
    """Recorta el panel al último día evaluado, para que las ventanas de Nixtla terminen ahí."""
    return df[df["ds"] < particion.fin + pd.Timedelta(days=1)].reset_index(drop=True)


def horizonte(cv: pd.DataFrame) -> pd.Series:
    """Horizonte en horas de cada pronóstico (1 = la hora siguiente al corte)."""
    return ((cv["ds"] - cv["cutoff"]) / pd.Timedelta(hours=1)).astype(int)


def filtrar_dia_objetivo(cv: pd.DataFrame) -> pd.DataFrame:
    """Conserva solo los horizontes 25–48 (el día D) de un backtest de Nixtla."""
    h = horizonte(cv)
    lo, hi = config.HORAS_DIA_OBJETIVO
    out = cv[(h >= lo) & (h <= hi)].copy()
    out["horizonte"] = h[out.index]
    return out.reset_index(drop=True)


def reemplazar_y_observado(cv: pd.DataFrame, observado: pd.DataFrame) -> pd.DataFrame:
    """Sustituye `y` del backtest por el valor observado original (NaN donde no hubo dato)."""
    obs = observado[["unique_id", "ds", "y"]].rename(columns={"y": "_y_obs"})
    out = cv.merge(obs, on=["unique_id", "ds"], how="left")
    out["y"] = out.pop("_y_obs")
    return out


# --- Métricas puntuales ----------------------------------------------------------------------


def _alinear(y: pd.Series, yhat: pd.Series) -> tuple[pd.Series, pd.Series]:
    m = y.notna() & yhat.notna()
    return y[m], yhat[m]


def escala_mase(y_train: pd.Series, m: int = config.ESTACIONALIDAD_MASE) -> float:
    """MAE in-sample del naive estacional (lag `m`), denominador de MASE."""
    return float((y_train - y_train.shift(m)).abs().mean())


def metricas_puntuales(
    y: pd.Series, yhat: pd.Series, escala: float | None = None
) -> dict[str, float]:
    """MAE, RMSE, MAPE (%), sesgo medio (yhat − y), MASE (si hay `escala`) y n."""
    y, yhat = _alinear(y, yhat)
    e = yhat - y
    out = {
        "MAE": float(e.abs().mean()),
        "RMSE": float(np.sqrt((e**2).mean())),
        "MAPE": float(100 * (e.abs() / y).mean()),
        "sesgo": float(e.mean()),
    }
    if escala:
        out["MASE"] = out["MAE"] / escala
    out["n"] = int(len(e))
    return out


def tabla_metricas(y: pd.Series, pronosticos: dict[str, pd.Series]) -> pd.DataFrame:
    """Una fila por pronóstico, evaluados sobre las mismas horas (intersección)."""
    comunes = y.notna()
    for p in pronosticos.values():
        comunes &= p.reindex(y.index).notna()
    filas = {
        nombre: metricas_puntuales(y[comunes], p.reindex(y.index)[comunes])
        for nombre, p in pronosticos.items()
    }
    return pd.DataFrame(filas).T


def error_por_grupo(y: pd.Series, yhat: pd.Series, grupo: pd.Series) -> pd.DataFrame:
    """MAE, MAPE y sesgo agregados por un grupo (p.ej. hora del día o mes)."""
    y, yhat = _alinear(y, yhat.reindex(y.index))
    e = yhat - y
    df = pd.DataFrame(
        {"ae": e.abs(), "ape": 100 * e.abs() / y, "e": e, "g": grupo.reindex(y.index)}
    )
    return df.groupby("g").agg(MAE=("ae", "mean"), MAPE=("ape", "mean"), sesgo=("e", "mean"))


# --- Métricas probabilísticas ----------------------------------------------------------------


def cobertura(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fracción de observaciones dentro de [lo, hi]."""
    return float(np.mean((y >= lo) & (y <= hi)))


def ancho_medio(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))


def winkler(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, nivel: float) -> float:
    """Winkler score (interval score) medio: ancho + 2/α × lo que y queda fuera del intervalo."""
    alpha = 1 - nivel / 100
    pen = (2 / alpha) * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
    return float(np.mean(hi - lo + pen))


def pinball(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    """Pinball loss medio del cuantil `q`."""
    d = y - q_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def crps_desde_cuantiles(y: np.ndarray, cuantiles: dict[float, np.ndarray]) -> float:
    """Aproximación del CRPS: 2 × la media de los pinball losses sobre una grilla de cuantiles.

    Es exacta en el límite de una grilla densa y uniforme. Con los cuantiles de los intervalos
    80/90/95 + la mediana es una aproximación gruesa, útil para comparar modelos entre sí.
    """
    return float(2 * np.mean([pinball(y, qp, q) for q, qp in cuantiles.items()]))


def cuantiles_de_intervalos(
    df: pd.DataFrame, modelo: str, niveles=config.NIVELES
) -> dict[float, np.ndarray]:
    """Convierte las columnas Nixtla `modelo-lo/hi-<nivel>` (y la puntual) a cuantiles."""
    q: dict[float, np.ndarray] = {0.5: df[modelo].to_numpy()}
    for n in niveles:
        a = (1 - n / 100) / 2
        q[round(a, 4)] = df[f"{modelo}-lo-{n}"].to_numpy()
        q[round(1 - a, 4)] = df[f"{modelo}-hi-{n}"].to_numpy()
    return dict(sorted(q.items()))


def metricas_probabilisticas(df: pd.DataFrame, modelo: str, niveles=config.NIVELES) -> dict:
    """Cobertura, ancho y Winkler por nivel, más el CRPS aproximado, para un modelo."""
    d = df.dropna(subset=["y", modelo])
    y = d["y"].to_numpy()
    out: dict[str, float] = {}
    for n in niveles:
        lo, hi = d[f"{modelo}-lo-{n}"].to_numpy(), d[f"{modelo}-hi-{n}"].to_numpy()
        out[f"cobertura_{n}"] = 100 * cobertura(y, lo, hi)
        out[f"ancho_{n}"] = ancho_medio(lo, hi)
        out[f"winkler_{n}"] = winkler(y, lo, hi, n)
    out["CRPS"] = crps_desde_cuantiles(y, cuantiles_de_intervalos(d, modelo, niveles))
    return out


def cobertura_por_grupo(df: pd.DataFrame, modelo: str, nivel: int, grupo: pd.Series) -> pd.Series:
    """Cobertura (%) condicional, p.ej. por hora del día: debe ser plana cerca del nominal."""
    dentro = (df["y"] >= df[f"{modelo}-lo-{nivel}"]) & (df["y"] <= df[f"{modelo}-hi-{nivel}"])
    return 100 * dentro.groupby(grupo).mean()


def tiene_intervalos(df: pd.DataFrame, modelo: str, niveles=config.NIVELES) -> bool:
    return all(f"{modelo}-lo-{n}" in df and f"{modelo}-hi-{n}" in df for n in niveles)


def evaluar(
    df: pd.DataFrame,
    modelos: list[str],
    escala: float | None = None,
    niveles=config.NIVELES,
) -> pd.DataFrame:
    """Tabla de métricas (una fila por modelo), sobre las horas en que todos tienen pronóstico."""
    comunes = df["y"].notna()
    for m in modelos:
        comunes &= df[m].notna()
    d = df[comunes]
    filas = {}
    for m in modelos:
        fila = metricas_puntuales(d["y"], d[m], escala)
        if tiene_intervalos(d, m, niveles):
            fila |= metricas_probabilisticas(d, m, niveles)
        filas[m] = fila
    return pd.DataFrame(filas).T


# --- Significancia ---------------------------------------------------------------------------


def diebold_mariano(
    y: pd.Series, yhat_a: pd.Series, yhat_b: pd.Series, potencia: int = 1
) -> dict[str, float]:
    """Test de Diebold-Mariano (con corrección de Harvey-Leybourne-Newbold) sobre pérdidas diarias.

    Compara la pérdida |e|^potencia de A vs B. Las horas de un mismo día están muy
    correlacionadas, así que primero se promedia por día y el test se hace sobre la serie
    diaria de diferencias, con varianza HAC de 1 rezago (los backtests diarios se solapan).
    `estadistico` < 0 indica que A tiene menor error. `p_valor` es bilateral.
    """
    d = pd.DataFrame({"y": y, "a": yhat_a, "b": yhat_b}).dropna()
    perdida = (d[["a", "b"]].sub(d["y"], axis=0)).abs() ** potencia
    dif = (perdida["a"] - perdida["b"]).groupby(d.index.normalize()).mean().to_numpy()
    n, h = len(dif), 2
    media = dif.mean()
    c = dif - media
    gamma = [np.sum(c[k:] * c[: n - k]) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    dm = media / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    estadistico = float(dm * hln)
    p = float(2 * stats.t.sf(abs(estadistico), df=n - 1))
    return {"estadistico": estadistico, "p_valor": p, "dias": n, "dif_media": float(media)}
