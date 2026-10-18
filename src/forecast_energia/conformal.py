"""Intervalos de predicción con garantía de cobertura: split conformal, CQR y ACI.

Todos los métodos operan sobre un backtest en formato Nixtla (`ds`, `y`, columnas de
pronóstico) y agregan columnas `<nombre>`, `<nombre>-lo-<nivel>` y `<nombre>-hi-<nivel>`,
compatibles con `evaluation.evaluar`.

**Conjunto de calibración**, siempre con datos conocidos en el corte del día D (D−2 23:00):
- *estático*: el periodo fijo de calibración (`periodo == "calibracion"`).
- *rodante*: los últimos `ventana_dias` días hasta D−2 inclusive. Se adapta a los cambios de
  régimen y sigue sin mirar el futuro.

**Mondrian por hora:** los scores se calibran por separado para cada hora del día. Así la
cobertura vale hora a hora (condicional) y no solo en promedio: el error de las 18 h es
casi 3 veces el de las 4 h.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RETRASO_DIAS = 2  # con el corte en D−2 23:00, el último día observado es D−2


def cuantil_conformal(scores: np.ndarray, cobertura: float) -> float:
    """Cuantil con corrección de muestra finita: el ⌈(n+1)·cobertura⌉-ésimo score ordenado.

    Garantiza P(score_nuevo ≤ q) ≥ cobertura si los scores son intercambiables. Devuelve +inf
    cuando n es demasiado chico para la cobertura pedida.
    """
    s = np.sort(scores[~np.isnan(scores)])
    n = len(s)
    if n == 0:
        return np.nan
    k = int(np.ceil((n + 1) * cobertura))
    return np.inf if k > n else float(s[k - 1])


class _Calibrador:
    """Selecciona los scores de calibración válidos para cada día objetivo y hora."""

    def __init__(self, cv: pd.DataFrame, ventana_dias: int | None, por_hora: bool):
        self.dia = cv["ds"].dt.normalize().to_numpy()
        self.hora = cv["ds"].dt.hour.to_numpy() if por_hora else np.zeros(len(cv), int)
        self.ventana = ventana_dias
        self.fijo = (cv["periodo"] == "calibracion").to_numpy()
        self.observado = cv["y"].notna().to_numpy()

    def mascara(self, dia_objetivo: np.datetime64) -> np.ndarray | None:
        """Filas usables para calibrar el día objetivo (None si no hay suficientes)."""
        if self.ventana is None:
            m = self.fijo & (self.dia < dia_objetivo)  # el día objetivo nunca se calibra a sí mismo
        else:
            fin = dia_objetivo - np.timedelta64(RETRASO_DIAS, "D")
            inicio = fin - np.timedelta64(self.ventana - 1, "D")
            m = (self.dia >= inicio) & (self.dia <= fin)
            if (self.dia[m & self.observado]).size < 0.8 * self.ventana * 24:
                return None
        return m & self.observado


def _nombres(nombre: str, niveles) -> list[str]:
    return [nombre] + [f"{nombre}-{lado}-{n}" for n in niveles for lado in ("lo", "hi")]


def _destino(cv: pd.DataFrame, solo_test: bool) -> np.ndarray:
    if not solo_test:
        return np.ones(len(cv), bool)
    return cv["periodo"].isin(["test", "prospectivo"]).to_numpy()


def split_conformal(
    cv: pd.DataFrame,
    modelo: str,
    nombre: str,
    *,
    niveles=(80, 90, 95),
    ventana_dias: int | None = None,
    por_hora: bool = True,
    solo_test: bool = True,
) -> pd.DataFrame:
    """Split conformal **asimétrico** sobre los residuos e = y − ŷ.

    Cada cola usa α/2 con la corrección de muestra finita: [ŷ + q_lo, ŷ + q_hi]. Al ser
    asimétrico, absorbe el sesgo del modelo (p.ej. si subestima la rampa de las 18 h).
    """
    cal = _Calibrador(cv, ventana_dias, por_hora)
    yhat = cv[modelo].to_numpy()
    e = cv["y"].to_numpy() - yhat
    out = {c: np.full(len(cv), np.nan) for c in _nombres(nombre, niveles)}
    out[nombre] = yhat.copy()
    destino = _destino(cv, solo_test)
    for dia in np.unique(cal.dia[destino]):
        m = cal.mascara(dia)
        if m is None:
            continue
        filas = destino & (cal.dia == dia)
        for h in np.unique(cal.hora[filas]):
            fh = filas & (cal.hora == h)
            scores = e[m & (cal.hora == h)]
            for n in niveles:
                a = 1 - n / 100
                lo = -cuantil_conformal(-scores, 1 - a / 2)
                hi = cuantil_conformal(scores, 1 - a / 2)
                out[f"{nombre}-lo-{n}"][fh] = yhat[fh] + lo
                out[f"{nombre}-hi-{n}"][fh] = yhat[fh] + hi
    return cv.assign(**out)


def cqr(
    cv: pd.DataFrame,
    modelo: str,
    nombre: str,
    *,
    niveles=(80, 90, 95),
    ventana_dias: int | None = None,
    por_hora: bool = True,
    solo_test: bool = True,
) -> pd.DataFrame:
    """Conformalized Quantile Regression (Romano et al., 2019) sobre los cuantiles del modelo.

    Score s = max(q_lo − y, y − q_hi). El intervalo [q_lo − Q, q_hi + Q], con Q el cuantil
    conformal de s, conserva la forma heterocedástica de los cuantiles del modelo y corrige
    su cobertura (angostos → Q > 0; anchos → Q < 0).
    """
    cal = _Calibrador(cv, ventana_dias, por_hora)
    y = cv["y"].to_numpy()
    out = {c: np.full(len(cv), np.nan) for c in _nombres(nombre, niveles)}
    out[nombre] = cv[modelo].to_numpy().copy()
    destino = _destino(cv, solo_test)
    for n in niveles:
        q_lo, q_hi = cv[f"{modelo}-lo-{n}"].to_numpy(), cv[f"{modelo}-hi-{n}"].to_numpy()
        s = np.maximum(q_lo - y, y - q_hi)
        for dia in np.unique(cal.dia[destino]):
            m = cal.mascara(dia)
            if m is None:
                continue
            filas = destino & (cal.dia == dia)
            for h in np.unique(cal.hora[filas]):
                fh = filas & (cal.hora == h)
                q = cuantil_conformal(s[m & (cal.hora == h)], n / 100)
                out[f"{nombre}-lo-{n}"][fh] = q_lo[fh] - q
                out[f"{nombre}-hi-{n}"][fh] = q_hi[fh] + q
    return cv.assign(**out)


def aci(
    cv: pd.DataFrame,
    modelo: str,
    nombre: str,
    *,
    niveles=(80, 90, 95),
    ventana_dias: int = 91,
    gamma: float = 0.05,
    por_hora: bool = True,
    solo_test: bool = True,
) -> pd.DataFrame:
    """Adaptive Conformal Inference (Gibbs & Candès, 2021) sobre el split conformal rodante.

    Para cada nivel, α_D se ajusta día a día: α_{D} = α + γ · Σ_{d ≤ D−2} (α − err_d), donde
    err_d es la fracción de horas del día d fuera del intervalo. El error de un día recién
    se conoce 2 días después, por el corte. Si el método venía subcubriendo, α baja y los
    intervalos se ensanchan, y viceversa. α_D se acota a [2/(n+1), 0.5], donde n es la cantidad
    de scores por hora: un α menor exigiría un cuantil más allá del máximo observado y daría un
    intervalo infinito.
    """
    cal = _Calibrador(cv, ventana_dias, por_hora)
    y = cv["y"].to_numpy()
    yhat = cv[modelo].to_numpy()
    e = y - yhat
    out = {c: np.full(len(cv), np.nan) for c in _nombres(nombre, niveles)}
    out[nombre] = yhat.copy()
    destino = _destino(cv, solo_test)
    dias = np.unique(cal.dia[destino])
    for n in niveles:
        alpha = 1 - n / 100
        errores: dict[np.datetime64, float] = {}
        for dia in dias:
            limite = dia - np.timedelta64(RETRASO_DIAS, "D")
            ajuste = sum(alpha - err for d, err in errores.items() if d <= limite)
            a_t = float(np.clip(alpha + gamma * ajuste, 1e-3, 0.5))
            m = cal.mascara(dia)
            if m is None:
                continue
            filas = destino & (cal.dia == dia)
            for h in np.unique(cal.hora[filas]):
                fh = filas & (cal.hora == h)
                scores = e[m & (cal.hora == h)]
                # Con n scores, el α más chico con intervalo finito es 2/(n+1) (α/2 por cola).
                a_h = max(a_t, 2 / (np.count_nonzero(~np.isnan(scores)) + 1))
                out[f"{nombre}-lo-{n}"][fh] = yhat[fh] - cuantil_conformal(-scores, 1 - a_h / 2)
                out[f"{nombre}-hi-{n}"][fh] = yhat[fh] + cuantil_conformal(scores, 1 - a_h / 2)
            lo, hi = out[f"{nombre}-lo-{n}"][filas], out[f"{nombre}-hi-{n}"][filas]
            obs = ~np.isnan(y[filas])
            if obs.any():
                fuera = (y[filas] < lo) | (y[filas] > hi)
                errores[dia] = float(fuera[obs].mean())
    return cv.assign(**out)
