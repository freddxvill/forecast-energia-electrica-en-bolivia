"""Reconciliación jerárquica SIN = Σ departamentos: BottomUp, TopDown y MinTrace (WLS / shrink).

Los pronósticos base de cada serie se hacen por separado y no suman exactamente (son
*incoherentes*). Reconciliar los proyecta a un conjunto coherente: ỹ = S·P·ŷ, con S la matriz
de agregación (9 × 8) y P la que define cada método.

MinTrace (Wickramasuriya, Athanasopoulos y Hyndman, 2019) elige P = (S′W⁻¹S)⁻¹S′W⁻¹, que
minimiza la varianza total del error reconciliado. W es la covarianza de los errores base y se
estima, para cada día D, con los errores de los días previos hasta D−2 (conocidos en el corte).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_energia.conformal import RETRASO_DIAS


def matriz_s(total: str, inferiores: list[str]) -> pd.DataFrame:
    """Matriz de agregación: fila del total = 1 en todas las columnas, identidad abajo."""
    filas = [total, *inferiores]
    s = np.vstack([np.ones(len(inferiores)), np.eye(len(inferiores))])
    return pd.DataFrame(s, index=filas, columns=inferiores)


def matriz_p_mint(s: np.ndarray, w: np.ndarray) -> np.ndarray:
    """P = (S′W⁻¹S)⁻¹S′W⁻¹ (m × n). Cumple P·S = I: no altera pronósticos ya coherentes."""
    w_inv = np.linalg.pinv(w)
    return np.linalg.solve(s.T @ w_inv @ s, s.T @ w_inv)


def matriz_p_bottom_up(n_inferiores: int) -> np.ndarray:
    """P = [0 | I]: toma los departamentos tal cual y el total es su suma."""
    return np.hstack([np.zeros((n_inferiores, 1)), np.eye(n_inferiores)])


def covarianza_shrink(errores: np.ndarray) -> np.ndarray:
    """Covarianza encogida hacia su diagonal (Schäfer y Strimmer, 2005), como en MinT-shrink.

    `errores`: n_obs × n_series. λ se estima de los datos: con pocas observaciones o
    correlaciones ruidosas, W se acerca a la diagonal (WLS).
    """
    x = errores[~np.isnan(errores).any(axis=1)]
    n = x.shape[0]
    cov = x.T @ x / n
    xs = x / np.sqrt(np.diag(cov))
    cor = xs.T @ xs / n
    v = (1 / (n * (n - 1))) * ((xs**2).T @ (xs**2) - (xs.T @ xs) ** 2 / n)
    np.fill_diagonal(v, 0)
    fuera = cor.copy()
    np.fill_diagonal(fuera, 0)
    lam = float(np.clip(v.sum() / (fuera**2).sum(), 0, 1))
    return lam * np.diag(np.diag(cov)) + (1 - lam) * cov


def _ancho(cv: pd.DataFrame, columna: str) -> pd.DataFrame:
    return cv.pivot(index="ds", columns="unique_id", values=columna).sort_index()


def reconciliar_backtest(
    cv: pd.DataFrame,
    base: dict[str, str],
    total: str,
    inferiores: list[str],
    ventana_dias: int = 91,
    min_dias: int = 14,
) -> pd.DataFrame:
    """Reconcilia día a día un backtest jerárquico y devuelve formato largo.

    `base` indica qué columna usar como pronóstico base de cada serie (p.ej. el total puede
    usar el pronóstico combinado con PREV). Las salidas son las columnas `Base`, `BottomUp`,
    `TopDown`, `MinT-WLS` y `MinT-shrink`. Para cada día D, W y las proporciones de TopDown
    se estiman con los días [D−2−ventana+1, D−2], con al menos `min_dias` días.
    """
    filas = [total, *inferiores]
    s = matriz_s(total, inferiores).to_numpy()
    yhat = pd.concat(
        {u: cv.loc[cv["unique_id"] == u].set_index("ds")[col] for u, col in base.items()}, axis=1
    ).sort_index()[filas]
    y = _ancho(cv, "y").reindex(index=yhat.index, columns=filas)
    errores = (y - yhat).to_numpy()
    participacion = y[inferiores].div(y[total], axis=0)
    dia = yhat.index.normalize()
    hora = yhat.index.hour.to_numpy()

    salidas = {
        m: np.full(yhat.shape, np.nan) for m in ("BottomUp", "TopDown", "MinT-WLS", "MinT-shrink")
    }
    p_bu = matriz_p_bottom_up(len(inferiores))
    for d in dia.unique():
        fin = d - pd.Timedelta(days=RETRASO_DIAS)
        hist = (dia <= fin) & (dia > fin - pd.Timedelta(days=ventana_dias))
        if dia[hist].nunique() < min_dias:
            continue
        hoy = np.asarray(dia == d)
        e_hist = errores[hist]
        e_hist = e_hist[~np.isnan(e_hist).any(axis=1)]
        if len(e_hist) < min_dias * 24:  # p.ej. si algún pronóstico base aún no existía
            continue
        w_wls = np.diag(np.nanvar(e_hist, axis=0))
        p = {
            "BottomUp": p_bu,
            "MinT-WLS": matriz_p_mint(s, w_wls),
            "MinT-shrink": matriz_p_mint(s, covarianza_shrink(e_hist)),
        }
        base_hoy = yhat.to_numpy()[hoy]
        for m, pm in p.items():
            salidas[m][hoy] = base_hoy @ (s @ pm).T
        # TopDown: total base × participación media de cada depto a esa hora (historia reciente)
        prop = participacion[hist].groupby(hora[hist]).mean()
        prop_hoy = prop.reindex(hora[hoy]).to_numpy()
        td = np.column_stack([base_hoy[:, 0], base_hoy[:, :1] * prop_hoy])
        salidas["TopDown"][hoy] = td

    partes = []
    for i, u in enumerate(filas):
        df = pd.DataFrame(
            {"unique_id": u, "ds": yhat.index, "y": y[u].to_numpy(), "Base": yhat[u].to_numpy()}
        )
        for m, arr in salidas.items():
            df[m] = arr[:, i]
        partes.append(df)
    return pd.concat(partes, ignore_index=True)
