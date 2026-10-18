import numpy as np
import pandas as pd
import pytest

from forecast_energia import hierarchy as H

DEPTOS = ["A", "B", "C"]


def test_matriz_s():
    s = H.matriz_s("T", DEPTOS)
    assert s.shape == (4, 3)
    assert s.loc["T"].tolist() == [1, 1, 1]
    np.testing.assert_array_equal(s.loc[DEPTOS].to_numpy(), np.eye(3))


def test_mint_es_insesgado_y_respeta_lo_coherente():
    rng = np.random.default_rng(0)
    s = H.matriz_s("T", DEPTOS).to_numpy()
    a = rng.normal(size=(4, 4))
    w = a @ a.T + 4 * np.eye(4)
    p = H.matriz_p_mint(s, w)
    np.testing.assert_allclose(p @ s, np.eye(3), atol=1e-10)
    coherente = s @ np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(s @ p @ coherente, coherente, atol=1e-10)


def test_covarianza_shrink_conserva_varianzas_y_es_simetrica():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 4)) @ rng.normal(size=(4, 4))
    w = H.covarianza_shrink(x)
    cov = x.T @ x / len(x)
    np.testing.assert_allclose(np.diag(w), np.diag(cov))
    np.testing.assert_allclose(w, w.T)
    assert np.all(np.linalg.eigvalsh(w) > 0)


def _cv_jerarquico(dias=40, seed=0):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2026-01-01", periods=24 * dias, freq="h")
    y = {d: 100 * (i + 1) + rng.normal(0, 5, len(ds)) for i, d in enumerate(DEPTOS)}
    y["T"] = sum(y[d] for d in DEPTOS)
    filas = []
    for u, v in y.items():
        sesgo = 30.0 if u == "T" else 0.0  # el total base está sesgado: incoherente
        filas.append(
            pd.DataFrame(
                {"unique_id": u, "ds": ds, "y": v, "M": v + rng.normal(0, 8, len(ds)) + sesgo}
            )
        )
    return pd.concat(filas, ignore_index=True)


def test_reconciliar_backtest_coherente_y_sin_leakage():
    cv = _cv_jerarquico()
    base = {u: "M" for u in ["T", *DEPTOS]}
    out = H.reconciliar_backtest(cv, base, "T", DEPTOS, min_dias=7)
    for m in ("BottomUp", "TopDown", "MinT-WLS", "MinT-shrink"):
        ancho = out.pivot(index="ds", columns="unique_id", values=m).dropna()
        assert len(ancho) > 0
        np.testing.assert_allclose(ancho["T"], ancho[DEPTOS].sum(axis=1), rtol=1e-9)
    # Alterar D−1 y D no cambia la reconciliación del día D
    dia_d = cv["ds"].max().normalize()
    alterado = cv.copy()
    alterado.loc[alterado["ds"] >= dia_d - pd.Timedelta(days=1), "y"] += 1e4
    out2 = H.reconciliar_backtest(alterado, base, "T", DEPTOS, min_dias=7)
    fila = out["ds"] >= dia_d
    pd.testing.assert_frame_equal(
        out.loc[fila, ["MinT-shrink", "TopDown"]], out2.loc[fila, ["MinT-shrink", "TopDown"]]
    )


def test_mint_corrige_el_total_sesgado():
    cv = _cv_jerarquico(dias=60)
    out = H.reconciliar_backtest(cv, {u: "M" for u in ["T", *DEPTOS]}, "T", DEPTOS, min_dias=7)
    t = out[(out["unique_id"] == "T")].dropna()
    mae = {m: (t[m] - t["y"]).abs().mean() for m in ("Base", "MinT-WLS")}
    assert mae["MinT-WLS"] < mae["Base"]
    assert t["Base"].sub(t["y"]).mean() == pytest.approx(30, abs=3)
