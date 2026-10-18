import numpy as np
import pandas as pd
import pytest
from scipy import stats

from forecast_energia import evaluation as E
from forecast_energia.data import io

# --- Protocolo -------------------------------------------------------------------------------


def _backtest_sintetico(cortes):
    """Un pronóstico de 48 h por cada corte (como devuelve Nixtla `cross_validation`)."""
    filas = []
    for c in pd.to_datetime(cortes):
        ds = c + pd.to_timedelta(np.arange(1, 49), unit="h")
        filas.append(pd.DataFrame({"unique_id": "SIN", "ds": ds, "cutoff": c, "y": 1.0, "M": 1.0}))
    return pd.concat(filas, ignore_index=True)


def test_filtrar_dia_objetivo_conserva_el_dia_d():
    # Corte = D−2 23:00 → se evalúa el día D completo (00:00 a 23:00)
    cv = _backtest_sintetico(["2026-05-01 23:00"])
    out = E.filtrar_dia_objetivo(cv)
    assert len(out) == 24
    assert out["ds"].min() == pd.Timestamp("2026-05-03 00:00")
    assert out["ds"].max() == pd.Timestamp("2026-05-03 23:00")
    assert out["horizonte"].tolist() == list(range(25, 49))


def test_particion_y_etiquetas():
    p = E.Particion(
        pd.Timestamp("2026-01-16"), pd.Timestamp("2026-04-17"), pd.Timestamp("2026-10-16")
    )
    assert p.dias_test == 183
    assert p.dias_calibracion == 91
    assert p.n_ventanas() == 91 + 183
    ds = pd.Series(pd.to_datetime(["2026-01-15 23:00", "2026-01-16 00:00", "2026-04-17 05:00"]))
    assert p.etiqueta(ds).tolist() == ["train", "calibracion", "test"]


def test_periodo_prospectivo_despues_de_test_fin():
    p = E.Particion(
        pd.Timestamp("2026-01-16"),
        pd.Timestamp("2026-04-17"),
        pd.Timestamp("2026-10-20"),
        pd.Timestamp("2026-10-16"),
    )
    assert p.dias_test == 183
    assert p.dias_prospectivo == 4
    assert p.n_ventanas() == 91 + 183 + 4
    ds = pd.Series(pd.to_datetime(["2026-10-16 23:00", "2026-10-17 00:00"]))
    assert p.etiqueta(ds).tolist() == ["test", "prospectivo"]


def test_particion_por_defecto_fija_el_fin_del_test():
    p = E.particion_por_defecto(pd.Timestamp("2026-11-30 23:00"))
    assert p.fin == pd.Timestamp("2026-11-30")
    assert p.fin_test == pd.Timestamp("2026-10-16")


def test_particion_por_defecto_excluye_dia_incompleto():
    assert E.particion_por_defecto(pd.Timestamp("2026-10-16 23:00")).fin == pd.Timestamp(
        "2026-10-16"
    )
    assert E.particion_por_defecto(pd.Timestamp("2026-10-16 10:00")).fin == pd.Timestamp(
        "2026-10-15"
    )


def test_panel_nixtla_sin_nan_y_reemplazo_de_observados():
    ds = pd.date_range("2026-01-01", periods=5, freq="h")
    df = pd.DataFrame({"unique_id": "A", "ds": ds, "y": [1.0, np.nan, 3.0, 4.0, 5.0]})
    panel = io.panel_nixtla(df)
    assert panel["y"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    cv = panel.assign(cutoff=ds[0], M=panel["y"])
    restaurado = E.reemplazar_y_observado(cv, df)
    assert np.isnan(restaurado.loc[1, "y"])


# --- Métricas --------------------------------------------------------------------------------


def test_metricas_puntuales_y_mase():
    y = pd.Series([100.0, 200.0])
    m = E.metricas_puntuales(y, pd.Series([110.0, 190.0]), escala=20.0)
    assert m["MAE"] == 10
    assert m["MAPE"] == pytest.approx(7.5)
    assert m["sesgo"] == 0
    assert m["MASE"] == pytest.approx(0.5)
    assert E.escala_mase(pd.Series([1.0, 2.0, 4.0, 7.0]), m=1) == pytest.approx(2.0)


def test_cobertura_ancho_y_winkler():
    y = np.array([0.0, 5.0, 12.0])
    lo, hi = np.full(3, 0.0), np.full(3, 10.0)
    assert E.cobertura(y, lo, hi) == pytest.approx(2 / 3)
    assert E.ancho_medio(lo, hi) == 10
    # nivel 90 → α = 0.1 → penalización 2/0.1 = 20 por unidad fuera
    assert E.winkler(y, lo, hi, 90) == pytest.approx(10 + 20 * 2 / 3)


def test_pinball():
    y = np.array([10.0])
    assert E.pinball(y, np.array([8.0]), 0.9) == pytest.approx(0.9 * 2)
    assert E.pinball(y, np.array([12.0]), 0.9) == pytest.approx(0.1 * 2)


def test_crps_aproxima_la_formula_cerrada_gaussiana():
    rng = np.random.default_rng(0)
    y = rng.normal(size=2000)
    qs = np.linspace(0.01, 0.99, 99)
    cuantiles = {q: np.full_like(y, stats.norm.ppf(q)) for q in qs}
    z = y
    exacto = np.mean(z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    assert E.crps_desde_cuantiles(y, cuantiles) == pytest.approx(exacto, rel=0.03)


def test_metricas_probabilisticas_cobertura_nominal():
    """Intervalos gaussianos correctos → cobertura empírica ≈ nominal."""
    rng = np.random.default_rng(1)
    n = 20_000
    df = pd.DataFrame({"y": rng.normal(size=n), "M": 0.0})
    for nivel in (80, 90, 95):
        z = stats.norm.ppf(0.5 + nivel / 200)
        df[f"M-lo-{nivel}"], df[f"M-hi-{nivel}"] = -z, z
    m = E.metricas_probabilisticas(df, "M")
    for nivel in (80, 90, 95):
        assert m[f"cobertura_{nivel}"] == pytest.approx(nivel, abs=1)
    tabla = E.evaluar(df, ["M"])
    assert {"MAE", "cobertura_90", "CRPS"} <= set(tabla.columns)


def test_cobertura_por_grupo():
    df = pd.DataFrame({"y": [1, 5, 1, 5], "M-lo-90": 0, "M-hi-90": 2})
    cob = E.cobertura_por_grupo(df, "M", 90, pd.Series([0, 0, 1, 1]))
    assert cob.tolist() == [50.0, 50.0]


# --- Diebold-Mariano -------------------------------------------------------------------------


def _pronosticos(ruido_a, ruido_b, dias=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=24 * dias, freq="h")
    y = pd.Series(1000 + 100 * np.sin(np.arange(len(idx)) / 24 * 2 * np.pi), index=idx)
    a = y + rng.normal(0, ruido_a, len(idx))
    b = y + rng.normal(0, ruido_b, len(idx))
    return y, a, b


def test_diebold_mariano_detecta_mejor_modelo():
    y, a, b = _pronosticos(ruido_a=20, ruido_b=40)
    r = E.diebold_mariano(y, a, b)
    assert r["estadistico"] < 0
    assert r["p_valor"] < 0.01
    assert r["dias"] == 120


def test_diebold_mariano_no_rechaza_si_son_iguales():
    y, a, b = _pronosticos(ruido_a=30, ruido_b=30, seed=3)
    assert E.diebold_mariano(y, a, b)["p_valor"] > 0.05
