import numpy as np
import pandas as pd
import pytest
from scipy import stats

from forecast_energia import conformal as C
from forecast_energia import evaluation as E


def _cv(dias_calib=91, dias_test=120, sigma_calib=10.0, sigma_test=10.0, seed=0):
    """Backtest sintético: y = 1000 + ruido, pronóstico puntual = 1000 y cuantiles muy angostos."""
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2026-01-01", periods=24 * (dias_calib + dias_test), freq="h")
    n_cal = 24 * dias_calib
    sigma = np.r_[np.full(n_cal, sigma_calib), np.full(len(ds) - n_cal, sigma_test)]
    cv = pd.DataFrame(
        {
            "ds": ds,
            "y": 1000 + rng.normal(0, 1, len(ds)) * sigma,
            "M": 1000.0,
            "periodo": np.where(np.arange(len(ds)) < n_cal, "calibracion", "test"),
        }
    )
    for n in (80, 90, 95):
        z = stats.norm.ppf(0.5 + n / 200) * sigma_calib * 0.3  # 30 % del ancho correcto
        cv[f"M-lo-{n}"], cv[f"M-hi-{n}"] = 1000 - z, 1000 + z
    return cv


def _cobertura(cv, nombre, nivel=90):
    t = cv[(cv["periodo"] == "test") & cv[f"{nombre}-lo-{nivel}"].notna()]
    return 100 * E.cobertura(
        t["y"].to_numpy(), t[f"{nombre}-lo-{nivel}"], t[f"{nombre}-hi-{nivel}"]
    )


def test_cuantil_conformal_correccion_de_muestra_finita():
    s = np.arange(1, 10, dtype=float)  # n = 9
    assert C.cuantil_conformal(s, 0.9) == 9  # ⌈10 · 0.9⌉ = 9
    assert C.cuantil_conformal(s, 0.5) == 5  # ⌈10 · 0.5⌉ = 5
    assert np.isinf(C.cuantil_conformal(s, 0.95))  # ⌈9.5⌉ = 10 > n
    assert np.isnan(C.cuantil_conformal(np.array([]), 0.9))


@pytest.mark.parametrize("ventana", [None, 60])
def test_split_conformal_cubre_el_nivel_nominal(ventana):
    cv = C.split_conformal(_cv(), "M", "CP", ventana_dias=ventana)
    for n in (80, 90, 95):
        assert _cobertura(cv, "CP", n) == pytest.approx(n, abs=2.5)
    assert cv.loc[cv["periodo"] == "calibracion", "CP-lo-90"].isna().all()


def test_ventana_rodante_no_usa_datos_posteriores_a_d_menos_2():
    cv = _cv(dias_calib=91, dias_test=10)
    dia = cv["ds"].dt.normalize().unique()[95]
    base = C.split_conformal(cv, "M", "CP", ventana_dias=30)
    alterado = cv.copy()
    futuro = alterado["ds"].dt.normalize() > dia - pd.Timedelta(days=2)
    alterado.loc[futuro, "y"] += 1e4  # cambiar D−1, D, ... no debe afectar al día D
    out = C.split_conformal(alterado, "M", "CP", ventana_dias=30)
    fila = cv["ds"].dt.normalize() == dia
    np.testing.assert_allclose(out.loc[fila, "CP-hi-90"], base.loc[fila, "CP-hi-90"])


def test_cqr_corrige_cuantiles_demasiado_angostos():
    cv = _cv()
    crudo = _cobertura(cv, "M")
    assert crudo < 50  # los cuantiles del "modelo" subcubren a propósito
    cv = C.cqr(cv, "M", "CQR")
    assert _cobertura(cv, "CQR") == pytest.approx(90, abs=2.5)


def test_aci_se_adapta_a_un_cambio_de_regimen():
    # En test el ruido se duplica: el conformal estático subcubre y ACI corrige.
    cv = _cv(dias_calib=91, dias_test=240, sigma_calib=10, sigma_test=20)
    cv = C.split_conformal(cv, "M", "CP", ventana_dias=None)
    cv = C.aci(cv, "M", "ACI", ventana_dias=91, gamma=0.05)
    estatico, adaptativo = _cobertura(cv, "CP"), _cobertura(cv, "ACI")
    assert estatico < 75
    assert abs(adaptativo - 90) < abs(estatico - 90)
    assert adaptativo > 84
