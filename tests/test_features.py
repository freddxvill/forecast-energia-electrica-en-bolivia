import numpy as np
import pandas as pd
import pytest

from forecast_energia import config, features


def _clima(ds, temps: dict[str, float]):
    filas = []
    for uid, t in temps.items():
        d = pd.DataFrame({"unique_id": uid, "ds": ds})
        for v in config.CLIMA_VARIABLES:
            d[v] = 0.0  # valor "observado": no debe usarse como exógena
            d[f"{v}{config.CLIMA_SUFIJO_PRONOSTICO}"] = 1.0
        d[f"temperature_2m{config.CLIMA_SUFIJO_PRONOSTICO}"] = t
        filas.append(d)
    return pd.concat(filas, ignore_index=True)


def test_clima_usa_solo_el_pronostico_sin_leakage():
    ds = pd.date_range("2026-01-01", periods=3, freq="h")
    clima = _clima(ds, {"SANTA CRUZ": 30.0, "LA PAZ": 10.0})
    pesos = pd.Series({"SANTA CRUZ": 0.75, "LA PAZ": 0.25})
    out = features.clima_pronosticado(clima, pesos).set_index("unique_id")
    assert (out.loc["SANTA CRUZ", "temp"] == 30).all()
    assert (out.loc["SANTA CRUZ", "humedad"] == 1).all()  # `_d2`, no el observado (0)
    assert (out.loc["SANTA CRUZ", "cdd"] == 30 - features.TEMP_CONFORT).all()
    assert (out.loc["LA PAZ", "cdd"] == 0).all()
    # SIN: promedio ponderado por participación
    assert out.loc[config.SIN_ID, "temp"].iloc[0] == pytest.approx(0.75 * 30 + 0.25 * 10)
    assert out.loc[config.SIN_ID, "cdd"].iloc[0] == pytest.approx(0.75 * 6)


def test_agregar_exogenas_cubre_el_futuro():
    ds = pd.date_range("2026-01-01", periods=48, freq="h")
    panel = pd.DataFrame(
        {"unique_id": config.SIN_ID, "ds": ds, "y": np.r_[np.ones(24), [np.nan] * 24]}
    )
    clima = _clima(ds, {"SANTA CRUZ": 25.0})
    out = features.agregar_exogenas(panel, clima, pd.Series({"SANTA CRUZ": 1.0}))
    exog = features.CALENDARIO + features.CICLICAS + features.CLIMA
    assert out[exog].notna().all().all()
    assert out.loc[0, "es_feriado"] == 1  # 1 de enero
    assert len(out) == 48


def test_pesos_departamentales_suman_uno():
    dem = pd.DataFrame(
        {
            "unique_id": ["SANTA CRUZ", "LA PAZ", "SIN"],
            "ds": pd.Timestamp("2026-01-01"),
            "y": [3.0, 1.0, 4.0],
        }
    )
    pesos = features.pesos_departamentales(dem)
    assert pesos.sum() == pytest.approx(1)
    assert pesos["SANTA CRUZ"] == pytest.approx(0.75)
    assert "SIN" not in pesos


def test_futuro():
    f = features.futuro(pd.Timestamp("2026-01-01 23:00"), ["A", "B"], 48)
    assert len(f) == 96
    assert f["ds"].min() == pd.Timestamp("2026-01-02 00:00")
