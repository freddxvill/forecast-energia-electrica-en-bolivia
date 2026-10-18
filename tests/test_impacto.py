import numpy as np
import pandas as pd
import pytest

from forecast_energia import impacto as I


def test_desvio_evitado_por_hora():
    y = pd.Series([100.0, 100.0, 100.0])
    base = pd.Series([110.0, 90.0, 100.0])
    modelo = pd.Series([105.0, 100.0, 103.0])
    # |e_base| − |e_modelo|: 10 − 5, 10 − 0, 0 − 3 (el modelo empeora en la última hora)
    assert I.desvio_evitado(y, base, modelo).tolist() == [5.0, 10.0, -3.0]


def test_anualizar():
    assert I.anualizar(183.0, 183) == pytest.approx(365.0)


def test_precio_por_hora_usa_el_mes_y_rellena_los_no_publicados():
    mensuales = pd.DataFrame(
        {"mes": pd.to_datetime(["2026-08-01", "2026-09-01"]), "precio_energia": [20.0, 25.0]}
    )
    ds = pd.Series(pd.to_datetime(["2026-08-31 23:00", "2026-09-01 00:00", "2026-10-05 10:00"]))
    assert I.precio_por_hora(ds, mensuales, "precio_energia").tolist() == [20.0, 25.0, 25.0]


def test_valor_desvios():
    mwh = pd.Series([10.0, -2.0])
    assert I.valor_desvios(mwh, pd.Series([20.0, 30.0])) == pytest.approx(140.0)
    assert I.valor_desvios(mwh, 50.0) == pytest.approx(400.0)


def test_costo_combustible_y_heat_rate_implicito():
    assert I.costo_combustible_usd_mwh(10.0, 6.5) == pytest.approx(65.0)
    assert I.heat_rate_implicito(13.0, 1.3) == pytest.approx(10.0)


def test_reserva_fija_cubre_el_nivel_en_calibracion():
    rng = np.random.default_rng(0)
    e = rng.normal(0, 50, 20_000)
    y, yhat = pd.Series(1000 + e), pd.Series(np.full(e.size, 1000.0))
    r = I.reserva_fija(y, yhat, 95)
    assert r == pytest.approx(1.645 * 50, rel=0.03)
    assert I.resumen_reserva(y, yhat, yhat + r)["cobertura_pct"] == pytest.approx(95, abs=0.1)


def test_reserva_fija_con_cobertura_iguala_la_cobertura_pedida():
    rng = np.random.default_rng(1)
    y = pd.Series(rng.normal(0, 10, 10_000))
    yhat = pd.Series(np.zeros(10_000))
    r = I.reserva_fija_con_cobertura(y, yhat, 94.3)
    assert I.resumen_reserva(y, yhat, yhat + r)["cobertura_pct"] == pytest.approx(94.3, abs=0.05)


def test_reserva_variable_por_hora_requiere_menos_mw_que_la_fija():
    # La mitad de las horas son tranquilas (σ = 10) y la otra mitad volátiles (σ = 50): una
    # reserva por hora cubre lo mismo con menos MW de media que una constante.
    rng = np.random.default_rng(2)
    n = 40_000
    sigma = np.where(np.arange(n) % 2 == 0, 10.0, 50.0)
    y, yhat = pd.Series(rng.normal(0, sigma)), pd.Series(np.zeros(n))
    fija = I.resumen_reserva(y, yhat, yhat + I.reserva_fija(y, yhat, 95))
    variable = I.resumen_reserva(y, yhat, yhat + pd.Series(1.645 * sigma))
    assert variable["cobertura_pct"] == pytest.approx(fija["cobertura_pct"], abs=0.5)
    # Fija: el cuantil 90 de N(0, 50) ≈ 64 MW. Variable: 1.645 × media(σ) ≈ 49 MW.
    assert variable["reserva_media_MW"] < 0.8 * fija["reserva_media_MW"]


def test_resumen_reserva_ignora_horas_sin_dato():
    y = pd.Series([10.0, np.nan, 30.0])
    yhat = pd.Series([0.0, 0.0, 0.0])
    out = I.resumen_reserva(y, yhat, pd.Series([20.0, 20.0, 20.0]))
    assert out["n"] == 2
    assert out["cobertura_pct"] == pytest.approx(50.0)
    assert out["reserva_media_MW"] == pytest.approx(20.0)


def test_valor_reserva_usd_anio():
    # 10 MW × 1000 kW/MW × 11 USD/kW-mes × 12 meses
    assert I.valor_reserva_usd_anio(10.0, 11.0) == pytest.approx(1_320_000.0)
