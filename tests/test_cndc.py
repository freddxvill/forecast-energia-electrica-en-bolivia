import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast_energia.data import cndc

FIXTURES = Path(__file__).parent / "fixtures"
FECHA = date(2026, 10, 15)


def _cargar(nombre: str):
    return json.loads((FIXTURES / nombre).read_text(encoding="utf-8"))


@pytest.fixture
def demanda_raw():
    return _cargar("cndc_rt_demanda_2026-10-15.json")


@pytest.fixture
def generacion_raw():
    return _cargar("cndc_rt_generacion_2026-10-15.json")


def test_parse_dia_formato_largo(demanda_raw):
    df = cndc.parse_dia(demanda_raw, FECHA)
    assert list(df.columns) == ["unique_id", "ds", "y"]
    assert len(df) == len(demanda_raw) * 96
    assert set(df["unique_id"]) >= {"SANTA CRUZ", "LA PAZ", "BENI"}


def test_parse_dia_timestamps_00_15_a_24_00(demanda_raw):
    df = cndc.parse_dia(demanda_raw, FECHA)
    ds = df.loc[df["unique_id"] == "LA PAZ", "ds"]
    assert ds.min() == pd.Timestamp("2026-10-15 00:15")
    assert ds.max() == pd.Timestamp("2026-10-16 00:00")


def test_parse_dia_centinela_a_nan(demanda_raw):
    df = cndc.parse_dia(demanda_raw, FECHA)
    negativos = sum(v == -1 for s in demanda_raw for v in s["valores"])
    assert negativos > 0
    assert df["y"].isna().sum() == negativos
    assert (df["y"].dropna() >= 0).all()


def test_parse_dia_vacio():
    df = cndc.parse_dia([], FECHA)
    assert df.empty
    assert list(df.columns) == ["unique_id", "ds", "y"]


def test_convencion_horaria_coincide_con_curva_oficial(generacion_raw):
    """El valor a las hh:00 coincide con la hora hh de rt/curva-dia (generación)."""
    curva = _cargar("cndc_rt_curva-dia_2026-10-15.json")
    df = cndc.parse_dia(generacion_raw, FECHA)
    tot = df[df["unique_id"] == "TOT"].set_index("ds")["y"]
    horas = pd.Timestamp(FECHA) + pd.to_timedelta(np.arange(1, 25), unit="h")
    mae = np.nanmean(np.abs(tot.reindex(horas).to_numpy() - np.asarray(curva["generacion"])))
    assert mae < 10, f"MAE {mae:.1f} MW: la convención de timestamps no coincide"


def test_parse_precios_mensuales():
    df = cndc.parse_precios_mensuales(_cargar("cndc_dashboard_precios_2026-09.json"))
    assert len(df) == 12
    assert df["mes"].min() == pd.Timestamp("2025-10-01")
    assert df["mes"].max() == pd.Timestamp("2026-09-01")
    sep = df.set_index("mes").loc["2026-09-01"]
    assert sep["precio_energia"] == pytest.approx(22.7222)
    assert sep["precio_potencia"] == pytest.approx(10.2299)


def test_parse_cm_diario_infiere_el_anio():
    df = cndc.parse_cm_diario(_cargar("cndc_cm-diario_10.json"), date(2026, 10, 17))
    assert df["fecha"].tolist() == list(pd.date_range("2026-10-06", "2026-10-15"))
    assert df["costo_marginal"].iloc[-1] == pytest.approx(15.529)


def test_parse_cm_diario_cruza_el_cambio_de_anio():
    raw = {"fechas": ["30/12", "31/12", "01/01", "02/01"], "valores": [1, 2, 3, 4]}
    df = cndc.parse_cm_diario(raw, date(2027, 1, 3))
    assert df["fecha"].dt.year.tolist() == [2026, 2026, 2027, 2027]
    assert df["costo_marginal"].tolist() == [1, 2, 3, 4]


def test_acumular_conserva_la_historia_y_prioriza_lo_nuevo():
    viejo = pd.DataFrame({"fecha": pd.to_datetime(["2026-01-01", "2026-01-02"]), "v": [1, 2]})
    nuevo = pd.DataFrame({"fecha": pd.to_datetime(["2026-01-02", "2026-01-03"]), "v": [9, 3]})
    out = cndc.acumular(nuevo, viejo, "fecha")
    assert out["v"].tolist() == [1, 9, 3]


def test_parse_precmp_total_mapea_h_menos_1():
    s = cndc.parse_precmp_total(_cargar("cndc_precmp_2026-10-18.json"), date(2026, 10, 18))
    assert len(s) == 23
    assert s.index[0] == pd.Timestamp("2026-10-18 01:00")
    assert s.index[-1] == pd.Timestamp("2026-10-18 23:00")
    assert s.iloc[0] == pytest.approx(1117.3, abs=0.1)


def test_parse_precmp_sin_datos_devuelve_vacio():
    assert cndc.parse_precmp_total({"sin_datos": True}, date(2026, 10, 25)).empty


def test_prev_predespacho_imputa_la_medianoche(monkeypatch):
    def falso(_session, _endpoint, fecha):
        base = 1000.0 if fecha == "2026-10-17" else 2000.0
        horas = [base + h for h in range(1, 24)] + [None]
        return {
            "grupos": [
                {
                    "tecnologia": "SISTEMA",
                    "filas": [{"unidad": "TOTAL", "es_subtotal": 1, "horas": horas}],
                }
            ]
        }

    monkeypatch.setattr(cndc, "_get_json", falso)
    s = cndc.prev_predespacho([date(2026, 10, 18)], session=object())
    assert s.index.min() == pd.Timestamp("2026-10-18 00:00")
    assert len(s) == 24
    # media de las 23:00 del 17 (1023) y la 01:00 del 18 (2001)
    assert s.iloc[0] == pytest.approx((1023 + 2001) / 2)
    assert s.loc["2026-10-18 05:00"] == pytest.approx(2005)
