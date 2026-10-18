import pandas as pd

from forecast_energia.data.calendar import features_calendario


def test_carnaval_es_feriado_nacional():
    ds = pd.Series(pd.to_datetime(["2025-03-03 10:00", "2025-03-05 10:00"]))
    cal = features_calendario(ds)
    assert cal["es_feriado"].tolist() == [1, 0]
    assert cal["post_feriado"].tolist() == [0, 1]


def test_feriado_departamental_solo_en_su_depto():
    ds = pd.Series(pd.to_datetime(["2025-09-24 12:00"] * 3))
    uid = pd.Series(["SANTA CRUZ", "LA PAZ", "SIN"])
    cal = features_calendario(ds, uid)
    assert cal["es_feriado"].tolist() == [1, 0, 0]


def test_variables_basicas():
    ds = pd.Series(pd.to_datetime(["2025-10-18 23:00"]))  # sábado
    cal = features_calendario(ds).iloc[0]
    assert (cal["hora"], cal["dia_semana"], cal["es_fin_de_semana"]) == (23, 5, 1)


def test_dia_de_la_dignidad_no_es_feriado():
    # Día cívico sin suspensión de actividades: la demanda no cae (ver calendar.DIAS_CIVICOS).
    ds = pd.Series(pd.to_datetime(["2025-10-17 12:00", "2026-10-17 12:00", "2026-10-18 12:00"]))
    cal = features_calendario(ds, pd.Series(["SIN", "BENI", "SIN"]))
    assert cal["es_feriado"].tolist() == [0, 0, 0]
    assert cal["post_feriado"].tolist() == [0, 0, 0]
