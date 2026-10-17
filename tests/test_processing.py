import numpy as np
import pandas as pd

from forecast_energia import config
from forecast_energia.data import processing


def _serie(valores, uid="A", inicio="2024-03-01 00:15", freq="15min"):
    ds = pd.date_range(inicio, periods=len(valores), freq=freq)
    return pd.DataFrame({"unique_id": uid, "ds": ds, "y": np.asarray(valores, dtype=float)})


def test_completar_grilla_expone_huecos():
    df = _serie([1, 2, 3, 4]).drop(index=[1, 2])
    out = processing.completar_grilla(df)
    assert len(out) == 4
    assert out["y"].isna().sum() == 2


def test_interpolar_solo_huecos_cortos():
    nan = np.nan
    valores = [0, nan, nan, 3] + [4] + [nan] * 5 + [10]
    out = processing.interpolar_huecos_cortos(_serie(valores), max_pasos=4)
    np.testing.assert_allclose(out["y"].iloc[:4], [0, 1, 2, 3])
    assert out["y"].iloc[5:10].isna().all()  # hueco de 5 > 4: no se inventa
    assert out["imputado"].sum() == 2


def test_interpolar_no_extrapola_bordes():
    out = processing.interpolar_huecos_cortos(_serie([np.nan, 1, 2, np.nan]))
    assert out["y"].isna().sum() == 2
    assert not out["imputado"].any()


def test_a_horario_hora_final_cerrada_a_derecha():
    # 00:15, 00:30, 00:45, 01:00 → hora 00:00; 01:15 … 02:00 → hora 01:00
    out = processing.a_horario(_serie([1, 2, 3, 4, 10, 10, 10, 10]))
    assert list(out["ds"]) == [pd.Timestamp("2024-03-01 00:00"), pd.Timestamp("2024-03-01 01:00")]
    np.testing.assert_allclose(out["y"], [2.5, 10])


def test_a_horario_exige_min_validos():
    out = processing.a_horario(_serie([1, np.nan, np.nan, 4]), min_validos=3)
    assert out["y"].isna().all()


def test_escalon_horario_robusto_a_la_fase():
    # Fase :00 → escalones [00:00, 00:45]; fase :15 → escalones [00:15, 01:00]
    fase_00 = _serie([5, 5, 5, 5, 7, 7, 7, 7], inicio="2024-03-01 00:00")
    fase_15 = _serie([5, 5, 5, 5, 7, 7, 7, 7], inicio="2024-03-01 00:15")
    for df in (fase_00, fase_15):
        out = processing.escalon_horario(df)
        assert list(out["ds"]) == [
            pd.Timestamp("2024-03-01 00:00"),
            pd.Timestamp("2024-03-01 01:00"),
        ]
        np.testing.assert_allclose(out["y"], [5, 7])


def test_construir_sin_suma_y_exige_todos_los_deptos():
    deptos = ["A", "B"]
    df = pd.concat([_serie([1, 2], "A", freq="h"), _serie([10, np.nan], "B", freq="h")])
    sin = processing.construir_sin(df, departamentos=deptos)
    assert (sin["unique_id"] == config.SIN_ID).all()
    assert sin["y"].iloc[0] == 11
    assert np.isnan(sin["y"].iloc[1])
