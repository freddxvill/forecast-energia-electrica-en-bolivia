import numpy as np
import pandas as pd

from forecast_energia.models import ensemble, ml


def test_nivel_solo_usa_datos_hasta_t_menos_48():
    ds = pd.date_range("2026-01-01", periods=400, freq="h")
    y = np.arange(400, dtype=float)
    panel = pd.DataFrame({"unique_id": "SIN", "ds": ds, "y": y})
    nivel = ml.nivel(panel)
    t = 300
    # media de y[t-48-167 .. t-48]
    assert nivel.iloc[t] == y[t - 48 - 167 : t - 48 + 1].mean()
    # cambiar el futuro (t-47 en adelante) no altera el nivel en t
    panel.loc[t - 47 :, "y"] = 1e6
    assert ml.nivel(panel).iloc[t] == nivel.iloc[t]


def test_ensamble_elige_por_calibracion_no_por_test():
    y = np.full(8, 100.0)
    cv = pd.DataFrame(
        {
            "y": y,
            "periodo": ["calibracion"] * 4 + ["test"] * 4,
            # A es el mejor en calibración; C es el mejor en test
            "A": [101, 101, 101, 101, 120, 120, 120, 120],
            "B": [105, 105, 105, 105, 110, 110, 110, 110],
            "C": [130, 130, 130, 130, 100, 100, 100, 100],
        }
    )
    miembros = ensemble.seleccionar(cv, ["A", "B", "C"], k=2)
    assert miembros == ["A", "B"]
    out = ensemble.agregar(cv, miembros)
    assert out[ensemble.NOMBRE].iloc[0] == 103
