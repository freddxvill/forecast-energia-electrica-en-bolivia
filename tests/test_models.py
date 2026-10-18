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


def test_correccion_de_sesgo_por_hora_sin_leakage():
    from forecast_energia.models import combinacion

    ds = pd.date_range("2026-01-01", periods=24 * 60, freq="h")
    cv = pd.DataFrame({"unique_id": "SIN", "ds": ds, "y": 100.0, "M": 90.0})  # sesgo −10
    out = combinacion.corregir_sesgo_por_hora(cv, "M", min_dias=5)
    assert out["M corregido"].iloc[-1] == 100.0  # corrige el sesgo de −10
    assert out["M corregido"].iloc[: 24 * 6].isna().all()  # 2 días de retraso + 5 de mínimo
    # Cambiar D−1 y D no altera la corrección del día D
    dia_d = ds[-1].normalize()
    alterado = cv.copy()
    alterado.loc[alterado["ds"] >= dia_d - pd.Timedelta(days=1), "y"] = 1e6
    out2 = combinacion.corregir_sesgo_por_hora(alterado, "M", min_dias=5)
    fila = cv["ds"] >= dia_d
    pd.testing.assert_series_equal(out2.loc[fila, "M corregido"], out.loc[fila, "M corregido"])


def test_sesgo_reciente_coincide_con_la_correccion_del_backtest():
    from forecast_energia.models import combinacion

    rng = np.random.default_rng(0)
    ds = pd.date_range("2026-01-01", periods=24 * 120, freq="h")
    cv = pd.DataFrame(
        {"unique_id": "SIN", "ds": ds, "y": 100 + rng.normal(0, 5, ds.size), "M": 95.0}
    )
    out = combinacion.corregir_sesgo_por_hora(cv, "M")
    dia_d = ds[-1].normalize()
    s = cv.set_index("ds")
    sesgo = combinacion.sesgo_reciente(s["y"], s["M"], hasta=dia_d - pd.Timedelta(days=2))
    ultimo = out[out["ds"] >= dia_d].set_index("ds")
    esperado = ultimo["M"] + sesgo.reindex(ultimo.index.hour).to_numpy()
    np.testing.assert_allclose(ultimo["M corregido"], esperado)
