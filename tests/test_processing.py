import numpy as np
import pandas as pd

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


def _deptos_sinteticos(dias=21, seed=0):
    """Dos departamentos con perfil diario estable y TOT = 1.03 × su suma."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-03-01", periods=24 * dias, freq="h")
    perfil = 100 + 30 * np.sin(2 * np.pi * idx.hour / 24)
    dep = pd.DataFrame(
        {"A": perfil * 3 + rng.normal(0, 2, len(idx)), "B": perfil + rng.normal(0, 1, len(idx))},
        index=idx,
    )
    tot = dep.sum(axis=1) * 1.03
    return dep, tot


def test_ajustar_a_tot_cierra_la_jerarquia():
    dep, tot = _deptos_sinteticos()
    ajustado = processing.ajustar_a_tot(dep, tot)
    np.testing.assert_allclose(ajustado.sum(axis=1), tot)
    # La participación de cada departamento no cambia
    np.testing.assert_allclose(ajustado["A"] / ajustado["B"], dep["A"] / dep["B"])


def test_detecta_caida_de_un_departamento():
    dep, tot = _deptos_sinteticos()
    t = dep.index[24 * 15 + 10]
    dep.loc[t, "B"] = 0.5  # falla de medición
    mascara = processing.detectar_anomalias_departamentales(dep, tot)
    assert mascara.loc[t, "B"]
    assert mascara.drop(index=t).to_numpy().sum() == 0


def test_detecta_hora_incoherente_con_tot():
    dep, tot = _deptos_sinteticos()
    t = dep.index[24 * 15 + 10]
    tot.loc[t] *= 1.2  # Σ deptos / TOT se aleja 17 % de su nivel
    mascara = processing.detectar_anomalias_departamentales(dep, tot)
    assert mascara.loc[t].all()


def test_imputar_por_participacion_usa_tot():
    dep, tot = _deptos_sinteticos()
    t = dep.index[24 * 15 + 10]
    real = dep.loc[t, "B"]
    dep.loc[t, "B"] = np.nan
    mascara = pd.DataFrame(False, index=dep.index, columns=dep.columns)
    imputado = processing.imputar_por_participacion(dep, tot, mascara)
    assert imputado.notna().all().all()
    assert abs(imputado.loc[t, "B"] - real) / real < 0.05
    pd.testing.assert_frame_equal(imputado.drop(index=t), dep.drop(index=t))
