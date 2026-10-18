import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mlforecast")
from forecast_energia.pipelines import pronostico as P  # noqa: E402


def _obs(fin: str, series=("SIN", "BENI")) -> pd.DataFrame:
    ds = pd.date_range("2026-10-01", fin, freq="h")
    return pd.concat(
        [pd.DataFrame({"unique_id": u, "ds": ds, "y": 1.0}) for u in series], ignore_index=True
    )


def test_corte_es_el_ultimo_dia_completo_de_todas_las_series():
    assert P.corte_actual(_obs("2026-10-16 23:00")) == pd.Timestamp("2026-10-16 23:00")
    assert P.corte_actual(_obs("2026-10-17 10:00")) == pd.Timestamp("2026-10-16 23:00")
    # Si una serie viene atrasada, manda la más atrasada.
    obs = _obs("2026-10-16 23:00")
    obs = obs[~((obs["unique_id"] == "BENI") & (obs["ds"] > "2026-10-16 05:00"))]
    assert P.corte_actual(obs) == pd.Timestamp("2026-10-15 23:00")


def _pred(ds: pd.DatetimeIndex, base: float) -> pd.DataFrame:
    return pd.DataFrame({"unique_id": "SIN", "ds": ds, "Base": base, "cutoff": ds[0]})


def test_historico_conserva_el_pronostico_de_mayor_anticipacion():
    ayer = _pred(pd.date_range("2026-10-17", periods=48, freq="h"), 100.0)
    hist = P.actualizar_historico(ayer.iloc[:0].reindex(columns=P.COLUMNAS_HISTORICO), ayer)
    hoy = _pred(pd.date_range("2026-10-18", periods=48, freq="h"), 200.0)
    hist = P.actualizar_historico(hist, hoy)
    assert len(hist) == 72  # 17, 18 y 19 de octubre
    # El 18 ya se había emitido ayer (day-ahead): no se pisa con el de menor anticipación.
    assert (hist.loc[hist["ds"].dt.day == 18, "Base"] == 100.0).all()
    assert (hist.loc[hist["ds"].dt.day == 19, "Base"] == 200.0).all()


def test_sin_predespacho_la_base_del_sin_es_el_ensamble(monkeypatch):
    ds = pd.date_range("2026-10-17", periods=48, freq="h")
    pred = pd.concat(
        [
            pd.DataFrame({"unique_id": u, "ds": ds, "Ensamble": 1000.0})
            for u in ("SIN", "SANTA CRUZ")
        ],
        ignore_index=True,
    )
    hist_ds = pd.date_range("2026-07-01", "2026-10-16 23:00", freq="h")
    obs = pd.DataFrame({"unique_id": "SIN", "ds": hist_ds, "y": 1000.0})
    gen = pd.DataFrame({"unique_id": "PREV", "ds": hist_ds, "y": 990.0})
    monkeypatch.setattr(P.io, "cargar_generacion", lambda: gen)

    # El CNDC solo publicó el día 17: el 18 queda sin PREV.
    prev = pd.Series(1100.0, index=pd.date_range("2026-10-17", periods=24, freq="h"))
    monkeypatch.setattr(P.cndc, "prev_predespacho", lambda dias: prev)
    out = P.agregar_prev(pred.copy(), obs, pd.Timestamp("2026-10-16 23:00"))
    sin = out[out["unique_id"] == "SIN"].set_index("ds")
    # Día 17: media del ensamble (1000) y PREV corregido (1100 + 10 de sesgo) = 1055.
    np.testing.assert_allclose(sin.loc["2026-10-17", "Base"], 1055.0)
    np.testing.assert_allclose(sin.loc["2026-10-18", "Base"], 1000.0)
    depto = out[out["unique_id"] == "SANTA CRUZ"]
    assert depto["PREV"].isna().all()
    assert (depto["Base"] == 1000.0).all()
