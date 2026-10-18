import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from forecast_energia.api import main  # noqa: E402

PUNTO = main.PUNTO


def _pronostico(ds: pd.DatetimeIndex) -> pd.DataFrame:
    filas = []
    for u in ["SIN", "SANTA CRUZ"]:
        punto = np.linspace(1000, 1500, len(ds))
        df = pd.DataFrame({"unique_id": u, "ds": ds, PUNTO: punto})
        for n, w in [(80, 50), (90, 70), (95, 90)]:
            df[f"{PUNTO}-lo-{n}"] = punto - w
            df[f"{PUNTO}-hi-{n}"] = punto + w
        df["PREV"] = punto + 10 if u == "SIN" else np.nan
        filas.append(df)
    out = pd.concat(filas, ignore_index=True)
    return out.assign(corte=pd.Timestamp("2026-10-16 23:00"), emitido=pd.Timestamp("2026-10-17"))


@pytest.fixture
def cliente(tmp_path, monkeypatch):
    ds = pd.date_range("2026-10-17", periods=48, freq="h")
    _pronostico(ds).to_parquet(tmp_path / "pronostico_actual.parquet", index=False)
    bt = _pronostico(pd.date_range("2026-10-01", periods=24 * 16, freq="h"))
    bt["y"] = bt[PUNTO] + 5
    bt.to_parquet(tmp_path / "backtest_reciente.parquet", index=False)
    (tmp_path / "metricas.json").write_text(json.dumps({"corte": "2026-10-16 23:00:00"}))
    monkeypatch.setenv("FORECAST_DEMO_DIR", str(tmp_path))
    return TestClient(main.app)


def test_health(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json()["dias"] == ["2026-10-17", "2026-10-18"]


def test_forecast_por_defecto_es_el_ultimo_dia(cliente):
    r = cliente.get("/forecast", params={"unidad": "SIN"})
    assert r.status_code == 200
    body = r.json()
    assert body["fecha"] == "2026-10-18"
    assert len(body["horas"]) == 24
    h = body["horas"][0]
    assert h["ds"] == "2026-10-18T00:00"
    assert h["lo_95"] < h["lo_80"] < h["punto"] < h["hi_80"] < h["hi_95"]
    assert "PREV" in h


def test_forecast_departamento_y_fecha(cliente):
    r = cliente.get("/forecast", params={"unidad": "santa_cruz", "fecha": "2026-10-17"})
    assert r.status_code == 200
    assert r.json()["unidad"] == "SANTA CRUZ"
    assert "PREV" not in r.json()["horas"][0]


def test_errores_claros(cliente):
    assert cliente.get("/forecast", params={"unidad": "ATLANTIS"}).status_code == 404
    assert cliente.get("/forecast", params={"fecha": "2030-01-01"}).status_code == 404


def test_backtest_filtra_fechas(cliente):
    r = cliente.get("/backtest", params={"desde": "2026-10-10", "hasta": "2026-10-11"})
    assert r.status_code == 200
    assert len(r.json()["horas"]) == 48


def test_metrics(cliente):
    assert cliente.get("/metrics").json()["corte"] == "2026-10-16 23:00:00"


def test_sin_artefactos_responde_503(tmp_path, monkeypatch):
    monkeypatch.setenv("FORECAST_DEMO_DIR", str(tmp_path / "vacio"))
    assert TestClient(main.app).get("/health").status_code == 503


def test_lee_desde_url_remota(cliente, tmp_path, monkeypatch):
    # Los artefactos solo están en la "URL" (file://): la carpeta local está vacía.
    monkeypatch.setenv("FORECAST_DEMO_URL", tmp_path.as_uri())
    monkeypatch.setenv("FORECAST_DEMO_DIR", str(tmp_path / "vacio"))
    main._descargar.cache_clear()
    assert cliente.get("/health").status_code == 200


def test_url_caida_usa_la_copia_local(cliente, monkeypatch):
    monkeypatch.setenv("FORECAST_DEMO_URL", "http://127.0.0.1:9/no-existe")
    main._descargar.cache_clear()
    assert cliente.get("/forecast").status_code == 200
