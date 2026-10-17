# Forecasting probabilístico de la demanda eléctrica de Bolivia ⚡🇧🇴

Pronóstico **day-ahead** (24 h) de la demanda horaria del **Sistema Interconectado Nacional (SIN)** y de sus 8 departamentos, con **intervalos de predicción con garantía estadística** (Conformal Prediction). Se usan **datos reales** del operador de la red boliviana.

> 🚧 Proyecto en construcción. El plan completo está en [`docs/plan-forecasting.md`](docs/plan-forecasting.md).

## El problema

El CNDC (Comité Nacional de Despacho de Carga) programa cada día cuánta generación térmica, hidroeléctrica y de reserva se despacha para el día siguiente. Un pronóstico puntual sin medida de incertidumbre obliga a elegir entre dos males: **sobredimensionar la reserva** (más gas quemado y más costo) o **arriesgar déficit**. Este proyecto entrega cuantiles calibrados que buscan superar al pronóstico oficial del CNDC (`PREV`).

## Datos

| Fuente | Qué | Resolución | Desde |
|---|---|---|---|
| [API pública del CNDC](https://www.cndc.bo/wp-json/cndc/v1) `rt/demanda` | Demanda (MW) por departamento | 15 min | 2024-03-01 |
| [API pública del CNDC](https://www.cndc.bo/wp-json/cndc/v1) `rt/generacion` | Generación por tecnología + pronóstico oficial `PREV` | 15 min | 2024-03-01 |
| [Open-Meteo Previous Runs](https://open-meteo.com/en/docs/previous-runs-api) | Clima de las 8 capitales: observado y **pronosticado el día anterior** (sin leakage) | 1 h | 2024-03-01 |
| [`holidays`](https://pypi.org/project/holidays/) | Feriados nacionales y departamentales de Bolivia | día | — |

## Cómo reproducir

```bash
uv sync --all-groups          # entorno completo (Python 3.13)
uv run poe data               # descarga incremental → data/processed/*.parquet
uv run poe test               # tests
uv run poe lint               # ruff
```

Salidas de `poe data` (formato largo de Nixtla `unique_id, ds, y`, con `ds` = inicio de la hora local de Bolivia):

- `demanda_horaria.parquet`: 8 departamentos + `SIN` (MW).
- `generacion_horaria.parquet`: `TOT`, `TERMO`, `HIDRO`, `SOLAR`, `EOL`, `BAGAZO`, `RENO`, `PREV` (MW).
- `clima_horario.parquet`: clima por departamento; las columnas `*_d1` son el pronóstico emitido el día anterior.
- `calidad.json`: faltantes, imputación y chequeos de coherencia.

## Estructura

```
src/forecast_energia/
  config.py            rutas, endpoints, departamentos
  data/cndc.py         cliente API CNDC con caché mensual
  data/weather.py      cliente Open-Meteo
  data/calendar.py     feriados y calendario
  data/processing.py   limpieza, interpolación de huecos cortos, agregación horaria
  data/build.py        pipeline de datos (CLI)
tests/                 pytest (con respuestas reales de la API como fixtures)
docs/                  plan del proyecto
```
