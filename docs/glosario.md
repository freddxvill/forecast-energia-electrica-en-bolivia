# Glosario de métodos

Explicación breve, para un lector no especialista, de los términos que aparecen en el proyecto. Cada entrada dice **qué es**, **para qué se usó aquí** (con el archivo del código) y dónde **saber más** (el paper original o la documentación oficial).

Índice: [Pronóstico y evaluación](#pronóstico-y-evaluación) · [Métricas](#métricas) · [Modelos](#modelos) · [Incertidumbre](#incertidumbre) · [Jerarquía](#jerarquía) · [Dominio eléctrico](#dominio-eléctrico) · [Infraestructura](#infraestructura)

---

## Pronóstico y evaluación

### Pronóstico day-ahead
Pronóstico del día siguiente, hora por hora (24 valores), hecho con anticipación para poder programar la generación.
- **Aquí:** es la tarea central. Se pronostica la demanda horaria del día D del SIN y de sus 8 departamentos, igual que el predespacho del CNDC.
- **Saber más:** Hong y Fan (2016), [*Probabilistic electric load forecasting: A tutorial review*](https://doi.org/10.1016/j.ijforecast.2015.11.011), *International Journal of Forecasting*.

### Horizonte y corte (D−2)
El **corte** es el último instante cuyos datos se pueden usar; el **horizonte** es cuántas horas después del corte cae lo que se pronostica.
- **Aquí:** el corte es D−2 a las 23:00, porque el CNDC publica el predespacho con 1–2 días de anticipación. Cada pronóstico cubre 48 h y se evalúan las horas 25–48, que son el día D (`evaluation.filtrar_dia_objetivo`, en [`evaluation.py`](../src/forecast_energia/evaluation.py)).
- **Saber más:** [Hyndman y Athanasopoulos, *Forecasting: Principles and Practice* (FPP3), cap. 5](https://otexts.com/fpp3/toolbox.html).

### Backtest rolling-origin
Simular el pasado como si se pronosticara en tiempo real: se corta en un día, se entrena con lo anterior, se pronostica, se avanza un día y se repite. Así cada pronóstico usa solo información disponible en su momento.
- **Aquí:** 274 ventanas diarias (`cross_validation(h=48, step_size=24)` de Nixtla) en [`pipelines/backtest.py`](../src/forecast_energia/pipelines/backtest.py).
- **Saber más:** [FPP3, sección 5.10 (validación cruzada de series de tiempo)](https://otexts.com/fpp3/tscv.html); Tashman (2000), [*Out-of-sample tests of forecasting accuracy*](https://doi.org/10.1016/S0169-2070%2800%2900065-0).

### Partición entrenamiento / calibración / test / prospectivo
Dividir el tiempo en tramos con funciones distintas: **entrenamiento** (ajustar modelos), **calibración** (elegir modelos y calibrar intervalos), **test** (medir, sin tocar) y **prospectivo** (datos que llegaron después de fijar el test, para validar con datos nunca vistos).
- **Aquí:** entrenamiento < 2026-01-16; calibración 2026-01-16 → 2026-04-16 (91 días); test 2026-04-17 → 2026-10-16 (183 días); prospectivo, lo posterior (`evaluation.Particion`, `config.TEST_FIN`).
- **Saber más:** [FPP3, sección 5.8](https://otexts.com/fpp3/accuracy.html).

### Data leakage (fuga de información)
Usar, sin querer, información que no se habría conocido al momento de pronosticar. Infla las métricas y el modelo falla en producción.
- **Aquí:** se detectó y corrigió un caso: el clima "pronosticado 24 h antes" usaba corridas posteriores al corte; se cambió por el pronosticado 48 h antes. Hay tests de no-leakage en [`tests/`](../tests).
- **Saber más:** Kaufman et al. (2012), [*Leakage in data mining*](https://doi.org/10.1145/2382577.2382579), *ACM TKDD*.

### Data snooping
Elegir el modelo o la idea mirando el mismo conjunto con el que después se reporta el resultado. El número final queda optimista porque se "eligió al ganador" del test.
- **Aquí:** los miembros del ensamble y el método de reconciliación se eligieron con el periodo de calibración, no con el test. La corrección de sesgo de PREV surgió mirando el test, y por eso se validó aparte en calibración y queda pendiente la validación prospectiva.
- **Saber más:** White (2000), [*A reality check for data snooping*](https://doi.org/10.1111/1468-0262.00152), *Econometrica*.

---

## Métricas

### MAE (error absoluto medio)
Promedio de |real − pronóstico|, en las mismas unidades que la serie (MW). Es la métrica principal del proyecto porque se interpreta directo como "cuántos MW me equivoco en promedio".
- **Aquí:** [`evaluation.metricas_puntuales`](../src/forecast_energia/evaluation.py). Multiplicado por las horas del año da la energía mal programada (GWh/año) del impacto.
- **Saber más:** [FPP3, sección 5.8](https://otexts.com/fpp3/accuracy.html).

### RMSE (raíz del error cuadrático medio)
Como el MAE, pero eleva los errores al cuadrado antes de promediar: castiga más los errores grandes.
- **Aquí:** se reporta junto al MAE en `reports/metricas_test_SIN.csv`.
- **Saber más:** [FPP3, sección 5.8](https://otexts.com/fpp3/accuracy.html).

### MAPE (error porcentual absoluto medio)
Promedio de |error| / real, en %. Permite comparar series de tamaños distintos (el SIN contra Beni).
- **Aquí:** se usa para comparar departamentos en la jerarquía.
- **Saber más:** [FPP3, sección 5.8](https://otexts.com/fpp3/accuracy.html).

### Sesgo
Promedio del error con signo (pronóstico − real). Un sesgo positivo significa que el modelo sobrestima sistemáticamente.
- **Aquí:** el ensamble tiene sesgo ~0; PREV tiene sesgos fuertes según la hora (subestima a las 18 h, sobrestima de 21 a 23 h).

### MASE (error absoluto medio escalado)
El MAE dividido por el MAE de un pronóstico ingenuo de referencia. MASE < 1 significa que el modelo le gana a esa referencia; sirve para comparar series sin depender de su escala.
- **Aquí:** la referencia es el naive de 168 h medido antes de la calibración (`evaluation.escala_mase`). El MASE medio de las 9 series eligió el método de reconciliación.
- **Saber más:** Hyndman y Koehler (2006), [*Another look at measures of forecast accuracy*](https://doi.org/10.1016/j.ijforecast.2006.03.001).

### Pinball loss (pérdida cuantílica)
Mide qué tan bueno es un cuantil pronosticado. Penaliza asimétricamente: para el cuantil 95, quedarse corto cuesta 19 veces más que pasarse.
- **Aquí:** `evaluation.pinball`. Es la base del CRPS y de la pérdida con que se entrenan NHITS y TFT.
- **Saber más:** [FPP3, sección 5.9](https://otexts.com/fpp3/distaccuracy.html).

### Winkler score
Evalúa un intervalo en una sola cifra: su ancho más una penalización cuando el valor real cae afuera. Premia intervalos angostos que igual aciertan.
- **Aquí:** `evaluation.winkler`, en la tabla de intervalos de la Fase 5.
- **Saber más:** Winkler (1972), [*A decision-theoretic approach to interval estimation*](https://doi.org/10.1080/01621459.1972.10481224), *JASA*; [FPP3, sección 5.9](https://otexts.com/fpp3/distaccuracy.html).

### CRPS (Continuous Ranked Probability Score)
Generaliza el MAE a pronósticos probabilísticos: compara toda la distribución pronosticada con el valor real. Menor es mejor, y se mide en MW.
- **Aquí:** se aproxima promediando la pinball loss de los cuantiles disponibles (`evaluation.crps_desde_cuantiles`).
- **Saber más:** Gneiting y Raftery (2007), [*Strictly proper scoring rules, prediction, and estimation*](https://doi.org/10.1198/016214506000001437), *JASA*.

### Cobertura
Porcentaje de horas en que el valor real cayó dentro del intervalo. Un intervalo del 90 % bien calibrado cubre ~90 %: si cubre menos, subestima la incertidumbre; si cubre mucho más, es innecesariamente ancho.
- **Aquí:** se mide global, por hora del día y por mes (`evaluation.cobertura_por_grupo`), porque un intervalo puede acertar en promedio y fallar justo en la hora pico.

### Test de Diebold-Mariano (con corrección HLN)
Test estadístico que responde "¿la diferencia de error entre dos pronósticos es real o es suerte?". La corrección de Harvey, Leybourne y Newbold (HLN) lo ajusta para muestras finitas.
- **Aquí:** `evaluation.diebold_mariano`, aplicado a pérdidas **diarias** (las horas de un mismo día están correlacionadas). Todas las mejoras principales frente a PREV tienen p < 0.001.
- **Saber más:** Diebold y Mariano (1995), [*Comparing predictive accuracy*](https://doi.org/10.1080/07350015.1995.10524599); Harvey, Leybourne y Newbold (1997), [*Testing the equality of prediction mean squared errors*](https://doi.org/10.1016/S0169-2070%2896%2900719-4).

---

## Modelos

### Naive estacional
El pronóstico más simple posible: "mañana a las 19 h será igual que hace una semana a las 19 h". Cualquier modelo serio tiene que ganarle.
- **Aquí:** naive de 168 h (una semana) y de 48 h en [`models/baselines.py`](../src/forecast_energia/models/baselines.py). El de 24 h no es válido con el corte en D−2.
- **Saber más:** [FPP3, sección 5.2](https://otexts.com/fpp3/simple-methods.html).

### ETS / AutoETS
Suavizado exponencial: el pronóstico es un promedio ponderado del pasado, con más peso en lo reciente, separando nivel, tendencia y estacionalidad (Error, Trend, Seasonal). *AutoETS* elige la variante automáticamente.
- **Aquí:** modela la parte desestacionalizada dentro de MSTL.
- **Saber más:** [FPP3, cap. 8](https://otexts.com/fpp3/expsmooth.html).

### MSTL
Descompone la serie en varias estacionalidades (diaria de 24 h y semanal de 168 h), una tendencia y un resto, y pronostica cada parte.
- **Aquí:** MSTL(24, 168) + AutoETS en [`models/stats.py`](../src/forecast_energia/models/stats.py), con las últimas 8 semanas (con toda la historia era demasiado lento).
- **Saber más:** Bandara, Hyndman y Bergmeir (2021), [*MSTL: A seasonal-trend decomposition algorithm for time series with multiple seasonal patterns*](https://arxiv.org/abs/2107.13462); [StatsForecast](https://nixtlaverse.nixtla.io/statsforecast/).

### LightGBM (gradient boosting)
Modelo de árboles de decisión que se construyen uno tras otro, cada uno corrigiendo los errores del anterior. Muy eficaz con datos tabulares y rápido en CPU.
- **Aquí:** usa como variables los valores pasados (lags de 48 h o más), el calendario y el clima pronosticado ([`models/ml.py`](../src/forecast_energia/models/ml.py)). Se reentrena cada día en producción (~20 s).
- **Saber más:** Ke et al. (2017), [*LightGBM: A highly efficient gradient boosting decision tree*](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree), NeurIPS; [documentación](https://lightgbm.readthedocs.io/).

### MLForecast
Librería de Nixtla que convierte una serie de tiempo en una tabla (lags, calendario) para entrenar modelos de machine learning como LightGBM, y maneja el pronóstico multi-paso.
- **Aquí:** envuelve a LightGBM en `models/ml.py`.
- **Saber más:** [MLForecast](https://nixtlaverse.nixtla.io/mlforecast/).

### Normalización por nivel
Dividir la demanda por su nivel reciente antes de modelar y multiplicar de vuelta al final. El modelo aprende la *forma* del día y no el nivel absoluto, que cambia con el crecimiento de la demanda.
- **Aquí:** `y / nivel`, con nivel = media de la semana que termina 48 h antes (`ml.nivel`). Los árboles no extrapolan: sin esto, LightGBM tenía −57 MW de sesgo en calibración.
- **Saber más:** es la misma idea que el escalado por serie de DeepAR: Salinas et al. (2020), [*DeepAR*](https://doi.org/10.1016/j.ijforecast.2019.07.001).

### NHITS
Red neuronal para series de tiempo que pronostica por bloques a distintas resoluciones (de lo grueso a lo fino), lo que la hace rápida y buena en horizontes largos.
- **Aquí:** candidata en el backtest, entrenada en Kaggle con GPU ([`models/neural.py`](../src/forecast_energia/models/neural.py)). Quedó por detrás de LightGBM y TFT.
- **Saber más:** Challu et al. (2023), [*NHITS: Neural hierarchical interpolation for time series forecasting*](https://arxiv.org/abs/2201.12886), AAAI.

### TFT (Temporal Fusion Transformer)
Red neuronal con mecanismos de atención que combina el pasado de la serie con variables conocidas a futuro (calendario, clima pronosticado) y elige qué variables importan en cada momento.
- **Aquí:** el mejor modelo individual (MAE 47.3 MW). Es miembro del ensamble de producción; se reentrena cada semana y se infiere cada día desde un checkpoint (`models/neural.py`).
- **Saber más:** Lim et al. (2021), [*Temporal Fusion Transformers for interpretable multi-horizon time series forecasting*](https://arxiv.org/abs/1912.09363), *IJF*; [NeuralForecast](https://nixtlaverse.nixtla.io/neuralforecast/).

### MQLoss / regresión cuantílica
En vez de pronosticar un solo número, el modelo pronostica varios cuantiles (p. ej., 5 %, 50 % y 95 %) a la vez, entrenando con la pinball loss de cada uno. *MQLoss* es la versión multi-cuantil de NeuralForecast.
- **Aquí:** NHITS y TFT se entrenan con MQLoss para los niveles 80/90/95. Sus cuantiles crudos resultaron mal calibrados (el 90 % de TFT cubría el 60 %), lo que motivó el conformal.
- **Saber más:** Koenker y Bassett (1978), [*Regression quantiles*](https://doi.org/10.2307/1913643), *Econometrica*.

### Ensamble
Promediar los pronósticos de varios modelos. Los errores de modelos distintos se compensan en parte, y el promedio suele ganarle a cada miembro.
- **Aquí:** media de los 2 mejores modelos **según la calibración**: LightGBM + TFT ([`models/ensemble.py`](../src/forecast_energia/models/ensemble.py)). MAE 44.0 MW, −20 % vs PREV.
- **Saber más:** Wang et al. (2023), [*Forecast combinations: An over 50-year review*](https://doi.org/10.1016/j.ijforecast.2022.11.005), *IJF*.

### Combinación de pronósticos
Lo mismo que un ensamble, pero mezclando pronósticos de fuentes distintas; aquí, el modelo propio con el pronóstico oficial.
- **Aquí:** Combinado = media del Ensamble y del PREV corregido ([`models/combinacion.py`](../src/forecast_energia/models/combinacion.py)). MAE 38.3 MW, −30 % vs PREV.
- **Saber más:** Bates y Granger (1969), [*The combination of forecasts*](https://doi.org/10.2307/3008764).

### Corrección de sesgo por hora
Restar al pronóstico el error típico que tuvo, en los días recientes, a esa misma hora. Si PREV suele quedarse 47 MW corto a las 18 h, se le suman esos 47 MW.
- **Aquí:** mediana del error de PREV por hora en los últimos 91 días hasta el corte (`combinacion.sesgo_reciente`). Por sí sola baja el MAE de PREV de 54.8 a 42.7 MW.
- **Saber más:** es la idea de *Model Output Statistics* de la meteorología: Glahn y Lowry (1972), [*The use of Model Output Statistics (MOS) in objective weather forecasting*](https://doi.org/10.1175/1520-0450%281972%29011%3C1203:TUOMOS%3E2.0.CO;2).

---

## Incertidumbre

### Intervalo de predicción
Rango dentro del cual se espera que caiga el valor real con cierta probabilidad (80, 90 o 95 %). Es lo que permite dimensionar la reserva: un número solo no dice cuánto puede equivocarse.
- **Aquí:** cada pronóstico trae intervalos del 80/90/95 %; la app muestra las bandas del 80 y del 95 %.
- **Saber más:** [FPP3, sección 5.5](https://otexts.com/fpp3/prediction-intervals.html).

### Conformal prediction (split conformal)
Método para construir intervalos con **garantía de cobertura** sin suponer una distribución: se miden los errores del modelo en un conjunto de calibración y el intervalo del 90 % es el pronóstico ± el cuantil 90 de esos errores (con una pequeña corrección por tamaño de muestra).
- **Aquí:** versión asimétrica (cuantiles separados para errores positivos y negativos), con los últimos 91 días ([`conformal.py`](../src/forecast_energia/conformal.py), `split_conformal`).
- **Saber más:** Angelopoulos y Bates (2021), [*A gentle introduction to conformal prediction*](https://arxiv.org/abs/2107.07511); Lei et al. (2018), [*Distribution-free predictive inference for regression*](https://doi.org/10.1080/01621459.2017.1307116).

### Mondrian por hora
Hacer conformal por separado dentro de cada grupo; aquí, un calibrador por hora del día. Así la cobertura vale en cada hora y no solo en promedio.
- **Aquí:** sin Mondrian, el intervalo del 90 % cubría solo el 72 % al mediodía.
- **Saber más:** Vovk, Gammerman y Shafer (2005), [*Algorithmic Learning in a Random World*](https://doi.org/10.1007/b106715), cap. 4.

### CQR (Conformalized Quantile Regression)
Conformal aplicado sobre los cuantiles que ya produce un modelo: los ensancha o angosta lo justo para que cubran lo nominal, manteniendo su forma.
- **Aquí:** corrige los cuantiles de TFT (de 60 % a 90 % de cobertura), en `conformal.cqr`.
- **Saber más:** Romano, Patterson y Candès (2019), [*Conformalized quantile regression*](https://arxiv.org/abs/1905.03222), NeurIPS.

### ACI (Adaptive Conformal Inference)
Ajuste en línea del conformal para cuando los datos cambian: si los últimos intervalos fallaron más de lo esperado, el siguiente se ensancha, y viceversa.
- **Aquí:** `conformal.aci`, con γ = 0.05 y la retroalimentación retrasada 2 días (el error del día D se conoce recién después). Sube la cobertura del peor mes de 72 % a 85 %.
- **Saber más:** Gibbs y Candès (2021), [*Adaptive conformal inference under distribution shift*](https://arxiv.org/abs/2106.00170), NeurIPS.

---

## Jerarquía

### Pronóstico jerárquico
Pronosticar series que se suman entre sí (los 8 departamentos suman el SIN) de forma que los pronósticos también sumen.
- **Aquí:** SIN → 8 departamentos, en [`hierarchy.py`](../src/forecast_energia/hierarchy.py) y [`pipelines/jerarquia.py`](../src/forecast_energia/pipelines/jerarquia.py).
- **Saber más:** [FPP3, cap. 11](https://otexts.com/fpp3/hierarchical.html).

### Coherencia
Un conjunto de pronósticos es coherente si respeta la suma: Σ departamentos = SIN. Pronosticar cada serie por separado casi nunca lo logra.
- **Aquí:** los pronósticos base diferían en 23 MW de media; los reconciliados cuadran exactamente.

### BottomUp
Pronosticar solo los departamentos y obtener el SIN sumándolos.
- **Aquí:** comparado en la Fase 6; empeora el SIN (pierde la información del CNDC que trae su pronóstico).
- **Saber más:** [FPP3, sección 11.2](https://otexts.com/fpp3/single-level.html).

### TopDown
Pronosticar solo el SIN y repartirlo entre departamentos según su participación histórica.
- **Aquí:** fue el peor método: proporciones fijas ignoran el clima propio de cada departamento.
- **Saber más:** [FPP3, sección 11.2](https://otexts.com/fpp3/single-level.html).

### MinT (WLS / shrink)
Reconciliación óptima: toma los pronósticos de todos los niveles y los ajusta lo mínimo necesario para que sumen, dando más peso a las series que se pronostican mejor. *WLS* usa solo las varianzas de los errores; *shrink* usa la matriz de covarianzas completa, estabilizada.
- **Aquí:** MinT-WLS, elegido por calibración. Lleva a Santa Cruz la información del pronóstico del SIN y baja su MAPE de 5.9 % a 5.6 % (p = 0.002).
- **Saber más:** Wickramasuriya, Athanasopoulos y Hyndman (2019), [*Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization*](https://doi.org/10.1080/01621459.2018.1448825), *JASA*; Schäfer y Strimmer (2005), [*A shrinkage approach to large-scale covariance matrix estimation*](https://doi.org/10.2202/1544-6115.1175).

---

## Dominio eléctrico

### SIN (Sistema Interconectado Nacional)
La red eléctrica principal de Bolivia, que conecta las centrales con los consumidores. Los datos del CNDC la desagregan en 8 departamentos (Pando no aparece en ellos).
- **Aquí:** la serie principal. Su demanda se iguala a la generación total `TOT`.
- **Saber más:** [CNDC](https://www.cndc.bo).

### CNDC (Comité Nacional de Despacho de Carga)
El operador del SIN: decide qué centrales generan, cuánto y cuándo, y publica los datos de operación.
- **Aquí:** fuente de todos los datos eléctricos, a través de su API pública ([`data/cndc.py`](../src/forecast_energia/data/cndc.py)).
- **Saber más:** [www.cndc.bo](https://www.cndc.bo).

### Predespacho (PREV)
La programación horaria que hace el CNDC para el día siguiente, a partir de su propio pronóstico de demanda. Es el benchmark a batir.
- **Aquí:** `PREV` sale de `rt/generacion` (histórico) y de `precmp` (el de mañana, en producción). Error en test: MAE 54.8 MW, MAPE 4.09 %.

### Reserva operativa
Capacidad de generación que se deja disponible, sin usar, para cubrir lo que el pronóstico no anticipó (más demanda o una central que sale de servicio).
- **Aquí:** se dimensiona como el límite superior del intervalo del 90 % menos el pronóstico (cubre el 95 % de los desvíos al alza). Varía por hora: ~47 MW de madrugada, ~105 MW a media tarde, contra 116 MW fijos con la regla sobre PREV ([`impacto.py`](../src/forecast_energia/impacto.py)).
- **Saber más:** Hong y Fan (2016), ver [pronóstico day-ahead](#pronóstico-day-ahead).

### Costo marginal
El costo de producir el último MWh necesario en una hora: el de la central más cara que está despachada. En Bolivia es bajo (~15–20 USD/MWh) porque el gas para generación tiene precio regulado.
- **Aquí:** uno de los precios con que se valorizan los desvíos evitados (`cm-diario`).

### Precio monómico
Precio medio que paga el mercado mayorista por MWh, sumando energía, potencia y transporte en un solo valor. El CNDC lo publica por mes.
- **Aquí:** `dashboard/precios` aporta el precio de energía (escenario de tarifa regulada) y el de potencia (USD/kW-mes, para valorizar la reserva).
- **Saber más:** [CNDC](https://www.cndc.bo).

### Heat rate (consumo específico)
Cuánta energía de combustible necesita una central térmica para producir 1 MWh eléctrico, en MMBtu/MWh. Menor es más eficiente.
- **Aquí:** con 9–12 MMBtu/MWh (turbinas de ciclo abierto) se convierte cada MWh de desvío en gas que podría exportarse; acotado con el costo marginal del CNDC (≤ 13.9).
- **Saber más:** [EIA, *heat rate*](https://www.eia.gov/tools/faqs/faq.php?id=107&t=3).

### MMBtu
Millón de BTU (unidades térmicas británicas), la unidad en que se comercializa el gas natural. 1 MMBtu ≈ 1.055 GJ ≈ 1 000 pies cúbicos de gas.
- **Aquí:** precio de exportación del gas a Brasil, ~6.5 USD/MMBtu en el escenario central; el regulado interno es ~1.3.

---

## Infraestructura

### Exógenas futuras
Variables externas a la serie cuyo valor se conoce (o se pronostica) para el periodo a pronosticar: el calendario se sabe de antemano; el clima, por su pronóstico.
- **Aquí:** hora, día de semana, feriados nacionales y departamentales y clima pronosticado (`*_d2`) en [`features.py`](../src/forecast_energia/features.py). Para el SIN, el clima se pondera por la participación de cada departamento.
- **Saber más:** [FPP3, cap. 10 (regresión dinámica)](https://otexts.com/fpp3/dynamic.html).

### Open-Meteo Previous Runs
API gratuita de Open-Meteo que guarda los pronósticos del tiempo tal como se emitieron días antes (no solo el valor observado). Permite entrenar con el clima que realmente se conocía al pronosticar.
- **Aquí:** las columnas `*_d2` son el pronóstico hecho 48 h antes de cada hora, el único libre de leakage con el corte en D−2 ([`data/weather.py`](../src/forecast_energia/data/weather.py)).
- **Saber más:** [documentación de Previous Runs](https://open-meteo.com/en/docs/previous-runs-api) (licencia CC BY 4.0).

### Checkpoint
Archivo con los pesos de un modelo ya entrenado, para usarlo sin reentrenar.
- **Aquí:** el TFT (2.8 MB) se reentrena cada semana y el pronóstico diario lo carga y solo infiere (10 s en CPU, contra 32 min de entrenamiento). Vive en la rama `demo-data`.
- **Saber más:** [NeuralForecast (`save` / `load`)](https://nixtlaverse.nixtla.io/neuralforecast/).

### FastAPI
Framework de Python para construir APIs web, con documentación interactiva generada sola (`/docs`).
- **Aquí:** expone `/forecast`, `/backtest`, `/metrics` y `/health` ([`api/main.py`](../src/forecast_energia/api/main.py)), desplegada en Hugging Face Spaces.
- **Saber más:** [fastapi.tiangolo.com](https://fastapi.tiangolo.com/).

### Streamlit
Framework de Python para hacer aplicaciones web de datos con pocas líneas, sin escribir HTML ni JavaScript.
- **Aquí:** la demo pública ([`app/streamlit_app.py`](../app/streamlit_app.py)), en Streamlit Community Cloud.
- **Saber más:** [docs.streamlit.io](https://docs.streamlit.io/).

### GitHub Actions / cron
Servicio de GitHub que ejecuta tareas automáticas en servidores de GitHub. *Cron* es la sintaxis para programarlas a horario fijo.
- **Aquí:** CI (lint y tests en cada push), pronóstico diario a las 06:00 de Bolivia y reentrenamiento semanal del TFT ([`.github/workflows/`](../.github/workflows)).
- **Saber más:** [documentación de GitHub Actions](https://docs.github.com/actions) y [eventos programados](https://docs.github.com/actions/writing-workflows/choosing-when-your-workflow-runs/events-that-trigger-workflows#schedule).
