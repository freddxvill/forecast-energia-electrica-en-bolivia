Periodo de test: 2026-04-17 → 2026-10-16 (183 días, 4,392 horas). Referencia: PREV (predespacho oficial del CNDC).

| KPI | PREV (CNDC) | Este proyecto | Mejora |
|---|---|---|---|
| Error medio (MAE) | 54.8 MW | 38.3 MW | **-30 %** |
| Energía mal programada (Σ\|error\|) | 480 GWh/año | 336 GWh/año | **-145 GWh/año** |
| Reserva al alza para cubrir el 95 % de los desvíos | 116 MW fijos (cubre 95.8 %) | 75 MW de media, variable por hora (cubre 94.3 %) | **-41 MW** (-27 MW a igual cobertura) |
| Cobertura del intervalo 90 % | sin intervalos | 89.4 % (peor hora 84 %, peor mes 84 %) | nominal ±2 pp: sí |

**Valor económico estimado (orden de magnitud, USD/año):**

| Escenario | USD/MWh | Desvíos evitados | Reserva liberada |
|---|---|---|---|
| Tarifa regulada (precio de energía del CNDC) | 21.2 | 3.1 M USD | 5.4 M USD (3.5 M USD a igual cobertura; 11.0 USD/kW-mes) |
| Costo de oportunidad del gas (bajo: 9 MMBtu/MWh × 5 USD/MMBtu) | 45 | 6.5 M USD | — |
| Costo de oportunidad del gas (central: 10.5 MMBtu/MWh × 6.5 USD/MMBtu) | 68 | 9.9 M USD | — |
| Costo de oportunidad del gas (alto: 12 MMBtu/MWh × 8 USD/MMBtu) | 96 | 13.9 M USD | — |

**De dónde sale la mejora** (MAE de test y MWh evitados frente a PREV):

| Pronóstico | MAE (MW) | MAPE | Δ vs PREV | GWh/año evitados | DM p-valor |
|---|---|---|---|---|---|
| PREV (CNDC) | 54.8 | 4.09 % | — | — | — |
| PREV corregido por hora | 42.7 | 3.19 % | -22 % | 106 | 0.000 |
| Ensamble (LightGBM + TFT) | 44.0 | 3.28 % | -20 % | 95 | 0.000 |
| Combinado (Ensamble + PREV corregido) | 38.3 | 2.86 % | -30 % | 145 | 0.000 |

**Reserva al alza** (MW sobre el pronóstico para cubrir el 95 % de las horas con demanda mayor a la pronosticada):

| Regla | Reserva media (MW) | Reserva máx. (MW) | Cobertura en test |
|---|---|---|---|
| PREV + regla fija | 116 | 116 | 95.8 % |
| PREV + conformal por hora | 81 | 228 | 95.1 % |
| Combinado + conformal por hora + ACI | 75 | 168 | 94.3 % |
| PREV + regla fija oráculo (misma cobertura) | 102 | 102 | 94.3 % |

**Supuestos y fuentes:**

| Supuesto | Valor (bajo / central / alto) | Fuente |
|---|---|---|
| Consumo específico de la unidad térmica que ajusta el desvío | 9 / 10.5 / 12 | Rango típico de turbinas a gas de ciclo abierto (el parque marginal del SIN). Chequeo con datos del CNDC: costo marginal mediano / precio regulado del gas (lo calcula pipelines.impacto). |
| Costo de oportunidad del gas: precio de exportación a Brasil | 5 / 6.5 / 8 | Contratos de YPFB con compradores brasileños indexados al Brent, ~6–7 USD/MMBtu (OilPrice.com, 2025); históricamente 4–6 USD/MMBtu. |
| Precio regulado del gas para generación termoeléctrica | 1.3 | DS 26037 (2000), vigente; ~1.3 USD/MMBtu (1 MPC ≈ 1.03 MMBtu). |
| Poder calorífico del gas natural boliviano | 1.03 | Valor típico del gas natural seco (~1 030 BTU/pie³). |
| % de los desvíos al alza (demanda > pronóstico) que cubre la reserva | 95 | Criterio del análisis (cuantil 95 unilateral = límite superior del IC 90 %). |

- Chequeo del heat rate: el costo marginal mediano del test (17.6 USD/MWh) dividido por el precio regulado del gas (1.26 USD/MMBtu) acota el consumo específico de la unidad marginal en ≤ 13.9 MMBtu/MWh (incluye O&M), coherente con el rango supuesto.
- Con el costo marginal diario del CNDC (18.6 USD/MWh ponderado; dato en el 96 % de las horas) los desvíos evitados valen 2.7 M USD/año, similar a la tarifa regulada.
- Anualizado como total del test × 365 / días de test. El test cubre abril–octubre (estación seca); no incluye el verano lluvioso.
- La reserva liberada se valoriza al precio de potencia (pago por capacidad firme) como referencia del costo de mantener capacidad disponible.
- Valorizar cada MWh de desvío supone que se cubre en tiempo real con generación térmica de ajuste. No es un ahorro contable del CNDC: es la escala del beneficio.
