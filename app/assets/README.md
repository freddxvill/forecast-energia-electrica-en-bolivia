# Banner de la app

`banner_mapa_sin_bolivia_v2.webp` es el banner activo, ubicado encima del título de Streamlit. Conserva el paisaje eléctrico original y superpone, en el tercio izquierdo, un mapa con los contornos departamentales de la cartografía del Órgano Electoral Plurinacional (OEP).

La geometría del mapa se dibuja directamente desde GeoJSON. Todos los departamentos comparten la misma proyección y escala: sus fronteras no se regeneran con IA ni se ajustan por separado. La capa descargada utiliza una tolerancia de 0,001 grados, inferior a un píxel a la escala del banner.

La red muestra ocho nodos: La Paz, Oruro, Potosí, Cochabamba, Chuquisaca, Tarija, Santa Cruz y Beni. Cada nodo está dentro de su departamento. Pando se muestra en gris, sin nodo ni conexión. Los enlaces eléctricos son esquemáticos.

- Formato: WebP RGB, 2715 × 905 píxeles (proporción 3:1).
- Peso: 310 776 bytes (303 KiB).
- Exportación: Chromium, calidad WebP 85; escala de pantalla 1,25 sobre el SVG de 2172 × 724.
- Uso: antes del título de `app/streamlit_app.py`, a todo el ancho del contenido.
- Se pasa como URL de datos para que Streamlit conserve el WebP al mostrarlo.
- Fondo: `footer_red_electrica_bolivia.webp`, generado previamente con `image_gen`.
- La versión anterior, `banner_mapa_sin_bolivia.webp`, se conserva como referencia y no se muestra en la app.

## Cartografía y archivos editables

Fuente: [Departamentos — Geoportal Electoral del OEP](https://geoelectoral.oep.org.bo/oep/rest/services/GeografiaElectoral/Provincia/MapServer/1).

- `cartografia/departamentos_oep.geojson`: respuesta descargada del servicio; contiene nueve departamentos y cinco entidades de lagos y salares.
- `cartografia/crear_banner.py`: genera el SVG directamente con esas coordenadas y comprueba la ubicación de los ocho nodos.
- `banner_mapa_sin_bolivia_v2.svg`: composición vectorial editable; utiliza la ilustración original como fondo mediante una referencia relativa.

Parámetros de descarga: `where=1=1`, `outFields=DEPARTAMEN,cod`, `returnGeometry=true`, `outSR=4326`, `f=geojson`, `maxAllowableOffset=0.001` y `geometryPrecision=5`. La capa se descargó el 18 de octubre de 2026.

Para regenerar el SVG desde la raíz del repositorio:

```powershell
.\.venv\Scripts\python.exe app/assets/cartografia/crear_banner.py
```

Después se puede abrir el SVG en Chromium y exportar su elemento completo a WebP, con calidad 85. La imagen final es una visualización temática; no es un plano técnico del trazado real del SIN.
