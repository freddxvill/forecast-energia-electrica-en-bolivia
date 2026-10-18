"""Traza el banner con la geometría del OEP, sin regenerar los límites con IA.

Uso: python app/assets/cartografia/crear_banner.py
El SVG usa como fondo la ilustración original y se exporta a WebP desde el navegador.
"""

from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path

CARPETA = Path(__file__).resolve().parent
RECURSOS = CARPETA.parent
ANCHO, ALTO = 2172, 724
IZQUIERDA_MAPA = 110

NODOS = {
    1: ("Chuquisaca", -64.3, -20.0, "#326f79", 23, -8, "start"),
    2: ("La Paz", -68.05, -15.3, "#365f7f", 0, -23, "middle"),
    3: ("Cochabamba", -65.6, -17.25, "#276b83", 0, -24, "middle"),
    4: ("Oruro", -67.5, -18.5, "#47788b", -7, 34, "middle"),
    5: ("Potosí", -66.75, -20.5, "#285b76", 0, 34, "middle"),
    6: ("Tarija", -64.1, -21.55, "#33796f", 23, 7, "start"),
    7: ("Santa Cruz", -61.9, -17.7, "#266665", 23, 7, "start"),
    8: ("Beni", -65.4, -14.2, "#2d736a", 23, 7, "start"),
}
ENLACES = [(2, 4), (2, 8), (4, 5), (4, 3), (3, 7), (8, 7), (3, 1), (5, 1), (1, 6)]


def poligonos(geometria: dict) -> list:
    if geometria["type"] == "Polygon":
        return [geometria["coordinates"]]
    if geometria["type"] == "MultiPolygon":
        return geometria["coordinates"]
    raise ValueError(f"Geometría inesperada: {geometria['type']}")


def contiene_anillo(anillo: list, lon: float, lat: float) -> bool:
    dentro = False
    for (x1, y1), (x2, y2) in zip(anillo, anillo[1:] + anillo[:1], strict=True):
        if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
            dentro = not dentro
    return dentro


def contiene(geometria: dict, lon: float, lat: float) -> bool:
    return any(
        contiene_anillo(poligono[0], lon, lat)
        and not any(contiene_anillo(hueco, lon, lat) for hueco in poligono[1:])
        for poligono in poligonos(geometria)
    )


def main() -> None:
    datos = json.loads((CARPETA / "departamentos_oep.geojson").read_text(encoding="utf-8"))
    elementos = datos["features"]
    departamentos = {f["properties"]["cod"]: f for f in elementos if f["properties"]["cod"]}
    assert set(departamentos) == set(range(1, 10))
    for codigo, (nombre, lon, lat, *_) in NODOS.items():
        assert contiene(departamentos[codigo]["geometry"], lon, lat), nombre
    puntos = [
        punto
        for elemento in elementos
        for poligono in poligonos(elemento["geometry"])
        for anillo in poligono
        for punto in anillo
    ]
    min_lon = min(p[0] for p in puntos)
    min_lat, max_lat = min(p[1] for p in puntos), max(p[1] for p in puntos)
    # Proyección equirectangular: misma escala para todos los departamentos.
    cos_lat = math.cos(math.radians((min_lat + max_lat) / 2))
    escala = 640 / (max_lat - min_lat)

    def proyectar(lon: float, lat: float) -> tuple[float, float]:
        return IZQUIERDA_MAPA + (lon - min_lon) * cos_lat * escala, 42 + (max_lat - lat) * escala

    def trazado(geometria: dict) -> str:
        partes = []
        for poligono in poligonos(geometria):
            for anillo in poligono:
                xy = [proyectar(*punto) for punto in anillo]
                partes.append("M" + "L".join(f"{x:.2f},{y:.2f}" for x, y in xy) + "Z")
        return "".join(partes)

    paths = {id(f): trazado(f["geometry"]) for f in elementos}
    recorte = "".join(
        f'<path d="{paths[id(f)]}" fill-rule="evenodd" clip-rule="evenodd"/>'
        for codigo, f in departamentos.items()
        if codigo != 9
    )
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{ANCHO}" height="{ALTO}" '
        f'viewBox="0 0 {ANCHO} {ALTO}" role="img">',
        "<title>Red eléctrica de Bolivia: ocho departamentos conectados</title>",
        "<desc>Límites departamentales trazados a partir de la cartografía del OEP. "
        "Pando permanece sin conexión. Las líneas eléctricas son esquemáticas.</desc>",
        "<defs>",
        '<filter id="sombra" x="-25%" y="-25%" width="150%" height="150%">'
        '<feDropShadow dx="0" dy="7" stdDeviation="12" flood-color="#051729" '
        'flood-opacity="0.8"/></filter>',
        '<filter id="luz" x="-200%" y="-200%" width="500%" height="500%">'
        '<feGaussianBlur stdDeviation="4"/></filter>',
        '<linearGradient id="superficie" x1="0" y1="0" x2="0.8" y2="1">'
        '<stop stop-color="#c6eef0" stop-opacity="0.15"/>'
        '<stop offset="0.55" stop-color="#071b36" stop-opacity="0.03"/>'
        '<stop offset="1" stop-color="#071b36" stop-opacity="0.22"/>'
        "</linearGradient>",
        f'<clipPath id="red-activa">{recorte}</clipPath>',
        "</defs>",
        f'<image href="footer_red_electrica_bolivia.webp" width="{ANCHO}" height="{ALTO}"/>',
        '<g filter="url(#sombra)" stroke-linejoin="round">',
    ]
    for f in elementos:
        codigo = f["properties"]["cod"]
        nombre = NODOS[codigo][0] if codigo in NODOS else f["properties"]["DEPARTAMEN"]
        color = NODOS[codigo][3] if codigo in NODOS else "#586578" if codigo == 9 else "#47738d"
        svg.append(
            f'<path data-codigo="{codigo}" data-departamento="{escape(nombre)}" '
            f'd="{paths[id(f)]}" fill="{color}" fill-rule="evenodd" '
            'stroke="#b8e9ec" stroke-opacity="0.92" stroke-width="1.45"/>'
        )
        svg.append(f'<path d="{paths[id(f)]}" fill="url(#superficie)" fill-rule="evenodd"/>')
    svg.append("</g>")
    xy_nodos = {codigo: proyectar(nodo[1], nodo[2]) for codigo, nodo in NODOS.items()}
    svg.append('<g clip-path="url(#red-activa)">')
    for origen, destino in ENLACES:
        x1, y1 = xy_nodos[origen]
        x2, y2 = xy_nodos[destino]
        svg.append(
            f'<path data-enlace="{origen}-{destino}" d="M{x1:.2f},{y1:.2f} '
            f'L{x2:.2f},{y2:.2f}" fill="none" stroke="#ffc66a" '
            'stroke-opacity="0.45" stroke-width="7" filter="url(#luz)"/>'
        )
        svg.append(
            f'<path d="M{x1:.2f},{y1:.2f}L{x2:.2f},{y2:.2f}" fill="none" '
            'stroke="#ffdc91" stroke-width="2.3" stroke-linecap="round"/>'
        )
    svg.append("</g>")
    for codigo, (nombre, _, _, _, dx, dy, ancla) in NODOS.items():
        x, y = xy_nodos[codigo]
        svg.extend(
            [
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="12" fill="#ffb344" '
                'opacity="0.8" filter="url(#luz)"/>',
                f'<circle data-nodo="{codigo}" cx="{x:.2f}" cy="{y:.2f}" r="6.7" '
                'fill="#ffe1a0" stroke="#fff5ce" stroke-width="2"/>',
                f'<text x="{x + dx:.2f}" y="{y + dy:.2f}" text-anchor="{ancla}" '
                'font-family="Segoe UI,Arial,sans-serif" font-size="25" font-weight="600" '
                'fill="#f7f4e9" stroke="#173349" stroke-width="4.5" '
                f'paint-order="stroke" stroke-linejoin="round">{nombre}</text>',
            ]
        )
    x_pando, y_pando = proyectar(-67.6, -11.3)
    svg.extend(
        [
            f'<text x="{x_pando:.2f}" y="{y_pando:.2f}" text-anchor="middle" '
            'font-family="Segoe UI,Arial,sans-serif" font-size="24" font-weight="600" '
            'fill="#d1dbe3">Pando</text>',
            f'<text x="{x_pando:.2f}" y="{y_pando + 23:.2f}" text-anchor="middle" '
            'font-family="Segoe UI,Arial,sans-serif" font-size="16" '
            'fill="#c1ccd7">Sin conexión</text>',
            "</svg>",
        ]
    )
    salida = RECURSOS / "banner_mapa_sin_bolivia_v2.svg"
    salida.write_text("\n".join(svg), encoding="utf-8")
    print(f"SVG: {salida}")
    print("9 departamentos con su geometría de origen; 8 nodos interiores; Pando sin nodo.")


if __name__ == "__main__":
    main()
