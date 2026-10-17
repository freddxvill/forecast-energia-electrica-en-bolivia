"""Estilo de gráficos estáticos (matplotlib) y guardado de figuras para el README.

La paleta categórica se validó para daltonismo (CVD) en el orden de abajo, con los
primeros 5 slots en pares adyacentes y los primeros 3 en todos los pares. Por eso se
asigna siempre en este orden y nunca se cicla.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from forecast_energia import config

FIG_DIR = config.ROOT / "reports" / "figures"

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_SUAVE = "#898781"
GRILLA = "#e1e0d9"
EJE = "#c3c2b7"
GRIS_CONTEXTO = "#c3c2b7"

CATEGORICA = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
# Colores fijos por entidad: no cambian aunque se filtren series.
COLOR_TECNOLOGIA = {
    "Hidro": CATEGORICA[0],
    "Térmica": CATEGORICA[1],
    "Eólica": CATEGORICA[2],
    "Solar": CATEGORICA[3],
    "Otras": CATEGORICA[4],
}
COLOR_TIPO_DIA = {
    "Laborable": CATEGORICA[0],
    "Sábado": CATEGORICA[1],
    "Domingo/feriado": CATEGORICA[2],
}
COLOR_REAL = TINTA
COLOR_PREV = CATEGORICA[1]
COLOR_MODELO = CATEGORICA[0]
# Rampa secuencial azul (claro → oscuro) para heatmaps.
SECUENCIAL = mpl.colors.LinearSegmentedColormap.from_list(
    "azul", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def estilo() -> None:
    """Aplica el estilo: marcas finas, grilla recesiva sólida y una sola tipografía sans."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SUPERFICIE,
            "axes.facecolor": SUPERFICIE,
            "savefig.facecolor": SUPERFICIE,
            "figure.dpi": 110,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 10,
            "text.color": TINTA,
            "axes.labelcolor": TINTA_2,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.edgecolor": EJE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRILLA,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": TINTA_SUAVE,
            "ytick.color": TINTA_SUAVE,
            "xtick.labelcolor": TINTA_2,
            "ytick.labelcolor": TINTA_2,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "axes.prop_cycle": mpl.cycler(color=CATEGORICA),
        }
    )


def guardar(fig: plt.Figure, nombre: str) -> Path:
    """Guarda la figura en `reports/figures/<nombre>.png` (para el README)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{nombre}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def etiqueta_final(ax: plt.Axes, x, y, texto: str, color: str = TINTA_2) -> None:
    """Etiqueta directa al final de una línea (texto en tinta, no en el color de la serie)."""
    ax.annotate(
        texto,
        xy=(x, y),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=color,
    )
