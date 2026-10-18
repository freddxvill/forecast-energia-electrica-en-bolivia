"""Arma las carpetas para subir a Kaggle: un dataset privado (código + datos) y un kernel con GPU.

Uso:
    uv run python kaggle/preparar.py --usuario <usuario_kaggle> --modelos naive mstl lgbm nhits tft
    kaggle datasets create -p kaggle/_build/dataset        # la primera vez
    kaggle datasets version -p kaggle/_build/dataset -m "actualización"   # las siguientes
    kaggle kernels push -p kaggle/_build/kernel
    kaggle kernels status <usuario>/forecast-energia-backtest
    kaggle kernels output <usuario>/forecast-energia-backtest -p kaggle/_build/salida
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
BUILD = RAIZ / "kaggle" / "_build"
SLUG_DATASET = "forecast-energia-bolivia"
SLUG_KERNEL = "forecast-energia-backtest"


def armar_dataset(usuario: str) -> Path:
    carpeta = BUILD / "dataset"
    shutil.rmtree(carpeta, ignore_errors=True)
    carpeta.mkdir(parents=True)
    archivos = [
        p for p in (RAIZ / "src" / "forecast_energia").rglob("*.py") if "__pycache__" not in p.parts
    ]
    archivos += sorted((RAIZ / "data" / "processed").glob("*"))
    with zipfile.ZipFile(carpeta / "proyecto.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in archivos:
            z.write(p, p.relative_to(RAIZ).as_posix())
    meta = {
        "title": SLUG_DATASET,
        "id": f"{usuario}/{SLUG_DATASET}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (carpeta / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    return carpeta


def armar_kernel(usuario: str, modelos: list[str], conjunto: str) -> Path:
    carpeta = BUILD / "kernel"
    shutil.rmtree(carpeta, ignore_errors=True)
    carpeta.mkdir(parents=True)
    codigo = (RAIZ / "kaggle" / "kernel_backtest.py").read_text(encoding="utf-8")
    codigo = codigo.replace("__MODELOS__", " ".join(modelos)).replace("__CONJUNTO__", conjunto)
    (carpeta / "kernel_backtest.py").write_text(codigo, encoding="utf-8")
    meta = {
        "id": f"{usuario}/{SLUG_KERNEL}",
        "title": SLUG_KERNEL,
        "code_file": "kernel_backtest.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,  # para instalar las librerías de Nixtla con pip
        "dataset_sources": [f"{usuario}/{SLUG_DATASET}"],
    }
    (carpeta / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return carpeta


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--usuario", required=True)
    parser.add_argument("--modelos", nargs="+", default=["naive", "mstl", "lgbm", "nhits", "tft"])
    parser.add_argument("--conjunto", default="SIN", choices=["SIN", "JERARQUIA"])
    args = parser.parse_args()
    ds = armar_dataset(args.usuario)
    k = armar_kernel(args.usuario, args.modelos, args.conjunto)
    tam = (ds / "proyecto.zip").stat().st_size / 1e6
    print(f"Dataset: {ds} (proyecto.zip {tam:.1f} MB)\nKernel:  {k}")


if __name__ == "__main__":
    main()
