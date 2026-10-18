"""Kernel de Kaggle: corre el backtest (pipelines.backtest) con GPU y deja los artefactos en
/kaggle/working para descargarlos con `kaggle kernels output`.

`kaggle/preparar.py` reemplaza los marcadores __MODELOS__ y __UNIQUE_ID__ al armar el kernel.
El código del proyecto y los datos procesados llegan como dataset privado (`proyecto.zip`).
"""

import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

MODELOS = "__MODELOS__".split()  # noqa: SIM905 (marcador reemplazado por preparar.py)
UNIQUE_ID = "__UNIQUE_ID__"

# Mismas versiones que uv.lock, para que los resultados sean comparables con los locales.
PAQUETES = [
    "statsforecast==2.1.1",
    "mlforecast==1.1.0",
    "neuralforecast==3.3.0",
    "utilsforecast==0.2.17",
    "coreforecast==0.0.18",
    "lightgbm==4.7.0",
    "holidays==0.106",
    "tenacity",
]
TRABAJO = Path("/kaggle/working")
PROYECTO = TRABAJO / "proyecto"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def localizar_proyecto() -> Path:
    """Devuelve la raíz del proyecto (la carpeta que contiene `src/forecast_energia`)."""
    entradas = Path("/kaggle/input")
    for cfg in entradas.rglob("forecast_energia/config.py"):
        raiz = cfg.parents[2]
        shutil.copytree(raiz, PROYECTO, dirs_exist_ok=True)  # /kaggle/input es de solo lectura
        return PROYECTO
    for archivo in entradas.rglob("proyecto.zip"):
        with zipfile.ZipFile(archivo) as z:
            z.extractall(PROYECTO)
        return PROYECTO
    raise FileNotFoundError("No se encontró el código del proyecto en /kaggle/input")


def instalar() -> None:
    import torch  # torch de Kaggle (con CUDA): no debe reemplazarse

    base = torch.__version__.split("+")[0]
    restricciones = TRABAJO / "constraints.txt"
    restricciones.write_text(f"torch=={base}\n")
    log(f"torch {torch.__version__}, CUDA disponible: {torch.cuda.is_available()}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-c", str(restricciones), *PAQUETES],
        check=True,
    )


def main() -> None:
    raiz = localizar_proyecto()
    log(f"Proyecto en {raiz}: {sorted(p.name for p in raiz.iterdir())}")
    instalar()
    env = os.environ | {"PYTHONPATH": str(raiz / "src"), "PYTHONIOENCODING": "utf-8"}
    cmd = [sys.executable, "-m", "forecast_energia.pipelines.backtest", "--unique-id", UNIQUE_ID]
    cmd += ["--modelos", *MODELOS]
    log("Ejecutando: " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=raiz, env=env)
    for carpeta in ("artifacts", "reports"):
        if (raiz / carpeta).exists():
            shutil.copytree(raiz / carpeta, TRABAJO / carpeta, dirs_exist_ok=True)
    shutil.rmtree(PROYECTO)  # solo se descargan los resultados
    log("Listo")


if __name__ == "__main__":
    main()
