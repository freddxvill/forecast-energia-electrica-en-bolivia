#!/usr/bin/env bash
# Artefactos vivos en ramas huérfanas de un único commit, que se reemplaza con force-push en cada
# publicación: queda solo la última versión y `main` no acumula megas de modelos ni de parquet.
#
#   modelos          artifacts/modelos/ (LightGBM se reentrena a diario; TFT, cada semana)
#   data-inference   artifacts/demo/ + artifacts/historico_pronosticos.parquet
#                    (los datos predichos: lo que leen la API y la app)
#
#   rama-artefactos.sh restaurar <rama>             trae los archivos de la rama, si existe
#   rama-artefactos.sh publicar <rama> "<mensaje>"  publica sus rutas como único commit de la rama
set -euo pipefail

rutas() {
  case "$1" in
    modelos) echo artifacts/modelos ;;
    data-inference) echo artifacts/demo artifacts/historico_pronosticos.parquet ;;
    *) echo "rama desconocida: $1" >&2; exit 2 ;;
  esac
}

ACCION="${1:-}"
RAMA="${2:-}"
RUTAS=$(rutas "$RAMA")

case "$ACCION" in
  restaurar)
    if git fetch --depth 1 origin "$RAMA" 2>/dev/null; then
      git checkout FETCH_HEAD -- .
      git reset -q  # deja los archivos en el workspace, fuera del índice de main
      echo "Restaurado desde $RAMA"
    else
      echo "La rama $RAMA no existe todavía: se usan los artefactos locales"
    fi
    ;;
  publicar)
    git config user.name "github-actions[bot]"
    git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
    git checkout -q --orphan "$RAMA-tmp"
    git rm -rq --cached .
    for ruta in $RUTAS; do
      if [ -e "$ruta" ]; then git add -f "$ruta"; fi
    done
    if git diff --cached --quiet; then
      echo "Nada que publicar en $RAMA" >&2
      exit 1
    fi
    git commit -qm "${3:-Artefactos del pronóstico}"
    git push -qf origin "HEAD:$RAMA"
    echo "Publicado en $RAMA"
    ;;
  *)
    echo "uso: $0 restaurar <rama> | publicar <rama> <mensaje>" >&2
    exit 2
    ;;
esac
