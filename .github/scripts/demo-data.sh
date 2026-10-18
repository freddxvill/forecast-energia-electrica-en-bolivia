#!/usr/bin/env bash
# Artefactos vivos del pronóstico en la rama huérfana `demo-data`: siempre tiene un único commit
# (se reemplaza con force-push), así `main` no acumula megas de parquet por día.
#
#   demo-data.sh restaurar          trae artifacts/{demo,modelos,historico} de la rama, si existe
#   demo-data.sh publicar "<msg>"   publica esos artefactos como único commit de la rama
set -euo pipefail

RAMA=demo-data
RUTAS=(artifacts/demo artifacts/modelos artifacts/historico_pronosticos.parquet)

case "${1:-}" in
  restaurar)
    if git fetch --depth 1 origin "$RAMA" 2>/dev/null; then
      git checkout FETCH_HEAD -- artifacts
      git reset -q  # deja los archivos en el workspace, fuera del índice de main
      echo "Restaurado desde $RAMA"
    else
      echo "La rama $RAMA no existe todavía: se usan los artefactos de main"
    fi
    ;;
  publicar)
    git config user.name "github-actions[bot]"
    git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
    git checkout -q --orphan "$RAMA-tmp"
    git rm -rq --cached .
    for ruta in "${RUTAS[@]}"; do
      if [ -e "$ruta" ]; then git add -f "$ruta"; fi
    done
    git commit -qm "${2:-Artefactos del pronóstico}"
    git push -qf origin "HEAD:$RAMA"
    echo "Publicado en $RAMA"
    ;;
  *)
    echo "uso: $0 restaurar | publicar <mensaje>" >&2
    exit 2
    ;;
esac
