#!/bin/sh
# Despliegue de VisionVial en Railway con la CLI (https://docs.railway.com/cli).
# Requisitos: CLI instalada (npm i -g @railway/cli) y .streamlit/secrets.toml con el endpoint y la clave.
# Uso: sh scripts/railway_desplegar.sh
set -e
cd "$(dirname "$0")/.."

leer() {  # lee una clave de .streamlit/secrets.toml
  python3 -c "import tomllib,sys; print(tomllib.load(open('.streamlit/secrets.toml','rb')).get(sys.argv[1],''))" "$1"
}

railway whoami >/dev/null 2>&1 || railway login

# Vincula la carpeta a un proyecto (crea uno nuevo si no hay ninguno vinculado)
railway status >/dev/null 2>&1 || railway init --name visionvial

ENDPOINT=$(leer CV_PREDICTION_ENDPOINT)
CLAVE=$(leer CV_PREDICTION_KEY)
if [ -n "$ENDPOINT" ] && [ -n "$CLAVE" ]; then
  railway variables --set "CV_PREDICTION_ENDPOINT=$ENDPOINT" --set "CV_PREDICTION_KEY=$CLAVE"
  echo "Variables del endpoint configuradas."
else
  echo "[aviso] No hay endpoint/clave en .streamlit/secrets.toml: la app usará el modelo ONNX del repo."
fi

railway up --detach      # construye con el Dockerfile y despliega
railway domain           # genera (o muestra) el dominio público https://<app>.up.railway.app
