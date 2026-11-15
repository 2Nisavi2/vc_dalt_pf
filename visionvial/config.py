"""
config.py
---------
Fuente única de rutas, clases y credenciales. La usan la CLI (vv.py), la app (app.py) y el notebook.
"""
import json
import os
import tomllib
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------ rutas del repositorio
CARPETA_DATA = RAIZ / "data"                          # no se versiona
CARPETA_GTSRB = CARPETA_DATA / "gtsrb"                # descarga original
CARPETA_DATASET = CARPETA_DATA / "dataset"            # train/ test/ manifest.csv
CARPETA_MODELOS = RAIZ / "modelos" / "actual"         # modelo registrado (se versiona)
CARPETA_RESULTADOS = RAIZ / "resultados"              # métricas y gráficos (se versiona)
CARPETA_CACHE = CARPETA_RESULTADOS / "cache"          # predicciones por modelo
CARPETA_CAPTURAS = RAIZ / "assets" / "entrenamiento"  # pantallazos de ACV (se versionan)
CARPETA_CAPTURAS_APP = RAIZ / "assets" / "app"        # pantallazos de la app (se versionan)
ARCHIVO_SECRETOS = RAIZ / ".streamlit" / "secrets.toml"

INFO_MODELO = CARPETA_MODELOS / "info.json"
METRICAS = CARPETA_RESULTADOS / "metricas.json"
INDICE_CAPTURAS = "capturas.json"

# ------------------------------------------------------------------ clases del modelo
# etiqueta -> nombre visible, categoría en Colombia, color, significado, clases de GTSRB que la componen
SENALES = {
    "pare": ("PARE", "Reglamentaria", "rojo", "Detenga completamente el vehículo antes de continuar.", [14]),
    "ceda_el_paso": ("Ceda el paso", "Reglamentaria", "rojo", "Ceda el paso a los vehículos de la vía a la que ingresa.", [13]),
    "velocidad_maxima": ("Velocidad máxima", "Reglamentaria", "rojo", "No supere la velocidad indicada en la señal.", [1, 2, 3, 4, 5, 7, 8]),
    "prohibido_adelantar": ("Prohibido adelantar", "Reglamentaria", "rojo", "No adelante a otros vehículos en este tramo.", [9]),
    "no_entre": ("No entre / prohibido el paso", "Reglamentaria", "rojo", "No ingrese a esta vía.", [17]),
    "curva_peligrosa": ("Curva peligrosa", "Preventiva", "amarillo", "Reduzca la velocidad: se aproxima una curva.", [19, 20, 21]),
    "semaforo_adelante": ("Semáforo adelante", "Preventiva", "amarillo", "Prepárese para detenerse: hay un semáforo próximo.", [26]),
    "peatones": ("Paso peatonal", "Preventiva", "amarillo", "Atención a peatones cruzando la vía.", [27]),
    "zona_escolar": ("Zona escolar", "Preventiva", "amarillo", "Reduzca la velocidad: presencia de niños.", [28]),
    "obras": ("Obras en la vía", "Transitoria", "naranja", "Precaución: trabajos en la vía.", [25]),
}
CATEGORIA = {k: (v[1], v[2]) for k, v in SENALES.items()}
MAPEO_GTSRB = {cid: etiqueta for etiqueta, v in SENALES.items() for cid in v[4]}

COLOR_HEX = {"rojo": "#d62828", "amarillo": "#f2c200", "naranja": "#f77f00", "azul": "#1d4ed8", "verde": "#2a9d4b"}
COLOR_CATEGORIA = {"Reglamentaria": "#d62828", "Preventiva": "#e0a800", "Transitoria": "#f77f00"}
PALETA = ["#2a78d6", "#eb6834", "#1baf7a", "#9b5de5", "#e0a800",
          "#d62828", "#00a6b4", "#7a5c3e", "#ef476f", "#4f5d75"]
COLOR_CLASE = dict(zip(sorted(SENALES), PALETA))

EXTENSIONES_IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm")
MAX_BYTES_PREDICCION = 4 * 1024 * 1024


def info_senal(etiqueta):
    """(nombre, categoría, color, significado) de una etiqueta; tolera etiquetas desconocidas."""
    v = SENALES.get(etiqueta)
    if v is None:
        return etiqueta.replace("_", " ").capitalize(), "Sin categoría", "-", ""
    return v[:4]


# ------------------------------------------------------------------ credenciales del endpoint
def leer_secreto(nombre, secretos_streamlit=None):
    """Busca un secreto en st.secrets, en .streamlit/secrets.toml y en variables de entorno."""
    if secretos_streamlit is not None:
        try:
            if nombre in secretos_streamlit:
                return secretos_streamlit[nombre]
        except Exception:  # noqa: BLE001  (Streamlit sin secrets.toml)
            pass
    if ARCHIVO_SECRETOS.exists():
        with open(ARCHIVO_SECRETOS, "rb") as f:
            valor = tomllib.load(f).get(nombre)
        if valor:
            return valor
    return os.environ.get(nombre)


def leer_json(ruta, defecto=None):
    ruta = Path(ruta)
    if not ruta.exists():
        return defecto
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def escribir_json(ruta, datos):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2, default=str)
