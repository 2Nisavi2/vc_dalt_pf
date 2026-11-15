"""
capturas.py
-----------
Importa al repositorio los pantallazos del entrenamiento en Azure Custom Vision (o de la app),
los ordena, los renombra (Paso_01.png, Paso_02.png, ...), los reduce para que el repo no pese
demasiado y escribe un índice (capturas.json) con título y descripción de cada uno.
La app de Streamlit y el notebook leen ese índice.

Títulos y descripciones: archivo descripciones.csv (columnas archivo,titulo,descripcion) dentro de la
carpeta de origen o indicado con --descripciones. Con --plantilla se genera ese CSV para completarlo.
"""
import csv
import re
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image

from . import config

EXT_CAPTURA = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")

# Orden sugerido de pantallazos del entrenamiento en Custom Vision
PASOS_SUGERIDOS = [
    ("Implementación del recurso en Azure", "Portal de Azure: recurso creado en la suscripción y grupo de recursos."),
    ("Creación del recurso de Custom Vision", "Recurso de Custom Vision: nombre, región y nivel de precios."),
    ("Creación del proyecto", "Proyecto de clasificación multiclase y dominio elegido (compact si se exporta)."),
    ("Carga de imágenes por etiqueta", "Carga de la carpeta de una etiqueta desde data/dataset/train/."),
    ("Imágenes cargadas y etiquetadas", "Las 10 etiquetas con su número de imágenes."),
    ("Selección del tipo de entrenamiento", "Quick Training o Advanced Training."),
    ("Resultados de la iteración 1", "Precision, recall y AP por etiqueta."),
    ("Resultados de la iteración 2", "Mejora frente a la iteración 1."),
    ("Publicación de la iteración", "Publish: nombre de la iteración publicada."),
    ("Datos del endpoint", "Prediction URL del servicio publicado."),
    ("Exportación del modelo", "Export > ONNX."),
    ("Prueba rápida", "Quick Test con una imagen nueva."),
]
PASOS_APP = [
    ("Pantalla inicial y perfil de visión", "Selección del tipo de daltonismo, severidad e intensidad."),
    ("Foto con la cámara del celular", "Señal reconocida con su categoría y color."),
    ("Imagen corregida y simulación", "Corrección de color y vista simulada."),
    ("Clasificación de un lote", "Carga de un ZIP con imágenes y métricas."),
    ("Desempeño del modelo", "Métricas y gráficos de evaluación."),
]


def _clave_natural(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def _titulo_desde_nombre(nombre: str) -> str:
    base = Path(nombre).stem
    base = re.sub(r"^(paso|app|captura|screenshot)?[\s_\-]*\d*[\s_\-\.]*", "", base, flags=re.IGNORECASE)
    base = re.sub(r"[_\-]+", " ", base).strip()
    return base[:1].upper() + base[1:] if base else ""


def listar(origen: Path):
    return sorted((p for p in Path(origen).iterdir() if p.is_file() and p.suffix.lower() in EXT_CAPTURA),
                  key=_clave_natural)


def _leer_descripciones(ruta):
    if ruta is None or not Path(ruta).exists():
        return {}
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        return {fila["archivo"].strip(): fila for fila in csv.DictReader(f) if fila.get("archivo")}


def plantilla(origen, prefijo="Paso"):
    """Escribe <origen>/descripciones.csv con los archivos encontrados y títulos sugeridos."""
    origen = Path(origen)
    archivos = listar(origen)
    if not archivos:
        raise SystemExit(f"No hay imágenes en {origen}")
    sugeridos = PASOS_APP if prefijo.lower() == "app" else PASOS_SUGERIDOS
    ruta = origen / "descripciones.csv"
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["archivo", "titulo", "descripcion"])
        w.writeheader()
        for i, p in enumerate(archivos):
            titulo, desc = sugeridos[i] if i < len(sugeridos) else (_titulo_desde_nombre(p.name), "")
            w.writerow({"archivo": p.name, "titulo": titulo, "descripcion": desc})
    print(f"Plantilla escrita en {ruta} ({len(archivos)} capturas). Edite títulos/descripciones y vuelva a importar.")
    return ruta


def importar(origen, destino=None, prefijo="Paso", descripciones=None, ancho_max=1600):
    origen = Path(origen)
    destino = Path(destino) if destino else (config.CARPETA_CAPTURAS_APP if prefijo.lower() == "app"
                                             else config.CARPETA_CAPTURAS)
    archivos = listar(origen)
    if not archivos:
        raise SystemExit(f"No hay imágenes ({', '.join(EXT_CAPTURA)}) en {origen}")
    textos = _leer_descripciones(descripciones or origen / "descripciones.csv")

    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)

    indice = []
    for i, p in enumerate(archivos, 1):
        nombre = f"{prefijo}_{i:02d}.png"
        with Image.open(p) as img:
            img = img.convert("RGB")
            if img.width > ancho_max:
                img = img.resize((ancho_max, round(img.height * ancho_max / img.width)), Image.LANCZOS)
            img.save(destino / nombre, optimize=True)
        fila = textos.get(p.name, {})
        titulo = (fila.get("titulo") or "").strip() or _titulo_desde_nombre(p.name) or f"{prefijo} {i}"
        indice.append({"archivo": nombre, "titulo": f"{prefijo.replace('_', ' ')} {i}: {titulo}",
                       "descripcion": (fila.get("descripcion") or "").strip(), "original": p.name})

    config.escribir_json(destino / config.INDICE_CAPTURAS,
                         {"generado": datetime.now().isoformat(timespec="seconds"), "capturas": indice})
    sin_texto = sum(1 for c in indice if not c["descripcion"])
    print(f"{len(indice)} capturas importadas en {destino}")
    for c in indice:
        print(f"  {c['archivo']}  <- {c['original']}  · {c['titulo']}")
    if sin_texto:
        print(f"  [aviso] {sin_texto} capturas sin descripción. Use --plantilla para generar descripciones.csv.")
    return indice


def leer_indice(carpeta):
    datos = config.leer_json(Path(carpeta) / config.INDICE_CAPTURAS, {}) or {}
    return [c for c in datos.get("capturas", []) if (Path(carpeta) / c["archivo"]).exists()]
