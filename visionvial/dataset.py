"""
dataset.py
----------
Descarga GTSRB, selecciona las clases equivalentes a señales de Colombia (config.SENALES) y genera
carpetas listas para cargar MANUALMENTE en Azure Custom Vision:

  data/dataset/train/<etiqueta>/*.jpg   -> se suben a Custom Vision
  data/dataset/test/<etiqueta>/*.jpg    -> prueba independiente (NO se suben)
  data/dataset/manifest.csv             -> inventario de todas las imágenes
"""
import csv
import random
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

from . import config

BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
ZIPS = ["GTSRB-Training_fixed.zip", "GTSRB_Final_Test_Images.zip", "GTSRB_Final_Test_GT.zip"]


def descargar(carpeta_gtsrb: Path) -> None:
    carpeta_gtsrb.mkdir(parents=True, exist_ok=True)
    for nombre in ZIPS:
        destino = carpeta_gtsrb / nombre
        if not destino.exists():
            print(f"Descargando {nombre} ...")
            parcial = destino.with_suffix(".part")
            try:
                urllib.request.urlretrieve(BASE_URL + nombre, parcial)
            except Exception as e:  # noqa: BLE001
                raise SystemExit(
                    f"No se pudo descargar {nombre}: {e}\n"
                    "Descargue GTSRB manualmente (oficial o Kaggle), descomprímalo y use --gtsrb-dir."
                ) from e
            parcial.rename(destino)
        marca = carpeta_gtsrb / f".{nombre}.ok"
        if not marca.exists():
            print(f"Descomprimiendo {nombre} ...")
            with zipfile.ZipFile(destino) as z:
                z.extractall(carpeta_gtsrb)
            marca.touch()


def _leer_csv(ruta: Path):
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        muestra = f.read(2048)
        f.seek(0)
        delim = ";" if muestra.count(";") > muestra.count(",") else ","
        return list(csv.DictReader(f, delimiter=delim))


def _track(stem: str) -> str:
    # 00000_00029 -> 00000 ; 00014_00000_00029 -> 00014_00000 ; 00123 -> 00123
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def indexar(carpeta_gtsrb: Path):
    """Funciona con el GTSRB oficial y con la versión de Kaggle."""
    items, vistos = [], set()
    etiquetas_test = {}
    for c in carpeta_gtsrb.rglob("*.csv"):
        if "test" not in c.name.lower():
            continue
        filas = _leer_csv(c)
        if not filas or "ClassId" not in filas[0]:
            continue
        col = "Filename" if "Filename" in filas[0] else "Path"
        for fila in filas:
            etiquetas_test[Path(fila[col]).name] = int(fila["ClassId"])

    for p in carpeta_gtsrb.rglob("*"):
        if p.suffix.lower() not in config.EXTENSIONES_IMG:
            continue
        partes = [x.lower() for x in p.parts]
        if any(x in ("train", "training") for x in partes) and p.parent.name.isdigit():
            split, cid = "train", int(p.parent.name)
        elif any("test" in x for x in partes) and p.name in etiquetas_test:
            split, cid = "test", etiquetas_test[p.name]
        else:
            continue
        if cid not in config.MAPEO_GTSRB:
            continue
        clave = (split, cid, p.name)  # Kaggle trae Train/ y train/ duplicados
        if clave in vistos:
            continue
        vistos.add(clave)
        items.append({"ruta": p, "class_id": cid, "split": split, "track": f"{cid}_{_track(p.stem)}"})
    return items


def _tamano(p: Path):
    with Image.open(p) as im:
        return im.size


def seleccionar(items, split, n_por_etiqueta, min_lado, rng):
    por_etiqueta = defaultdict(list)
    for it in items:
        if it["split"] != split:
            continue
        w, h = _tamano(it["ruta"])
        if min(w, h) < min_lado:
            continue
        it["ancho"], it["alto"] = w, h
        por_etiqueta[config.MAPEO_GTSRB[it["class_id"]]].append(it)

    seleccion = {}
    for etiqueta, lista in por_etiqueta.items():
        if split == "test":
            rng.shuffle(lista)
            seleccion[etiqueta] = lista[:n_por_etiqueta]
            continue
        # ~30 tomas casi idénticas por señal física: round-robin entre tracks, priorizando las más grandes
        tracks = defaultdict(list)
        for it in lista:
            tracks[it["track"]].append(it)
        colas = [sorted(v, key=lambda x: x["ancho"] * x["alto"], reverse=True) for v in tracks.values()]
        rng.shuffle(colas)
        elegidas = []
        while len(elegidas) < n_por_etiqueta and any(colas):
            for cola in colas:
                if cola and len(elegidas) < n_por_etiqueta:
                    elegidas.append(cola.pop(0))
        seleccion[etiqueta] = elegidas
    return seleccion


def guardar_imagen(origen: Path, destino: Path, lado_min_salida: int) -> None:
    with Image.open(origen) as im:
        im = im.convert("RGB")
        w, h = im.size
        if min(w, h) < lado_min_salida:  # Custom Vision recomienda >= 256 px en el lado corto
            f = lado_min_salida / min(w, h)
            im = im.resize((round(w * f), round(h * f)), Image.LANCZOS)
        destino.parent.mkdir(parents=True, exist_ok=True)
        im.save(destino, "JPEG", quality=95)


def generar(salida=config.CARPETA_DATASET, gtsrb_dir=None, extra=None, n_train=150, n_test=50,
            min_lado=32, lado_salida=256, seed=42):
    rng = random.Random(seed)
    salida = Path(salida)
    raiz_gtsrb = Path(gtsrb_dir) if gtsrb_dir else config.CARPETA_GTSRB
    if not gtsrb_dir:
        descargar(raiz_gtsrb)

    print("Indexando imágenes ...")
    items = indexar(raiz_gtsrb)
    if not items:
        raise SystemExit(f"No se encontraron imágenes de las clases seleccionadas en {raiz_gtsrb}")

    for split in ("train", "test"):
        if (salida / split).exists():
            shutil.rmtree(salida / split)

    filas = []
    for split, n in (("train", n_train), ("test", n_test)):
        for etiqueta, lista in seleccionar(items, split, n, min_lado, rng).items():
            for i, it in enumerate(lista):
                dest = salida / split / etiqueta / f"{etiqueta}_{i:04d}.jpg"
                guardar_imagen(it["ruta"], dest, lado_salida)
                cat, color = config.CATEGORIA[etiqueta]
                filas.append({
                    "split": split, "etiqueta": etiqueta, "categoria": cat, "color": color,
                    "fuente": "GTSRB", "class_id_gtsrb": it["class_id"], "track": it["track"],
                    "ancho_original": it["ancho"], "alto_original": it["alto"],
                    "archivo_original": str(it["ruta"]), "archivo": dest.relative_to(salida).as_posix(),
                })

    if extra:  # fotos propias: <extra>/<etiqueta>/*.jpg -> 70 % train / 30 % test
        for carpeta in sorted(Path(extra).iterdir()):
            if not carpeta.is_dir():
                continue
            if carpeta.name not in config.SENALES:
                print(f"  [aviso] '{carpeta.name}' no es una etiqueta válida; se omite.")
                continue
            fotos = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in config.EXTENSIONES_IMG)
            rng.shuffle(fotos)
            corte = round(len(fotos) * 0.7)
            for j, p in enumerate(fotos):
                split = "train" if j < corte else "test"
                dest = salida / split / carpeta.name / f"{carpeta.name}_propia_{j:04d}.jpg"
                w, h = _tamano(p)
                guardar_imagen(p, dest, lado_salida)
                cat, color = config.CATEGORIA[carpeta.name]
                filas.append({
                    "split": split, "etiqueta": carpeta.name, "categoria": cat, "color": color,
                    "fuente": "propia", "class_id_gtsrb": "", "track": "",
                    "ancho_original": w, "alto_original": h,
                    "archivo_original": str(p), "archivo": dest.relative_to(salida).as_posix(),
                })

    with open(salida / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    conteo = defaultdict(lambda: {"train": 0, "test": 0})
    for r in filas:
        conteo[r["etiqueta"]][r["split"]] += 1
    print(f"\n{'etiqueta':22s} {'categoria':14s} {'train':>6s} {'test':>6s}")
    for et in sorted(conteo):
        print(f"{et:22s} {config.CATEGORIA[et][0]:14s} {conteo[et]['train']:6d} {conteo[et]['test']:6d}")
    print(f"{'TOTAL':37s} {sum(v['train'] for v in conteo.values()):6d} {sum(v['test'] for v in conteo.values()):6d}")
    print(f"\nListo. Suba a Custom Vision solo: {salida / 'train'}  (una carpeta = una etiqueta)")
    return conteo
