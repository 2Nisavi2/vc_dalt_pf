"""
modelo.py
---------
Registro y uso del modelo entrenado en Azure Custom Vision. Hay dos formas de usarlo:

* ENDPOINT: la iteración publicada en Azure (Prediction API). Es la que exige la actividad para la app.
* ONNX:     el modelo exportado desde Custom Vision (Export > ONNX, requiere dominio "compact").
            Corre local, sin costo por predicción; se usa para evaluar y como respaldo de la app.

Ambos devuelven lo mismo: un DataFrame con columnas `etiqueta` y `probabilidad`, ordenado de mayor a menor.
"""
import hashlib
import io
import shutil
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageOps

from . import config


# ====================================================================== utilidades de imagen
def a_pil(origen):
    """Ruta, bytes, PIL o numpy -> PIL RGB con la orientación EXIF corregida."""
    if isinstance(origen, Image.Image):
        img = origen
    elif isinstance(origen, np.ndarray):
        img = Image.fromarray(origen)
    elif isinstance(origen, (bytes, bytearray)):
        img = Image.open(io.BytesIO(origen))
    else:
        img = Image.open(origen)
    return ImageOps.exif_transpose(img).convert("RGB")


def a_jpeg(img: Image.Image, calidad=95) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=calidad)
    datos = buf.getvalue()
    if len(datos) > config.MAX_BYTES_PREDICCION:
        img = img.copy()
        img.thumbnail((1600, 1600))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        datos = buf.getvalue()
    return datos


def _ordenar(probs: dict) -> pd.DataFrame:
    tabla = pd.DataFrame({"etiqueta": list(probs), "probabilidad": [float(v) for v in probs.values()]})
    return tabla.sort_values("probabilidad", ascending=False).reset_index(drop=True)


# ====================================================================== clasificadores
class ClasificadorEndpoint:
    tipo = "endpoint"

    def __init__(self, url, clave, pausa=0.0):
        url = url.rstrip("/")
        for sufijo in ("/image", "/url"):
            url = url.removesuffix(sufijo)
        self.url_base = url
        self.url_imagen = f"{url}/image"
        self.clave = clave
        self.pausa = pausa
        self.iteracion = url.rsplit("/", 1)[-1]
        self.id = f"endpoint_{self.iteracion}"
        self.descripcion = f"Azure Custom Vision · endpoint (iteración {self.iteracion})"

    def _post(self, **kwargs):
        headers = {"Prediction-Key": self.clave, **kwargs.pop("headers", {})}
        url = kwargs.pop("url", self.url_imagen)
        for intento in range(6):
            r = requests.post(url, headers=headers, timeout=60, **kwargs)
            if r.status_code != 429:  # demasiadas solicitudes por segundo: esperar y reintentar
                break
            time.sleep(2 ** intento)
        if not r.ok:
            try:
                detalle = r.json().get("message", r.text)
            except ValueError:
                detalle = r.text
            raise RuntimeError(f"El servicio respondió {r.status_code}: {detalle}")
        if self.pausa:
            time.sleep(self.pausa)
        return r.json()

    def predecir(self, origen) -> pd.DataFrame:
        respuesta = self._post(data=a_jpeg(a_pil(origen)), headers={"Content-Type": "application/octet-stream"})
        return _ordenar({p["tagName"]: p["probability"] for p in respuesta["predictions"]})

    def predecir_url(self, url_imagen) -> pd.DataFrame:
        respuesta = self._post(url=f"{self.url_base}/url", json={"Url": url_imagen.strip()})
        return _ordenar({p["tagName"]: p["probability"] for p in respuesta["predictions"]})


class ClasificadorONNX:
    """Modelo exportado por Custom Vision en formato ONNX.

    Preprocesamiento igual al de los ejemplos oficiales de exportación de Custom Vision:
    recorte cuadrado al centro -> 256 px -> recorte central al tamaño de entrada -> orden BGR, valores 0-255.
    """
    tipo = "onnx"

    def __init__(self, carpeta=config.CARPETA_MODELOS):
        import onnxruntime as ort  # import diferido: solo se necesita si se usa ONNX

        carpeta = Path(carpeta)
        info = config.leer_json(carpeta / "info.json", {})
        self.ruta = carpeta / info.get("archivo", "model.onnx")
        self.etiquetas = [l.strip() for l in (carpeta / "labels.txt").read_text(encoding="utf-8").splitlines()
                          if l.strip()]
        self.sesion = ort.InferenceSession(str(self.ruta), providers=["CPUExecutionProvider"])
        entrada = self.sesion.get_inputs()[0]
        self.nombre_entrada = entrada.name
        forma = list(entrada.shape)
        self.nchw = len(forma) == 4 and forma[1] == 3
        lado = forma[2] if self.nchw else (forma[1] if len(forma) == 4 else 224)
        self.lado = int(lado) if isinstance(lado, (int, np.integer)) and lado > 0 else 224
        self.id = f"onnx_{info.get('id', hashlib.sha256(self.ruta.read_bytes()).hexdigest()[:12])}"
        self.iteracion = info.get("iteracion") or "exportada"
        self.descripcion = f"Azure Custom Vision · ONNX local (iteración {self.iteracion})"

    def _preprocesar(self, img: Image.Image) -> np.ndarray:
        img = img.copy()
        if max(img.size) > 1600:
            img.thumbnail((1600, 1600))
        w, h = img.size
        m = min(w, h)
        img = img.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m))
        img = img.resize((256, 256), Image.BILINEAR)
        if self.lado <= 256:
            d = (256 - self.lado) // 2
            img = img.crop((d, d, d + self.lado, d + self.lado))
        else:
            img = img.resize((self.lado, self.lado), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)[:, :, ::-1]  # RGB -> BGR
        if self.nchw:
            arr = arr.transpose(2, 0, 1)
        return np.ascontiguousarray(arr[np.newaxis])

    def predecir(self, origen) -> pd.DataFrame:
        salidas = self.sesion.run(None, {self.nombre_entrada: self._preprocesar(a_pil(origen))})
        for s in salidas:  # exportaciones antiguas: lista de diccionarios {etiqueta: prob}
            if isinstance(s, list) and s and isinstance(s[0], dict):
                return _ordenar(s[0])
        for s in salidas:  # exportaciones recientes: vector de probabilidades en el orden de labels.txt
            v = np.asarray(s, dtype=np.float64).ravel()
            if v.size == len(self.etiquetas):
                if v.min() < 0 or abs(v.sum() - 1) > 1e-3:
                    v = np.exp(v - v.max())
                    v = v / v.sum()
                return _ordenar(dict(zip(self.etiquetas, v)))
        raise RuntimeError("No se reconoció la salida del modelo ONNX "
                           f"({[getattr(s, 'shape', type(s)) for s in salidas]}) con {len(self.etiquetas)} etiquetas")


def credenciales_endpoint(secretos_streamlit=None):
    info = config.leer_json(config.INFO_MODELO, {})
    url = config.leer_secreto("CV_PREDICTION_ENDPOINT", secretos_streamlit) or info.get("endpoint")
    clave = config.leer_secreto("CV_PREDICTION_KEY", secretos_streamlit)
    return url, clave


def cargar_clasificador(preferencia="auto", secretos_streamlit=None, pausa=0.0):
    """preferencia: 'endpoint', 'onnx' o 'auto' (endpoint si hay credenciales; si no, ONNX)."""
    url, clave = credenciales_endpoint(secretos_streamlit)
    hay_endpoint = bool(url and clave)
    hay_onnx = (config.CARPETA_MODELOS / "labels.txt").exists() and any(config.CARPETA_MODELOS.glob("*.onnx"))
    if preferencia == "endpoint" or (preferencia == "auto" and hay_endpoint):
        if not hay_endpoint:
            raise RuntimeError("Faltan CV_PREDICTION_ENDPOINT y CV_PREDICTION_KEY (.streamlit/secrets.toml).")
        return ClasificadorEndpoint(url, clave, pausa=pausa)
    if hay_onnx:
        return ClasificadorONNX()
    raise RuntimeError("No hay modelo disponible: registre el ONNX con `python vv.py modelo --archivo export.zip` "
                       "o configure el endpoint en .streamlit/secrets.toml.")


# ====================================================================== registro del modelo
def _validar_etiquetas(etiquetas):
    faltan = sorted(set(config.SENALES) - set(etiquetas))
    sobran = sorted(set(etiquetas) - set(config.SENALES))
    if faltan:
        print(f"  [aviso] El modelo no tiene estas etiquetas del proyecto: {', '.join(faltan)}")
    if sobran:
        print(f"  [aviso] El modelo tiene etiquetas que no están en config.SENALES: {', '.join(sobran)}")


def registrar_onnx(archivo, iteracion=None, dominio=None, nota=None):
    """Copia a modelos/actual/ el ONNX exportado (zip de Custom Vision o .onnx + labels.txt al lado)."""
    archivo = Path(archivo)
    if not archivo.exists():
        raise SystemExit(f"No existe {archivo}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if archivo.suffix.lower() == ".zip":
            with zipfile.ZipFile(archivo) as z:
                z.extractall(tmp)
            onnx = sorted(tmp.rglob("*.onnx"))
            labels = sorted(tmp.rglob("labels.txt"))
            if not onnx:
                encontrados = ", ".join(p.name for p in tmp.rglob("*") if p.is_file())[:300]
                raise SystemExit("El zip no contiene un modelo .onnx. En Custom Vision use Export > ONNX "
                                 f"(el proyecto debe usar un dominio 'compact'). Archivos: {encontrados}")
            onnx, labels = onnx[0], (labels[0] if labels else None)
        else:
            onnx, labels = archivo, archivo.with_name("labels.txt")
            labels = labels if labels.exists() else None
        if labels is None:
            raise SystemExit("No se encontró labels.txt junto al modelo.")

        info_previa = config.leer_json(config.INFO_MODELO, {}) or {}
        destino = config.CARPETA_MODELOS
        if destino.exists():
            for p in destino.iterdir():
                if p.name != ".gitkeep":
                    shutil.rmtree(p) if p.is_dir() else p.unlink()
        destino.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx, destino / "model.onnx")
        shutil.copy2(labels, destino / "labels.txt")

    etiquetas = [l.strip() for l in (destino / "labels.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    info = {
        "tipo": "onnx",
        "archivo": "model.onnx",
        "id": hashlib.sha256((destino / "model.onnx").read_bytes()).hexdigest()[:12],
        "iteracion": iteracion,
        "dominio": dominio,
        "nota": nota,
        "etiquetas": etiquetas,
        "origen": archivo.name,
        "registrado": datetime.now().isoformat(timespec="seconds"),
    }
    for k in ("endpoint", "iteracion_endpoint", "nota_endpoint", "endpoint_registrado", "etiquetas_endpoint"):
        if info_previa.get(k):
            info[k] = info_previa[k]
    config.escribir_json(config.INFO_MODELO, info)
    _validar_etiquetas(etiquetas)

    modelo = ClasificadorONNX()  # prueba de humo: el modelo carga y responde
    prueba = modelo.predecir(Image.new("RGB", (300, 300), (128, 128, 128)))
    print(f"Modelo ONNX registrado en {destino} (id {info['id']}, {len(etiquetas)} etiquetas, "
          f"entrada {modelo.lado}×{modelo.lado}, {'NCHW' if modelo.nchw else 'NHWC'}).")
    print(f"  Prueba de humo: responde con {len(prueba)} probabilidades (suma {prueba['probabilidad'].sum():.3f}).")
    return info


def registrar_endpoint(url, iteracion=None, nota=None, imagen_prueba=None):
    """Guarda la URL del endpoint en info.json (la clave queda solo en .streamlit/secrets.toml)."""
    clave = config.leer_secreto("CV_PREDICTION_KEY")
    if not clave:
        raise SystemExit("Defina CV_PREDICTION_KEY en .streamlit/secrets.toml o como variable de entorno.")
    clf = ClasificadorEndpoint(url, clave)
    info = config.leer_json(config.INFO_MODELO, {}) or {}
    info.update({"endpoint": clf.url_base, "iteracion_endpoint": iteracion or clf.iteracion,
                 "nota_endpoint": nota, "endpoint_registrado": datetime.now().isoformat(timespec="seconds")})
    if imagen_prueba is not None:
        pred = clf.predecir(imagen_prueba)
        info["etiquetas_endpoint"] = sorted(pred["etiqueta"])
        _validar_etiquetas(info["etiquetas_endpoint"])
        print(f"  Prueba del endpoint: {pred.iloc[0]['etiqueta']} ({100 * pred.iloc[0]['probabilidad']:.1f} %)")
    config.escribir_json(config.INFO_MODELO, info)
    print(f"Endpoint registrado: {clf.url_imagen}")
    return info
