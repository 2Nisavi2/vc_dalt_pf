#!/usr/bin/env python3
"""
vv.py - línea de comandos de VisionVial
=======================================

Flujo completo:

  1) python vv.py dataset                         Descarga GTSRB y arma data/dataset/{train,test}
     (entrenar en Azure Custom Vision con data/dataset/train/, publicar y exportar a ONNX)
  2) python vv.py modelo --archivo export.zip     Registra el modelo exportado (ONNX)
     python vv.py modelo --endpoint <URL>         Registra el endpoint publicado (clave en secrets.toml)
  3) python vv.py capturas --origen <carpeta>     Importa los pantallazos del entrenamiento en ACV
  4) python vv.py evaluar                         Calcula métricas y gráficos en resultados/
  5) python vv.py publicar                        git commit + push -> Streamlit Cloud se actualiza

  python vv.py estado                             Qué hay en el repo y qué falta
"""
import argparse
import subprocess
import sys
from pathlib import Path

from visionvial import config


def cmd_dataset(a):
    from visionvial import dataset
    dataset.generar(salida=a.salida, gtsrb_dir=a.gtsrb_dir, extra=a.extra, n_train=a.n_train,
                    n_test=a.n_test, min_lado=a.min_lado, seed=a.seed)


def cmd_modelo(a):
    from visionvial import modelo
    if not a.archivo and not a.endpoint:
        sys.exit("Indique --archivo (zip/onnx exportado de Custom Vision) y/o --endpoint (URL publicada).")
    if a.archivo:
        modelo.registrar_onnx(a.archivo, iteracion=a.iteracion, dominio=a.dominio, nota=a.nota)
    if a.endpoint:
        prueba = next((config.CARPETA_DATASET / "test").rglob("*.jpg"), None) \
            if (config.CARPETA_DATASET / "test").exists() else None
        modelo.registrar_endpoint(a.endpoint, iteracion=a.iteracion, nota=a.nota, imagen_prueba=prueba)


def cmd_capturas(a):
    from visionvial import capturas
    prefijo = "App" if a.app else "Paso"
    if a.plantilla:
        capturas.plantilla(a.origen, prefijo=prefijo)
        return
    capturas.importar(a.origen, prefijo=prefijo, descripciones=a.descripciones, ancho_max=a.ancho_max)


def cmd_evaluar(a):
    from visionvial import evaluacion, modelo
    clf = modelo.cargar_clasificador(a.backend, pausa=a.pausa)
    evaluacion.evaluar(clf, experimento=not a.sin_experimento, n_experimento=a.n_experimento)


def _git(*args, check=True):
    return subprocess.run(["git", *args], cwd=config.RAIZ, check=check, text=True, capture_output=True)


def cmd_publicar(a):
    if _git("rev-parse", "--is-inside-work-tree", check=False).returncode != 0:
        sys.exit("Esta carpeta no es un repositorio git. Ejecute: git init && git remote add origin <url>")
    rutas = [p for p in ("modelos", "resultados", "assets") if (config.RAIZ / p).exists()]
    _git("add", "--all", *rutas)
    if not _git("diff", "--cached", "--name-only").stdout.strip():
        print("No hay cambios nuevos en modelos/, resultados/ ni assets/.")
    else:
        mensaje = a.mensaje or "Actualiza modelo, métricas y capturas"
        _git("commit", "-m", mensaje)
        print(f"Commit creado: {mensaje}")
    if a.sin_push:
        return
    remoto = _git("remote", check=False).stdout.split()
    if not remoto:
        sys.exit("No hay un remoto configurado. Ejecute: git remote add origin <url-del-repo>")
    r = _git("push", check=False)
    if r.returncode != 0:
        sys.exit(f"git push falló:\n{r.stderr}")
    print("Cambios enviados. Streamlit Community Cloud redespliega la app automáticamente en unos minutos.")


def cmd_estado(_a):
    from visionvial import capturas
    def marca(ok):
        return "OK " if ok else "-- "
    test = config.CARPETA_DATASET / "test"
    n_train = sum(1 for _ in (config.CARPETA_DATASET / "train").rglob("*.jpg")) if test.exists() else 0
    n_test = sum(1 for _ in test.rglob("*.jpg")) if test.exists() else 0
    info = config.leer_json(config.INFO_MODELO, {}) or {}
    metricas = config.leer_json(config.METRICAS)
    caps = capturas.leer_indice(config.CARPETA_CAPTURAS)
    caps_app = capturas.leer_indice(config.CARPETA_CAPTURAS_APP)
    clave = bool(config.leer_secreto("CV_PREDICTION_KEY"))
    print("ESTADO DE VISIONVIAL")
    print(f"  {marca(n_test > 0)}Dataset:          {n_train} imágenes de entrenamiento, {n_test} de prueba")
    print(f"  {marca(info.get('tipo') == 'onnx')}Modelo ONNX:      "
          f"{info.get('id', 'sin registrar')} {('· iteración ' + str(info['iteracion'])) if info.get('iteracion') else ''}")
    print(f"  {marca(bool(info.get('endpoint')) and clave)}Endpoint:         "
          f"{info.get('endpoint', 'sin registrar')} · clave {'configurada' if clave else 'FALTA'}")
    print(f"  {marca(len(caps) > 0)}Capturas ACV:     {len(caps)}")
    print(f"  {marca(len(caps_app) > 0)}Capturas de app:  {len(caps_app)}")
    if metricas:
        g = metricas["global"]
        print(f"  OK Métricas:         accuracy {g['accuracy']:.3f}, F1 macro {g['f1_macro']:.3f}, "
              f"AUC macro {g['auc_macro']:.3f} ({metricas['modelo']['id']}, {metricas['evaluado']})")
        if metricas["modelo"]["tipo"] == "onnx" and info.get("id") and metricas["modelo"]["id"] != f"onnx_{info['id']}":
            print("     [aviso] Las métricas son de otro modelo ONNX: vuelva a ejecutar `python vv.py evaluar`.")
    else:
        print("  -- Métricas:         sin evaluar")


def main():
    p = argparse.ArgumentParser(prog="vv", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="comando", required=True)

    s = sub.add_parser("dataset", help="Descarga GTSRB y genera data/dataset/{train,test}")
    s.add_argument("--salida", default=config.CARPETA_DATASET, type=Path)
    s.add_argument("--gtsrb-dir", help="GTSRB ya descargado y descomprimido (omite la descarga)")
    s.add_argument("--extra", help="Fotos propias: <extra>/<etiqueta>/*.jpg")
    s.add_argument("--n-train", type=int, default=150)
    s.add_argument("--n-test", type=int, default=50)
    s.add_argument("--min-lado", type=int, default=32)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_dataset)

    s = sub.add_parser("modelo", help="Registra el modelo entrenado (ONNX exportado y/o endpoint)")
    s.add_argument("--archivo", help="Zip exportado por Custom Vision (Export > ONNX) o archivo .onnx")
    s.add_argument("--endpoint", help="Prediction URL de la iteración publicada")
    s.add_argument("--iteracion", help="Nombre de la iteración (para el registro)")
    s.add_argument("--dominio", help="Dominio del proyecto, p. ej. 'General (compact) [S1]'")
    s.add_argument("--nota")
    s.set_defaults(func=cmd_modelo)

    s = sub.add_parser("capturas", help="Importa pantallazos del entrenamiento en ACV (o de la app con --app)")
    s.add_argument("--origen", required=True, type=Path, help="Carpeta con los pantallazos")
    s.add_argument("--app", action="store_true", help="Son pantallazos de la aplicación (App_XX)")
    s.add_argument("--descripciones", type=Path, help="CSV archivo,titulo,descripcion (por defecto <origen>/descripciones.csv)")
    s.add_argument("--plantilla", action="store_true", help="Solo genera <origen>/descripciones.csv para completar")
    s.add_argument("--ancho-max", type=int, default=1600)
    s.set_defaults(func=cmd_capturas)

    s = sub.add_parser("evaluar", help="Evalúa el modelo con data/dataset/test y escribe resultados/")
    s.add_argument("--backend", choices=["auto", "onnx", "endpoint"], default="onnx",
                   help="onnx (local, sin costo; por defecto), endpoint (Azure) o auto")
    s.add_argument("--n-experimento", type=int, default=20, help="Imágenes por clase en el experimento de daltonismo")
    s.add_argument("--sin-experimento", action="store_true")
    s.add_argument("--pausa", type=float, default=0.12, help="Segundos entre llamadas al endpoint")
    s.set_defaults(func=cmd_evaluar)

    s = sub.add_parser("publicar", help="Commit de modelos/, resultados/ y assets/ y push (actualiza Streamlit)")
    s.add_argument("-m", "--mensaje")
    s.add_argument("--sin-push", action="store_true")
    s.set_defaults(func=cmd_publicar)

    s = sub.add_parser("estado", help="Muestra qué hay en el repositorio y qué falta")
    s.set_defaults(func=cmd_estado)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
