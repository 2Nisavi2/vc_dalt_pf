"""
evaluacion.py
-------------
Evalúa el modelo registrado (ONNX o endpoint) sobre data/dataset/test/ y escribe en resultados/:

  metricas.json                resumen que lee la app (accuracy, F1, AUC, umbral recomendado, experimento)
  reporte_clasificacion.csv    precision, recall, F1 y soporte por clase
  matriz_confusion.csv / .png
  auc_por_clase.csv, curvas_roc.png
  desempeno_por_categoria.csv
  umbral_confianza.csv / .png
  resultados_experimento.csv, experimento_daltonismo.png
  metricas_por_clase.png, errores.png
  cache/<modelo>/<condicion>.csv   predicciones guardadas (no se repiten llamadas al endpoint)
"""
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix,  # noqa: E402
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize  # noqa: E402

from . import config  # noqa: E402
from . import daltonismo as dal  # noqa: E402

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})

TIPOS = list(dal.TIPOS)
PRECISION_OBJETIVO_UMBRAL = 0.95


# ====================================================================== datos
def conjunto_prueba(dataset=config.CARPETA_DATASET):
    carpeta = Path(dataset) / "test"
    if not carpeta.exists():
        raise SystemExit(f"No existe {carpeta}. Ejecute primero: python vv.py dataset")
    filas = [{"id": f"test/{sub.name}/{f.name}", "clase": sub.name, "archivo": f.name, "ruta": str(f)}
             for sub in sorted(p for p in carpeta.iterdir() if p.is_dir())
             for f in sorted(sub.iterdir()) if f.suffix.lower() in config.EXTENSIONES_IMG]
    return pd.DataFrame(filas)


def predecir_conjunto(clasificador, registros, condicion, transformar=None, cada=50):
    """Una fila por imagen con la probabilidad de cada etiqueta; caché por modelo y condición."""
    ruta_cache = config.CARPETA_CACHE / clasificador.id / f"{condicion}.csv"
    ruta_cache.parent.mkdir(parents=True, exist_ok=True)
    previas = pd.read_csv(ruta_cache) if ruta_cache.exists() else pd.DataFrame(columns=["id"])
    pendientes = registros[~registros["id"].isin(set(previas["id"]))]
    if len(pendientes):
        print(f"  [{condicion}] {len(pendientes)} imágenes por clasificar ({len(registros) - len(pendientes)} en caché)")
    nuevas, errores = [], 0
    for i, (_, fila) in enumerate(pendientes.iterrows(), 1):
        with Image.open(fila["ruta"]) as img:
            arr = np.asarray(img.convert("RGB"))
        if transformar is not None:
            arr = transformar(arr)
        try:
            pred = clasificador.predecir(arr)
        except Exception as e:  # noqa: BLE001
            errores += 1
            print(f"    {fila['id']}: {e}")
            if "401" in str(e) or "404" in str(e) or errores >= 10:
                print("    Se detiene: revise la clave, la iteración publicada o la conexión.")
                break
            continue
        nuevas.append({"id": fila["id"], "clase_real": fila["clase"], "condicion": condicion,
                       **dict(zip(pred["etiqueta"], pred["probabilidad"]))})
        if i % cada == 0 or i == len(pendientes):
            pd.concat([previas, pd.DataFrame(nuevas)], ignore_index=True).to_csv(ruta_cache, index=False)
            print(f"    {i}/{len(pendientes)}")
    todas = pd.concat([previas, pd.DataFrame(nuevas)], ignore_index=True)
    return todas[todas["id"].isin(set(registros["id"]))].reset_index(drop=True)


def separar(pred, clases):
    pred = pred.copy()
    for c in clases:
        if c not in pred:
            pred[c] = 0.0
    P = pred[clases].fillna(0.0).to_numpy(dtype=float)
    y_real = pred["clase_real"].to_numpy()
    y_pred = np.array(clases)[P.argmax(axis=1)]
    return y_real, y_pred, P


def categoria_de(etiquetas):
    return np.array([config.CATEGORIA.get(e, ("-",))[0] for e in etiquetas])


def _guardar_fig(fig, nombre, salida):
    fig.savefig(Path(salida) / nombre, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ====================================================================== métricas y gráficos
def _graficos_metricas(reporte, clases, salida):
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(clases) + 1.3))
    sub = reporte.loc[clases[::-1]]
    alto = 0.27
    for k, (m, color) in enumerate([("precision", "#2a78d6"), ("recall", "#eb6834"), ("f1-score", "#1baf7a")]):
        ax.barh(np.arange(len(sub)) + (k - 1) * alto, sub[m], height=alto, color=color, label=m)
    ax.set_yticks(np.arange(len(sub)), sub.index)
    ax.set_xlim(0, 1.05)
    ax.set_title("Precision, recall y F1-score por clase (conjunto de prueba)")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.05))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _guardar_fig(fig, "metricas_por_clase.png", salida)


def _grafico_confusion(mc, clases, salida):
    mc_norm = mc / np.maximum(mc.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"wspace": 0.08})
    for ax, matriz, titulo, fmt in [(axes[0], mc, "Conteo", "{:d}"), (axes[1], mc_norm, "Normalizada por fila", "{:.0%}")]:
        im = ax.imshow(mc_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(clases)), clases, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(clases)), clases if ax is axes[0] else [""] * len(clases), fontsize=8)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Clase real" if ax is axes[0] else "")
        ax.set_title(titulo)
        ax.grid(False)
        for (i, j), v in np.ndenumerate(matriz):
            if mc[i, j] > 0:
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                        color="white" if mc_norm[i, j] > 0.6 else "#0b0b0b")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Proporción de la clase real")
    _guardar_fig(fig, "matriz_confusion.png", salida)


def _roc(y_real, P, clases, salida):
    Y = label_binarize(y_real, classes=clases)
    if Y.shape[1] == 1:  # caso binario
        Y = np.hstack([1 - Y, Y])
    auc_clase, curvas = {}, {}
    for i, c in enumerate(clases):
        if Y[:, i].sum() == 0 or Y[:, i].sum() == len(Y):
            continue
        fpr, tpr, _ = roc_curve(Y[:, i], P[:, i])
        curvas[c] = (fpr, tpr)
        auc_clase[c] = float(auc(fpr, tpr))
    fpr_micro, tpr_micro, _ = roc_curve(Y.ravel(), P.ravel())
    auc_micro = float(auc(fpr_micro, tpr_micro))
    auc_macro = float(np.mean(list(auc_clase.values()))) if auc_clase else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, zoom in [(axes[0], False), (axes[1], True)]:
        for c, (fpr, tpr) in curvas.items():
            ax.plot(fpr, tpr, color=config.COLOR_CLASE.get(c, "#888"), linewidth=1.4,
                    label=f"{c} (AUC {auc_clase[c]:.3f})")
        ax.plot(fpr_micro, tpr_micro, color="#0b0b0b", linewidth=2, linestyle=":", label=f"micro (AUC {auc_micro:.3f})")
        ax.plot([0, 1], [0, 1], color="#bdbdbd", linestyle="--", linewidth=1)
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Tasa de verdaderos positivos")
        if zoom:
            ax.set_xlim(0, 0.2)
            ax.set_ylim(0.8, 1.005)
            ax.set_title("Detalle de la esquina superior izquierda")
        else:
            ax.set_title(f"Curvas ROC uno-contra-resto · AUC macro {auc_macro:.3f}")
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    _guardar_fig(fig, "curvas_roc.png", salida)

    tabla = pd.DataFrame({"AUC": auc_clase}).rename_axis("clase")
    tabla["categoria"] = [config.CATEGORIA.get(c, ("-",))[0] for c in tabla.index]
    tabla.loc["micro"] = [auc_micro, ""]
    tabla.loc["macro"] = [auc_macro, ""]
    tabla["AUC"] = tabla["AUC"].astype(float).round(4)
    tabla.to_csv(Path(salida) / "auc_por_clase.csv")
    return auc_clase, auc_micro, auc_macro


def _umbral(resultado, salida):
    filas = []
    for u in np.round(np.arange(0, 0.96, 0.05), 2):
        cubiertas = resultado["confianza"] >= u
        filas.append({"umbral": float(u), "cobertura_pct": 100 * cubiertas.mean(),
                      "accuracy_pct": 100 * resultado.loc[cubiertas, "acierto"].mean() if cubiertas.any() else np.nan})
    tabla = pd.DataFrame(filas).set_index("umbral").round(2)
    tabla.to_csv(Path(salida) / "umbral_confianza.csv")

    # Umbral recomendado: el menor que logra la precisión objetivo con la mayor cobertura posible
    candidatos = tabla[tabla["accuracy_pct"] >= 100 * PRECISION_OBJETIVO_UMBRAL]
    recomendado = float(candidatos.index.min()) if len(candidatos) else 0.6

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tabla.index, tabla["cobertura_pct"], color="#2a78d6", linewidth=2, label="Cobertura")
    ax.plot(tabla.index, tabla["accuracy_pct"], color="#eb6834", linewidth=2, label="Accuracy (imágenes cubiertas)")
    ax.axvline(recomendado, color="#52514e", linestyle="--", linewidth=1, label=f"Umbral recomendado {recomendado:.2f}")
    ax.set_xlabel("Umbral de confianza")
    ax.set_ylabel("%")
    ax.set_title("Cobertura y accuracy según el umbral de confianza")
    ax.legend(frameon=False)
    fig.tight_layout()
    _guardar_fig(fig, "umbral_confianza.png", salida)
    return recomendado, tabla


def _galeria_errores(resultado, prueba, salida, n=12):
    errores = resultado[~resultado["acierto"]].sort_values("confianza", ascending=False).head(n)
    ruta_png = Path(salida) / "errores.png"
    if not len(errores):
        ruta_png.unlink(missing_ok=True)
        return
    rutas = prueba.set_index("id")["ruta"]
    columnas = 6
    filas = int(np.ceil(len(errores) / columnas))
    fig, axes = plt.subplots(filas, columnas, figsize=(2.3 * columnas, 2.7 * filas), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, e) in zip(axes.ravel(), errores.iterrows()):
        with Image.open(rutas[e["id"]]) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"real: {e['clase_real']}\npred: {e['prediccion']} ({100 * e['confianza']:.0f} %)", fontsize=7)
    fig.suptitle("Errores con mayor confianza")
    fig.tight_layout()
    _guardar_fig(fig, "errores.png", salida)


def _experimento(clasificador, prueba, clases, n_por_clase, salida):
    muestra = pd.concat([g.sample(min(len(g), n_por_clase), random_state=0)
                         for _, g in prueba.groupby("clase")], ignore_index=True)
    condiciones = {"original": None}
    for t in TIPOS:
        condiciones[f"{t}_sin_correccion"] = (lambda a, t=t: dal.simular(a, t))
        condiciones[f"{t}_con_correccion"] = (lambda a, t=t: dal.simular(dal.corregir(a, t), t))
    print(f"Experimento de daltonismo: {len(muestra)} imágenes × {len(condiciones)} condiciones")

    filas = []
    for condicion, transformar in condiciones.items():
        pred = predecir_conjunto(clasificador, muestra, condicion, transformar)
        if not len(pred):
            continue
        yr, yp, Pc = separar(pred, clases)
        Yc = label_binarize(yr, classes=clases)
        presentes = (Yc.sum(axis=0) > 0) & (Yc.sum(axis=0) < len(Yc))
        tipo = next((t for t in TIPOS if condicion.startswith(t)), "ninguno")
        filas.append({
            "condicion": condicion, "tipo": tipo,
            "correccion": "con" if condicion.endswith("con_correccion") else ("sin" if tipo != "ninguno" else "-"),
            "imagenes": len(pred),
            "accuracy": accuracy_score(yr, yp),
            "f1_macro": f1_score(yr, yp, labels=clases, average="macro", zero_division=0),
            "auc_macro": roc_auc_score(Yc[:, presentes], Pc[:, presentes], average="macro") if presentes.sum() > 1 else np.nan,
            "error_categoria_pct": 100 * (categoria_de(yr) != categoria_de(yp)).mean(),
            "confianza_media": Pc.max(axis=1).mean(),
        })
    res = pd.DataFrame(filas).set_index("condicion").round(4)
    res.to_csv(Path(salida) / "resultados_experimento.csv")

    base = res.loc["original"] if "original" in res.index else None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metrica, titulo in [(axes[0], "accuracy", "Accuracy"), (axes[1], "f1_macro", "F1 macro"),
                                (axes[2], "error_categoria_pct", "Error de categoría (%)")]:
        x = np.arange(len(TIPOS))
        for k, (corr, color) in enumerate([("sin", "#bdbdbd"), ("con", "#1baf7a")]):
            valores = [res.loc[f"{t}_{corr}_correccion", metrica] if f"{t}_{corr}_correccion" in res.index else np.nan
                       for t in TIPOS]
            barras = ax.bar(x + (k - 0.5) * 0.38, valores, width=0.38, color=color, label=f"{corr} corrección")
            ax.bar_label(barras, fmt="%.2f" if metrica != "error_categoria_pct" else "%.1f", fontsize=8, padding=2)
        if base is not None:
            ax.axhline(base[metrica], color="#2a78d6", linestyle="--", linewidth=1.5, label="original")
        ax.set_xticks(x, [dal.TIPOS[t].split(" ")[0] for t in TIPOS])
        ax.set_title(titulo)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Desempeño del modelo con la imagen como la percibe una persona con daltonismo")
    fig.tight_layout()
    _guardar_fig(fig, "experimento_daltonismo.png", salida)
    return res


# ====================================================================== orquestación
def evaluar(clasificador, dataset=config.CARPETA_DATASET, salida=config.CARPETA_RESULTADOS,
            experimento=True, n_experimento=20):
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    prueba = conjunto_prueba(dataset)
    clases = sorted(prueba["clase"].unique())
    print(f"Evaluando {clasificador.descripcion}")
    print(f"Conjunto de prueba: {len(prueba)} imágenes, {len(clases)} clases")

    pred = predecir_conjunto(clasificador, prueba, "original")
    if not len(pred):
        raise SystemExit("No se obtuvo ninguna predicción.")
    y_real, y_pred, P = separar(pred, clases)

    reporte = pd.DataFrame(classification_report(y_real, y_pred, labels=clases, output_dict=True, zero_division=0)).T
    reporte.loc["accuracy", ["precision", "recall", "f1-score"]] = accuracy_score(y_real, y_pred)
    reporte.loc["accuracy", "support"] = len(y_real)
    reporte["categoria"] = [config.CATEGORIA.get(i, ("", ""))[0] for i in reporte.index]
    reporte = reporte.round(4)
    reporte.to_csv(salida / "reporte_clasificacion.csv")
    _graficos_metricas(reporte, clases, salida)

    mc = confusion_matrix(y_real, y_pred, labels=clases)
    pd.DataFrame(mc, index=clases, columns=clases).to_csv(salida / "matriz_confusion.csv")
    _grafico_confusion(mc, clases, salida)

    auc_clase, auc_micro, auc_macro = _roc(y_real, P, clases, salida)

    resultado = pred.assign(prediccion=y_pred, confianza=P.max(axis=1),
                            categoria_real=categoria_de(y_real), categoria_pred=categoria_de(y_pred))
    resultado["acierto"] = resultado["clase_real"] == resultado["prediccion"]
    resultado["acierto_categoria"] = resultado["categoria_real"] == resultado["categoria_pred"]
    por_categoria = resultado.groupby("categoria_real").agg(
        imagenes=("id", "count"), accuracy=("acierto", "mean"),
        error_categoria_pct=("acierto_categoria", lambda s: 100 * (1 - s.mean()))).round(4)
    por_categoria.to_csv(salida / "desempeno_por_categoria.csv")
    resultado[["id", "clase_real", "prediccion", "confianza", "acierto"]].to_csv(salida / "predicciones_prueba.csv",
                                                                                 index=False)
    _galeria_errores(resultado, prueba, salida)
    umbral, _ = _umbral(resultado, salida)

    res_exp = _experimento(clasificador, prueba, clases, n_experimento, salida) if experimento else None

    info = config.leer_json(config.INFO_MODELO, {}) or {}
    metricas = {
        "evaluado": datetime.now().isoformat(timespec="seconds"),
        "modelo": {"id": clasificador.id, "tipo": clasificador.tipo, "descripcion": clasificador.descripcion,
                   "iteracion": getattr(clasificador, "iteracion", None), "dominio": info.get("dominio")},
        "conjunto_prueba": {"imagenes": int(len(y_real)), "clases": clases},
        "global": {
            "accuracy": float(accuracy_score(y_real, y_pred)),
            "precision_macro": float(precision_score(y_real, y_pred, labels=clases, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_real, y_pred, labels=clases, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_real, y_pred, labels=clases, average="macro", zero_division=0)),
            "auc_macro": auc_macro, "auc_micro": auc_micro,
            "error_categoria_pct": float(100 * (~resultado["acierto_categoria"]).mean()),
        },
        "umbral_recomendado": umbral,
        "por_clase": [{"clase": c, **{k: float(reporte.loc[c, k]) for k in ("precision", "recall", "f1-score", "support")},
                       "auc": auc_clase.get(c)} for c in clases],
        "experimento": (res_exp.reset_index().to_dict(orient="records") if res_exp is not None else None),
    }
    config.escribir_json(config.METRICAS, metricas)

    g = metricas["global"]
    print("\nRESULTADOS")
    print(f"  Accuracy {g['accuracy']:.3f} · F1 macro {g['f1_macro']:.3f} · AUC macro {g['auc_macro']:.3f} "
          f"· error de categoría {g['error_categoria_pct']:.1f} %")
    print(f"  Umbral de confianza recomendado para la app: {umbral:.2f}")
    print(f"  Archivos en: {salida}")
    return metricas
