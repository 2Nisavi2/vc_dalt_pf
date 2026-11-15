"""
app.py - VisionVial (Streamlit)
-------------------------------
Lee del repositorio lo que dejan los comandos de vv.py, así que se actualiza sola al hacer push:

  modelos/actual/       modelo registrado (ONNX) e info del endpoint      <- python vv.py modelo
  resultados/           métricas y gráficos de evaluación                 <- python vv.py evaluar
  assets/entrenamiento/ pantallazos del entrenamiento en Custom Vision    <- python vv.py capturas

Ejecución local: streamlit run app.py
"""
import hashlib
import io
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from visionvial import capturas, config
from visionvial import daltonismo as dal
from visionvial import modelo as mod

st.set_page_config(page_title="VisionVial · Señales para daltonismo", page_icon="🚦", layout="wide")

LADO_MAX_PROCESO = 1280
EXT_SUBIDA = ["jpg", "jpeg", "png", "bmp", "webp"]


# ====================================================================== carga de artefactos del repo
def _huella_modelo():
    """Cambia cuando se registra otro modelo o cambian las credenciales -> invalida la caché."""
    partes = [str(p.stat().st_mtime_ns) for p in sorted(config.CARPETA_MODELOS.glob("*")) if p.is_file()]
    return "|".join(partes)


def motores_disponibles():
    url, clave = mod.credenciales_endpoint(st.secrets)
    motores = []
    if url and clave:
        motores.append("endpoint")
    if (config.CARPETA_MODELOS / "labels.txt").exists() and any(config.CARPETA_MODELOS.glob("*.onnx")):
        motores.append("onnx")
    return motores


@st.cache_resource(show_spinner="Cargando el modelo...")
def obtener_clasificador(motor: str, _huella: str):
    return mod.cargar_clasificador(motor, secretos_streamlit=st.secrets)


@st.cache_data(show_spinner=False, max_entries=1000)
def _predecir_cacheado(_clf, id_modelo: str, huella_img: str, _datos: bytes) -> pd.DataFrame:
    return _clf.predecir(_datos)


def clasificar(clf, img: Image.Image) -> pd.DataFrame:
    datos = mod.a_jpeg(img, 92)
    return _predecir_cacheado(clf, clf.id, hashlib.md5(datos).hexdigest(), datos)


def abrir(contenido: bytes) -> Image.Image:
    img = mod.a_pil(contenido)
    img.thumbnail((LADO_MAX_PROCESO, LADO_MAX_PROCESO))
    return img


# ====================================================================== componentes
def tarjeta_senal(pred: pd.DataFrame, umbral: float):
    top = pred.iloc[0]
    nombre, categoria, color, significado = config.info_senal(top["etiqueta"])
    prob = 100 * top["probabilidad"]
    if top["probabilidad"] < umbral:
        st.warning(f"No hay una señal clara (la más probable es **{nombre}** con {prob:.0f} %). "
                   "Acerque la cámara y encuadre la señal en el centro.")
        return
    st.markdown(
        f"""
        <div style="border:1px solid #ddd;border-radius:12px;padding:16px 18px;">
          <div style="font-size:0.85rem;opacity:0.7;">Señal reconocida · {prob:.1f} % de confianza</div>
          <div style="font-size:1.8rem;font-weight:700;margin:2px 0 8px;">{nombre}</div>
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
            <span style="display:inline-block;width:22px;height:22px;border-radius:50%;
                         background:{config.COLOR_HEX.get(color, '#888')};border:2px solid #333;"></span>
            <span><b>{categoria}</b> · color <b>{color.upper()}</b></span>
          </div>
          <div>{significado}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    with st.expander("Probabilidad de cada clase"):
        tabla = pred.head(5).copy()
        tabla["señal"] = tabla["etiqueta"].map(lambda e: config.info_senal(e)[0])
        st.bar_chart(tabla.set_index("señal")["probabilidad"] * 100, horizontal=True)


def panel_correccion(img: Image.Image, tipo, severidad, intensidad, nombre="imagen"):
    arr = np.asarray(img)
    corregida = Image.fromarray(dal.corregir(arr, tipo, severidad, intensidad))
    c1, c2 = st.columns(2)
    c1.image(img, caption="Original")
    c2.image(corregida, caption=f"Corregida para {dal.TIPOS[tipo].lower()}")
    st.download_button("Descargar imagen corregida", mod.a_jpeg(corregida, 92),
                       file_name=f"{nombre}_corregida.jpg", mime="image/jpeg")
    with st.expander("¿Cómo se percibe? (simulación para personas sin daltonismo)"):
        s1, s2 = st.columns(2)
        s1.image(dal.simular(arr, tipo, severidad), caption="Original, como la percibe la persona")
        s2.image(dal.simular(np.asarray(corregida), tipo, severidad), caption="Corregida, como la percibe la persona")


def leer_lote(archivos):
    """(nombre, etiqueta_real o None, bytes) desde imágenes sueltas o ZIP con carpetas por etiqueta."""
    items = []
    for f in archivos:
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(f.getvalue())) as z:
                for n in z.namelist():
                    p = PurePosixPath(n)
                    if (p.suffix.lower().lstrip(".") in EXT_SUBIDA and not p.name.startswith("._")
                            and "__MACOSX" not in p.parts):
                        real = p.parent.name if p.parent.name in config.SENALES else None
                        items.append((str(p), real, z.read(n)))
        elif f.name.lower().rsplit(".", 1)[-1] in EXT_SUBIDA:
            items.append((f.name, None, f.getvalue()))
    return items


def galeria(carpeta: Path, vacio: str):
    indice = capturas.leer_indice(carpeta)
    if not indice:
        st.info(vacio)
        return
    for c in indice:
        st.markdown(f"#### {c['titulo']}")
        if c.get("descripcion"):
            st.caption(c["descripcion"])
        st.image(str(carpeta / c["archivo"]))


# ====================================================================== interfaz
metricas = config.leer_json(config.METRICAS)
info_modelo = config.leer_json(config.INFO_MODELO, {}) or {}
motores = motores_disponibles()

st.title("🚦 VisionVial")
st.caption("Reconocimiento de señales de tránsito y corrección de color para personas con daltonismo · "
           "Modelo entrenado en Azure Custom Vision")

with st.sidebar:
    st.header("Tu perfil de visión")
    tipo = st.radio("Tipo de daltonismo", list(dal.TIPOS), format_func=dal.TIPOS.get)
    severidad = st.slider("Severidad", 0.0, 1.0, 1.0, 0.1,
                          help="1 = dicromacia (no percibe ese tipo de cono); valores menores = anomalía parcial.")
    intensidad = st.slider("Intensidad de la corrección", 0.0, 1.5, 1.0, 0.1)
    st.divider()
    st.header("Modelo")
    umbral_def = float(metricas["umbral_recomendado"]) if metricas else 0.6
    umbral = st.slider("Confianza mínima para mostrar la señal", 0.0, 1.0, umbral_def, 0.05,
                       help="Valor por defecto: umbral recomendado por la evaluación (python vv.py evaluar).")
    clf = None
    if motores:
        etiquetas_motor = {"endpoint": "Azure endpoint (API)", "onnx": "ONNX local"}
        motor = st.radio("Motor de predicción", motores, format_func=etiquetas_motor.get,
                         help="El endpoint consume la API publicada en Azure; ONNX corre el modelo exportado en el servidor.")
        try:
            clf = obtener_clasificador(motor, _huella_modelo())
            st.success(clf.descripcion)
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo cargar el modelo: {e}")
    else:
        st.error("No hay modelo: registre el ONNX (`python vv.py modelo`) o configure el endpoint en los secretos.")
    if metricas:
        g = metricas["global"]
        st.caption(f"Evaluación ({metricas['evaluado'][:10]}): accuracy {g['accuracy']:.1%} · "
                   f"F1 {g['f1_macro']:.3f} · AUC {g['auc_macro']:.3f}")

tab_foto, tab_lote, tab_sim, tab_metricas, tab_acv, tab_info = st.tabs(
    ["📷 Foto", "🗂️ Lote / dataset", "👁️ Simulador", "📊 Desempeño del modelo", "🛠️ Entrenamiento en ACV", "ℹ️ Acerca de"])

# --------------------------------------------------------------------- Foto
with tab_foto:
    fuente = st.radio("Origen de la imagen", ["Cámara", "Subir archivo"], horizontal=True,
                      help="En el celular, 'Subir archivo' también permite tomar la foto con la cámara trasera.")
    archivo = (st.camera_input("Encuadre la señal en el centro y tome la foto") if fuente == "Cámara"
               else st.file_uploader("Imagen", type=EXT_SUBIDA))
    if archivo is not None:
        img = abrir(archivo.getvalue())
        if clf is not None:
            try:
                with st.spinner("Consultando el modelo..."):
                    pred = clasificar(clf, img)
                tarjeta_senal(pred, umbral)
            except Exception as e:  # noqa: BLE001
                st.error(f"No fue posible clasificar la imagen. {e}")
        st.subheader("Corrección de color")
        panel_correccion(img, tipo, severidad, intensidad, nombre=PurePosixPath(getattr(archivo, "name", "foto")).stem)

# --------------------------------------------------------------------- Lote
with tab_lote:
    st.write("Suba varias imágenes o un **ZIP**. Si el ZIP tiene una carpeta por señal "
             "(`pare/`, `obras/`, ... como `data/dataset/test/`), la app compara la predicción con la etiqueta real.")
    archivos = st.file_uploader("Imágenes o ZIP", type=EXT_SUBIDA + ["zip"], accept_multiple_files=True, key="lote")
    if archivos and st.button("Clasificar lote", type="primary", disabled=clf is None):
        items = leer_lote(archivos)
        filas, barra = [], st.progress(0.0, text="Clasificando...")
        for i, (nombre, real, contenido) in enumerate(items, 1):
            try:
                pred = clasificar(clf, abrir(contenido))
                top = pred.iloc[0]
                n, cat, col, _ = config.info_senal(top["etiqueta"])
                filas.append({"archivo": nombre, "etiqueta_real": real, "prediccion": top["etiqueta"], "señal": n,
                              "categoria": cat, "color": col, "probabilidad_%": round(100 * top["probabilidad"], 2),
                              "segura": bool(top["probabilidad"] >= umbral)})
            except Exception as e:  # noqa: BLE001
                filas.append({"archivo": nombre, "etiqueta_real": real, "prediccion": None, "error": str(e)})
            barra.progress(i / max(len(items), 1), text=f"Clasificando {i}/{len(items)}")
        st.session_state["resultado_lote"] = pd.DataFrame(filas)
    res = st.session_state.get("resultado_lote")
    if res is not None and len(res):
        validas = res.dropna(subset=["prediccion"])
        con_real = validas.dropna(subset=["etiqueta_real"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Imágenes", len(res))
        c2.metric("Confianza media", f"{validas['probabilidad_%'].mean():.1f} %" if len(validas) else "-")
        if len(con_real):
            c3.metric("Accuracy", f"{100 * (con_real['etiqueta_real'] == con_real['prediccion']).mean():.1f} %")
        st.dataframe(res, hide_index=True)
        if len(con_real):
            st.write("**Matriz de confusión** (filas = real, columnas = predicción)")
            st.dataframe(pd.crosstab(con_real["etiqueta_real"], con_real["prediccion"]))
        st.download_button("Descargar resultados (CSV)", res.to_csv(index=False).encode("utf-8"),
                           file_name="resultados_visionvial.csv", mime="text/csv")

# --------------------------------------------------------------------- Simulador
with tab_sim:
    st.write("Compare cómo se percibe una imagen con cada tipo de daltonismo, antes y después de la corrección.")
    sim = st.file_uploader("Imagen", type=EXT_SUBIDA, key="sim")
    if sim is not None:
        img_sim = abrir(sim.getvalue())
        img_sim.thumbnail((800, 800))
        arr = np.asarray(img_sim)
        st.image(arr, caption="Visión típica", width=320)
        for t, etiqueta in dal.TIPOS.items():
            corr = dal.corregir(arr, t, severidad, intensidad)
            a, b = st.columns(2)
            a.image(dal.simular(arr, t, severidad), caption=f"{etiqueta}: sin corrección")
            b.image(dal.simular(corr, t, severidad), caption=f"{etiqueta}: con corrección")

# --------------------------------------------------------------------- Desempeño
with tab_metricas:
    if not metricas:
        st.info("Aún no hay evaluación. Ejecute `python vv.py evaluar` y publique los resultados (`python vv.py publicar`).")
    else:
        g = metricas["global"]
        st.caption(f"{metricas['modelo']['descripcion']} · evaluado {metricas['evaluado'].replace('T', ' ')} · "
                   f"{metricas['conjunto_prueba']['imagenes']} imágenes de prueba independientes (no usadas en el entrenamiento)")
        cols = st.columns(6)
        for col, (titulo, valor) in zip(cols, [
                ("Accuracy", f"{g['accuracy']:.1%}"), ("Precision macro", f"{g['precision_macro']:.3f}"),
                ("Recall macro", f"{g['recall_macro']:.3f}"), ("F1 macro", f"{g['f1_macro']:.3f}"),
                ("AUC macro", f"{g['auc_macro']:.3f}"), ("Error de categoría", f"{g['error_categoria_pct']:.1f} %")]):
            col.metric(titulo, valor)

        R = config.CARPETA_RESULTADOS

        def mostrar(png, titulo, texto=None):
            if (R / png).exists():
                st.subheader(titulo)
                if texto:
                    st.caption(texto)
                st.image(str(R / png))

        st.subheader("Métricas por clase")
        st.dataframe(pd.DataFrame(metricas["por_clase"]).set_index("clase").round(3))
        mostrar("metricas_por_clase.png", "Precision, recall y F1-score")
        mostrar("matriz_confusion.png", "Matriz de confusión", "Filas: clase real; columnas: predicción.")
        mostrar("curvas_roc.png", "Curvas ROC y AUC", "Cada clase contra el resto (one-vs-rest).")
        if (R / "desempeno_por_categoria.csv").exists():
            st.subheader("Desempeño por categoría")
            st.caption("El error de categoría (p. ej. una reglamentaria tomada por preventiva) es el más grave para el usuario.")
            st.dataframe(pd.read_csv(R / "desempeno_por_categoria.csv", index_col=0))
        mostrar("umbral_confianza.png", "Umbral de confianza",
                f"Umbral recomendado {metricas['umbral_recomendado']:.2f}: es el valor por defecto de la barra lateral.")
        mostrar("errores.png", "Errores con mayor confianza")
        if metricas.get("experimento"):
            mostrar("experimento_daltonismo.png", "Experimento: robustez ante el daltonismo",
                    "El modelo clasifica la imagen tal como la percibe una persona con cada tipo de daltonismo, "
                    "sin corrección y con la corrección de la app.")
            st.dataframe(pd.DataFrame(metricas["experimento"]).set_index("condicion").round(3))

# --------------------------------------------------------------------- Entrenamiento ACV
with tab_acv:
    st.write("Proceso de entrenamiento del modelo en **Azure Custom Vision**: creación del recurso, carga de "
             "imágenes por etiqueta, iteraciones, publicación y exportación.")
    if info_modelo:
        datos_modelo = {k: v for k, v in {
            "Iteración (ONNX)": info_modelo.get("iteracion"), "Dominio": info_modelo.get("dominio"),
            "Id del modelo": info_modelo.get("id"), "Registrado": info_modelo.get("registrado"),
            "Iteración del endpoint": info_modelo.get("iteracion_endpoint"),
            "Etiquetas": ", ".join(info_modelo.get("etiquetas") or info_modelo.get("etiquetas_endpoint") or []),
        }.items() if v}
        st.table(pd.DataFrame(datos_modelo.items(), columns=["", "valor"]).set_index(""))
    galeria(config.CARPETA_CAPTURAS, "Aún no hay capturas. Importe los pantallazos con "
                                     "`python vv.py capturas --origen <carpeta>` y publíquelos.")

# --------------------------------------------------------------------- Acerca de
with tab_info:
    st.markdown(
        """
**Cómo funciona**

1. La foto **original** se envía al modelo entrenado en **Azure Custom Vision** (por la API publicada o con el
   modelo exportado en ONNX), que reconoce la señal con toda la información de color.
2. La categoría y el color se muestran **en texto** y con una muestra de color, para no depender de la percepción del color.
3. La imagen se **daltoniza**: se simula cómo la percibe la persona (Machado et al., 2009), se calcula la
   información de color que pierde y se redistribuye hacia los canales que sí percibe (Fidaner et al., 2005).

**Señales que reconoce el modelo**
"""
    )
    st.dataframe(pd.DataFrame([{"etiqueta": k, "señal": v[0], "categoría": v[1], "color": v[2]}
                               for k, v in config.SENALES.items()]), hide_index=True)
    with st.expander("Capturas de la aplicación"):
        galeria(config.CARPETA_CAPTURAS_APP, "Sin capturas de la app (`python vv.py capturas --origen <carpeta> --app`).")
    st.caption("Prototipo académico. No reemplaza la atención del conductor ni la señalización real.")
