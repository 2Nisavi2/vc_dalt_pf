# 🚦 VisionVial

**Reconocimiento de señales de tránsito y corrección de color para personas con daltonismo.**
Proyecto final de Visión por Computadora (Maestría en Inteligencia Artificial): un clasificador de 10 señales
entrenado y publicado en **Azure AI Custom Vision**, integrado en una aplicación web con cámara (**Streamlit**)
que además corrige los colores de la imagen según el tipo de daltonismo del usuario.

## Estructura

```
visionvial/
├── vv.py                      Línea de comandos: dataset, modelo, capturas, evaluar, publicar, estado
├── app.py                     Aplicación Streamlit (lee modelos/, resultados/ y assets/)
├── visionvial/                Paquete compartido por la CLI, la app y el notebook
│   ├── config.py              Rutas, clases (SENALES) y credenciales
│   ├── dataset.py             Descarga GTSRB y arma data/dataset/{train,test}
│   ├── modelo.py              Registro del modelo y predicción (endpoint de Azure u ONNX exportado)
│   ├── capturas.py            Importa los pantallazos del entrenamiento en Custom Vision
│   ├── evaluacion.py          Precision, recall, F1, matriz de confusión, ROC/AUC, umbral y experimento
│   └── daltonismo.py          Simulación (Machado 2009) y corrección (Fidaner 2005)
├── notebooks/custom_vision_senales.ipynb   Descriptivos, documentación y evaluación por endpoint
├── modelos/actual/            Modelo registrado: model.onnx, labels.txt, info.json        (se versiona)
├── resultados/                metricas.json, CSV y PNG de la evaluación                   (se versiona)
├── assets/entrenamiento/      Pantallazos de Custom Vision + capturas.json                (se versiona)
├── assets/app/                Pantallazos de la aplicación                                (se versiona)
├── data/                      Dataset local (no se versiona)
└── tests/                     Modelo ONNX sintético y pruebas del flujo
```

## Flujo de trabajo

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml     # completar endpoint y clave
```

### 1. Descargar y preparar las imágenes

```bash
python vv.py dataset
```

Descarga GTSRB (~300 MB), toma 150 imágenes por etiqueta para entrenamiento y 50 para prueba, y deja:

- `data/dataset/train/<etiqueta>/`: **se suben a Custom Vision**, una carpeta por etiqueta.
- `data/dataset/test/<etiqueta>/`: prueba independiente, **no se suben**.

Opciones: `--gtsrb-dir` (GTSRB ya descargado, p. ej. de Kaggle), `--extra fotos_bogota` (fotos propias en
subcarpetas con el nombre de la etiqueta), `--n-train`, `--n-test`.

### 2. Entrenar en Azure Custom Vision (manual)

1. Proyecto **Classification · Multiclass**, dominio **General (compact)** para poder exportar.
2. Cargar cada carpeta de `data/dataset/train/` con su etiqueta.
3. Entrenar, revisar la iteración, **publicarla** y **exportarla** (Export → **ONNX**).
4. Tomar pantallazos de cada paso y guardarlos en una carpeta (en orden).

### 3. Cargar los resultados del entrenamiento al repositorio

**a) El modelo entrenado** (para evaluarlo con métricas):

```bash
python vv.py modelo --archivo ~/Descargas/export_onnx.zip --iteracion Iteration2 --dominio "General (compact)"
python vv.py modelo --endpoint "https://<region>.api.cognitive.microsoft.com/customvision/v3.0/Prediction/<id>/classify/iterations/<iteracion>/image"
```

El zip exportado se registra en `modelos/actual/`, con una prueba de carga. El endpoint se registra en
`info.json`; la clave queda solo en `.streamlit/secrets.toml`.

**b) Los pantallazos del entrenamiento:**

```bash
python vv.py capturas --origen ~/pantallazos_acv --plantilla   # genera descripciones.csv para completar
python vv.py capturas --origen ~/pantallazos_acv               # copia, ordena, renombra y crea el índice
python vv.py capturas --origen ~/pantallazos_app --app         # (opcional) pantallazos de la app
```

### 4. Evaluar el modelo

```bash
python vv.py evaluar                      # con el ONNX exportado (local, sin costo)
python vv.py evaluar --backend endpoint   # con la API publicada (~1.700 llamadas, quedan en caché)
```

Escribe en `resultados/`: `metricas.json`, reporte de clasificación (precision, recall, F1), matriz de confusión,
curvas ROC y AUC, desempeño por categoría, umbral de confianza recomendado, galería de errores y el
experimento de daltonismo (imagen como la percibe la persona, sin y con corrección).

### 5. Actualizar la aplicación

```bash
python vv.py estado      # qué hay y qué falta
python vv.py publicar    # commit de modelos/, resultados/ y assets/ + git push
```

Streamlit Community Cloud redespliega la app con cada push. La app lee directamente:

| Pestaña | Fuente |
| --- | --- |
| 📷 Foto, 🗂️ Lote | Endpoint de Azure (si hay secretos) o `modelos/actual/model.onnx` |
| 📊 Desempeño del modelo | `resultados/metricas.json` y gráficos |
| 🛠️ Entrenamiento en ACV | `assets/entrenamiento/capturas.json` |
| Barra lateral | Umbral por defecto = `umbral_recomendado` de la evaluación |

## Publicar la app (una sola vez)

1. Subir el repositorio a GitHub (`git remote add origin <url> && git push -u origin main`).
2. En [share.streamlit.io](https://share.streamlit.io) crear la app con el archivo principal `app.py`
   (Python 3.11 o superior).
3. En **Settings → Secrets** pegar `CV_PREDICTION_ENDPOINT` y `CV_PREDICTION_KEY`.
4. Abrir `https://<app>.streamlit.app` en el celular (HTTPS habilita la cámara). En el celular, **Subir archivo**
   abre la cámara trasera nativa.

## Ejecutar en local

```bash
streamlit run app.py
```

## Probar el flujo sin el modelo real

```bash
python tests/onnx_sintetico.py /tmp/modelo_sintetico.zip
python vv.py modelo --archivo /tmp/modelo_sintetico.zip --iteracion prueba
python vv.py evaluar --sin-experimento
```

El modelo sintético tiene la misma interfaz que el exportado por Custom Vision, pero sus predicciones son
aleatorias: sirve solo para verificar el flujo. Vuelva a registrar el modelo real antes de publicar.

## Referencias

- Fidaner, O., Lin, P. y Ozguven, N. (2005). *Analysis of Color Blindness*. Stanford University.
- Machado, G. M., Oliveira, M. M. y Fernandes, L. A. F. (2009). A physiologically-based model for simulation of color
  vision deficiency. *IEEE TVCG*, 15(6), 1291–1298.
- Stallkamp, J., Schlipsing, M., Salmen, J. e Igel, C. (2012). Man vs. computer: Benchmarking machine learning
  algorithms for traffic sign recognition. *Neural Networks*, 32, 323–332.
