<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://cmspreprod.lasalle.edu.co/sites/default/files/2026-04/Versi%C3%B3n%20en%20blanco.png">
    <img src="https://cmspreprod.lasalle.edu.co/sites/default/files/2026-04/Versi%C3%B3n%20en%20blanco.png" alt="Universidad de La Salle" width="320">
  </picture>
</p>

<h1 align="center">🚦 VisionVial</h1>
<p align="center"><b>Reconocimiento de señales de tránsito y corrección de color para personas con daltonismo</b></p>
<p align="center"><b>Visión por Computadora · Actividad 6: Proyecto Final · Maestría en Inteligencia Artificial</b></p>

<p align="center">
  <a href="https://vcdaltpf-production.up.railway.app/"><img src="https://img.shields.io/badge/Aplicaci%C3%B3n-en%20l%C3%ADnea-1BAF7A?style=for-the-badge&logo=railway&logoColor=white" alt="Aplicación en línea"></a>
  <img src="https://img.shields.io/badge/Python-3.12-003580?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Azure-Custom%20Vision-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white" alt="Azure Custom Vision">
  <img src="https://img.shields.io/badge/ONNX-Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white" alt="ONNX Runtime">
</p>

---

## 📌 Descripción

Proyecto final de Visión por Computadora (Maestría en Inteligencia Artificial, Universidad de La Salle).
Las personas con daltonismo pueden ver la **forma** de una señal de tránsito, pero no siempre su **categoría**,
que en Colombia se comunica por color: rojo para las reglamentarias, amarillo para las preventivas y naranja
para las de obras. VisionVial resuelve ese problema en dos frentes:

1. 🧠 **Clasificador de imágenes** de 10 señales de tránsito, entrenado y publicado en **Azure AI Custom Vision**
   (área de aplicación: reconocimiento de objetos). La app comunica la señal, su categoría y su color **en texto**.
2. 🎨 **Corrección de color (daltonización)** de la foto según el tipo de daltonismo del usuario (protanopía,
   deuteranopía o tritanopía), para recuperar las diferencias de color que la persona no percibe.

El repositorio contiene:

- ⌨️ **Línea de comandos** (`python vv.py`) para preparar el dataset, cargar el modelo entrenado y los
  pantallazos del entrenamiento, evaluar el modelo y publicar los resultados.
- 📱 **Aplicación web en Streamlit** con cámara, carga de lotes (ZIP), simulador de daltonismo, métricas del
  modelo y documentación del entrenamiento. Se publica en **Railway** y se actualiza con cada push.
- 📓 **Notebook** con el análisis descriptivo del dataset, la documentación del entrenamiento y la evaluación
  por el endpoint.

🔗 **Aplicación en línea:** https://vcdaltpf-production.up.railway.app/

👤 **Autores:** Brhayan Liberato Tafur y Diego Nicolás Ávila Moreno.

---

## 📊 Resultados

Evaluación de la iteración publicada (**Iteration3**, endpoint de Azure) sobre **500 imágenes de prueba
independientes** (50 por clase), que no se usaron en el entrenamiento:

| Métrica | Valor |
|---|---|
| Accuracy | **0,910** |
| F1-score macro | **0,912** |
| AUC macro (uno contra el resto) | **0,993** |

El detalle por clase, la matriz de confusión, las curvas ROC, el error de categoría, el umbral de confianza
recomendado y el experimento de daltonismo están en `resultados/` y en la pestaña **📊 Desempeño del modelo**
de la aplicación.

---

## 🗂️ Estructura

```
├── vv.py                        # CLI: dataset, modelo, capturas, evaluar, publicar, estado
├── app.py                       # aplicación Streamlit (lee modelos/, resultados/ y assets/)
├── visionvial/                  # paquete compartido por la CLI, la app y el notebook
│   ├── config.py                # rutas, clases (SENALES), categorías y credenciales
│   ├── dataset.py               # descarga GTSRB y arma data/dataset/{train,test}
│   ├── modelo.py                # registro y predicción: endpoint de Azure u ONNX exportado
│   ├── capturas.py              # importa los pantallazos del entrenamiento en Custom Vision
│   ├── evaluacion.py            # precision, recall, F1, matriz de confusión, ROC/AUC, umbral, experimento
│   └── daltonismo.py            # simulación (Machado 2009) y corrección (Fidaner 2005)
├── notebooks/
│   └── custom_vision_senales.ipynb   # descriptivos, entrenamiento documentado y evaluación por endpoint
├── modelos/actual/              # modelo registrado: model.onnx, labels.txt, info.json
├── resultados/                  # metricas.json, tablas (CSV) y gráficos (PNG) de la evaluación
├── assets/
│   ├── entrenamiento/           # pantallazos de Custom Vision + capturas.json
│   └── app/                     # pantallazos de la aplicación
├── tests/                       # pruebas y modelo ONNX sintético
├── data/                        # dataset local (no se versiona)
├── Dockerfile                   # imagen de la aplicación (Railway la usa automáticamente)
├── scripts/start.sh             # arranque de Streamlit en el puerto asignado ($PORT)
├── scripts/railway_desplegar.sh # despliegue con la CLI de Railway
├── .railway/railway.ts          # infraestructura como código de Railway (opcional)
├── requirements.txt             # dependencias de la CLI y el notebook
└── requirements-app.txt         # dependencias mínimas de la aplicación (imagen de Railway)
```

## ⚙️ 1. Instalación

> 💡 Se recomienda **Python 3.12** (el mismo de la imagen de Railway).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # completar endpoint y clave
```

> ⚠️ En Windows, las rutas con espacios van **entre comillas** y sin `~`:
> `--archivo "G:\Mi unidad\...\export_onnx.zip"`.

## 📥 2. Preparación del dataset

```bash
python vv.py dataset
```

Se usa el **GTSRB** (*German Traffic Sign Recognition Benchmark*): más de 50.000 recortes de señales en 43 clases,
tomados de video en condiciones reales. Se seleccionan las clases con equivalente en Colombia y se agrupan en
**10 etiquetas**:

| Etiqueta | Clases GTSRB | Categoría (Colombia) | Color |
|---|---|---|---|
| `velocidad_maxima` | 1, 2, 3, 4, 5, 7, 8 | Reglamentaria | 🔴 Rojo |
| `prohibido_adelantar` | 9 | Reglamentaria | 🔴 Rojo |
| `ceda_el_paso` | 13 | Reglamentaria | 🔴 Rojo |
| `pare` | 14 | Reglamentaria | 🔴 Rojo |
| `no_entre` | 17 | Reglamentaria | 🔴 Rojo |
| `curva_peligrosa` | 19, 20, 21 | Preventiva | 🟡 Amarillo |
| `semaforo_adelante` | 26 | Preventiva | 🟡 Amarillo |
| `peatones` | 27 | Preventiva | 🟡 Amarillo |
| `zona_escolar` | 28 | Preventiva | 🟡 Amarillo |
| `obras` | 25 | Transitoria | 🟠 Naranja |

Por cada etiqueta se toman **150 imágenes de entrenamiento** de la partición oficial de entrenamiento. GTSRB
guarda unas 30 tomas casi idénticas de cada señal física, por lo que la selección alterna entre ellas para evitar
duplicados. También se toman **50 imágenes de prueba** de la partición oficial de prueba, que no comparte señales
físicas con la de entrenamiento. Las imágenes se amplían a 256 px en el lado corto, como recomienda Custom Vision.

| Carpeta | Uso |
|---|---|
| `data/dataset/train/<etiqueta>/` | se **cargan en Custom Vision**, una carpeta por etiqueta |
| `data/dataset/test/<etiqueta>/` | prueba independiente: **no se cargan** |
| `data/dataset/manifest.csv` | inventario con el origen y el tamaño original de cada imagen |

Opciones: `--gtsrb-dir` (GTSRB ya descargado, p. ej. de Kaggle), `--extra <carpeta>` (fotos propias en
subcarpetas con el nombre de la etiqueta), `--n-train`, `--n-test`.

## ☁️ 3. Entrenamiento en Azure Custom Vision

El entrenamiento se hace en el portal de [Custom Vision](https://www.customvision.ai/):

1. Proyecto **Classification · Multiclass** con dominio **General (compact)**. Solo las iteraciones entrenadas con
   un dominio *compact* se pueden exportar.
2. Carga de cada carpeta de `data/dataset/train/` con su etiqueta.
3. Entrenamiento, revisión de las métricas de la iteración, **publicación** y **exportación** (*Export → ONNX*).
4. Pantallazos de cada paso, guardados en una carpeta en orden.

## 🧠 4. Carga del modelo y de los pantallazos

**Modelo entrenado** (para evaluarlo con métricas):

```bash
python vv.py modelo --archivo "<export_onnx.zip>" --iteracion Iteration3 --dominio "General (compact)"
python vv.py modelo --endpoint "<Prediction URL de la iteración publicada>"
```

El ONNX se registra en `modelos/actual/` con una prueba de carga. El endpoint se prueba con una imagen y se
guarda en `info.json`; la **Prediction-Key** queda solo en `.streamlit/secrets.toml`, que no se versiona.

**Pantallazos del entrenamiento:**

```bash
python vv.py capturas --origen "<carpeta>" --plantilla      # genera descripciones.csv para completar
python vv.py capturas --origen "<carpeta>"                  # ordena, renombra (Paso_01.png…) y crea el índice
python vv.py capturas --origen "<carpeta_app>" --app        # pantallazos de la aplicación
```

Para revisar qué hay y qué falta: `python vv.py estado`.

## 📏 5. Evaluación del modelo

```bash
python vv.py evaluar                      # con el ONNX exportado (local, sin costo por predicción)
python vv.py evaluar --backend endpoint   # con la API publicada (~1.700 llamadas, guardadas en caché)
```

Al terminar, la evaluación imprime un resumen:

```
RESULTADOS
  Accuracy 0.910 · F1 macro 0.912 · AUC macro 0.993 · error de categoría …
  Umbral de confianza recomendado para la app: …
  Archivos en: resultados/
```

| Salida en `resultados/` | Contenido |
|---|---|
| `metricas.json` | resumen que lee la aplicación (métricas globales, por clase, umbral y experimento) |
| `reporte_clasificacion.csv`, `metricas_por_clase.png` | precision, recall, F1-score y soporte por clase |
| `matriz_confusion.csv` / `.png` | conteo y matriz normalizada por clase real |
| `auc_por_clase.csv`, `curvas_roc.png` | curvas ROC uno contra el resto, AUC por clase, micro y macro |
| `desempeno_por_categoria.csv` | accuracy y **error de categoría** (p. ej., una reglamentaria tomada por preventiva) |
| `umbral_confianza.csv` / `.png` | cobertura y accuracy por umbral; umbral recomendado para la app |
| `errores.png` | errores con mayor confianza |
| `resultados_experimento.csv`, `experimento_daltonismo.png` | experimento de daltonismo (ver abajo) |

**🔬 Experimento de daltonismo.** El modelo se usa como un observador estandarizado: clasifica la imagen
**original**, la imagen **como la percibe** una persona con cada tipo de daltonismo y la imagen **corregida**
por la aplicación tal como la percibe esa misma persona. Si la corrección funciona, el desempeño con corrección
se acerca más al original que el desempeño sin corrección.

## 🎨 6. Método de corrección de color

Para cada tipo de daltonismo, el módulo `visionvial/daltonismo.py`:

1. **Simula** cómo percibe la imagen la persona, con las matrices de Machado, Oliveira y Fernandes (2009) en RGB
   lineal y una severidad de 0 a 1.
2. **Calcula la información perdida**: original − simulada.
3. **Redistribuye** esa pérdida hacia los canales que la persona sí percibe (Fidaner, Lin y Ozguven, 2005) y la
   suma a la imagen original.

| Tipo | Colores que confunde | Efecto de la corrección |
|---|---|---|
| Protanopía (conos L) | rojo ↔ verde | los rojos se desplazan hacia rosado/magenta y los verdes se oscurecen |
| Deuteranopía (conos M) | rojo ↔ verde | los rojos tiran a magenta y los verdes se vuelven más puros |
| Tritanopía (conos S) | azul ↔ verde, amarillo ↔ rosado | los azules y amarillos cambian de tono para separarse |

Ejemplo: con deuteranopía, la diferencia de color percibida entre un rojo y un verde de señal pasa de un ΔE de
10,8 sin corrección a 52 con corrección, casi cinco veces mayor. Los blancos, negros y grises no cambian.

## 📱 7. Aplicación

```bash
pip install -r requirements-app.txt
streamlit run app.py                 # http://localhost:8501
```

| Pestaña | Contenido | Fuente |
|---|---|---|
| 📷 Foto | cámara o archivo: señal, categoría y color en texto, imagen corregida y simulación | endpoint de Azure u ONNX |
| 🗂️ Lote / dataset | varias imágenes o un ZIP con carpetas por etiqueta: accuracy, matriz de confusión y CSV | endpoint de Azure u ONNX |
| 👁️ Simulador | la imagen con cada tipo de daltonismo, sin y con corrección | `daltonismo.py` |
| 📊 Desempeño del modelo | métricas, gráficos y experimento de daltonismo | `resultados/` |
| 🛠️ Entrenamiento en ACV | datos de la iteración y pantallazos del entrenamiento | `assets/entrenamiento/` |
| ℹ️ Acerca de | funcionamiento, señales reconocidas y pantallazos de la app | `config.py`, `assets/app/` |

En la barra lateral el usuario elige su **tipo de daltonismo**, la **severidad**, la **intensidad de la corrección**
y el **umbral de confianza**, que arranca en el valor recomendado por la evaluación. También elige el **motor de
predicción**: el endpoint de Azure (la API publicada) o el ONNX local.

## 🚀 8. Publicación en Railway

🔗 **En línea:** https://vcdaltpf-production.up.railway.app/

La aplicación lee del repositorio el modelo, las métricas y los pantallazos, así que se actualiza con cada push:

```bash
python vv.py evaluar
python vv.py publicar        # commit de modelos/, resultados/ y assets/ + git push → Railway redespliega
```

Configuración del servicio en Railway:

| Elemento | Valor |
|---|---|
| Construcción | `Dockerfile` de la raíz (Railway lo detecta automáticamente) |
| Arranque | `scripts/start.sh`; no se define *Start Command* |
| Variables | `CV_PREDICTION_ENDPOINT`, `CV_PREDICTION_KEY` y `PORT=8080` |
| Dominio | *Settings → Networking → Generate Domain*, puerto `8080` |
| Health check | `/_stcore/health` |

Recomendaciones:

- 🔒 **Credenciales:** la Prediction-Key solo va en `.streamlit/secrets.toml` (local) y en las variables de Railway;
  nunca en el código ni en el notebook. Conviene regenerarla en Azure después de la entrega.
- 💾 **Memoria:** la imagen instala solo `requirements-app.txt` (sin scikit-learn ni matplotlib). Aun así, revise la
  pestaña *Metrics* del servicio, porque la app procesa fotos de celular.
- 🧩 **Config as Code:** Railway marcó `railway.toml`/`railway.json` como obsoletos; por eso la configuración está en
  el `Dockerfile` y, de forma opcional, en `.railway/railway.ts`.

Alternativa desde la terminal: `sh scripts/railway_desplegar.sh` (CLI de Railway).

## 🧪 9. Pruebas

```bash
python -m pytest tests -q
```

Verifican que la corrección aumente el contraste rojo-verde percibido, que la severidad 0 no altere la imagen y que
el clasificador ONNX funcione con un modelo sintético que tiene la misma interfaz que el exportado por Custom Vision.
Para probar el flujo sin el modelo real:

```bash
python tests/onnx_sintetico.py /tmp/modelo_sintetico.zip
python vv.py modelo --archivo /tmp/modelo_sintetico.zip --iteracion prueba
```

## 📝 Notas y limitaciones

- **Señalización alemana vs. colombiana:** las reglamentarias de GTSRB son prácticamente iguales a las colombianas
  (Convención de Viena), pero las preventivas alemanas son triángulos blancos con borde rojo y en Colombia son
  rombos amarillos. La opción `--extra` permite sumar fotos propias para acercar el dataset a la señalización local.
- **Clasificación, no detección:** el modelo clasifica la imagen completa, por lo que funciona mejor con la señal
  centrada y encuadrada. La detección en la escena completa queda como trabajo futuro.
- **Dominio compact:** se eligió para poder exportar el modelo a ONNX. Estos dominios están optimizados para
  dispositivos y pueden ser algo menos precisos que *General [A2]*.
- **Experimento de daltonismo:** el modelo no es una persona; reconoce principalmente la forma. Por eso el
  experimento se complementa con la pérdida de color (ΔE) del notebook. La simulación es un modelo matemático de la
  dicromacia, no una medición con usuarios.
- **Prototipo académico:** no reemplaza la atención del conductor ni la señalización real.

## 📚 Referencias

- Fidaner, O., Lin, P. y Ozguven, N. (2005). *Analysis of Color Blindness*. Stanford University.
- Machado, G. M., Oliveira, M. M. y Fernandes, L. A. F. (2009). A physiologically-based model for simulation of color
  vision deficiency. *IEEE Transactions on Visualization and Computer Graphics*, 15(6), 1291–1298.
- Microsoft. *Azure AI Custom Vision documentation*. https://learn.microsoft.com/azure/ai-services/custom-vision-service/
- Ministerio de Transporte de Colombia. *Manual de Señalización Vial*.
- Stallkamp, J., Schlipsing, M., Salmen, J. e Igel, C. (2012). Man vs. computer: Benchmarking machine learning
  algorithms for traffic sign recognition. *Neural Networks*, 32, 323–332.
