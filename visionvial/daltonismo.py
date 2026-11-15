"""
daltonismo.py
-------------
Simulacion y correccion (daltonizacion) de imagenes para los tres tipos de
dicromacia. Lo usan tanto el notebook como la aplicacion de Streamlit.

- Simulacion: matrices de Machado, Oliveira y Fernandes (2009), aplicadas en RGB lineal.
  La severidad (0 a 1) se aproxima interpolando entre la identidad y la matriz de severidad 1.
- Correccion: metodo de Fidaner, Lin y Ozguven (2005). Se calcula la informacion de color
  que el usuario pierde (original - simulada) y se redistribuye hacia los canales que si percibe.
- Delta E (CIE76) en espacio Lab para medir cuanto color se pierde en la simulacion.
"""
import numpy as np

TIPOS = {
    "protanopia": "Protanopía (rojo)",
    "deuteranopia": "Deuteranopía (verde)",
    "tritanopia": "Tritanopía (azul)",
}

# Machado et al. (2009), severidad 1.0, RGB lineal
_MACHADO = {
    "protanopia": np.array([[0.152286, 1.052583, -0.204868],
                            [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968],
                              [0.280085, 0.672501, 0.047413],
                              [-0.011820, 0.042940, 0.968881]]),
    "tritanopia": np.array([[1.255528, -0.076749, -0.178779],
                            [-0.078411, 0.930809, 0.147602],
                            [0.004733, 0.691367, 0.303900]]),
}

# Fidaner et al. (2005): como se reparte el error hacia los canales visibles
_CORRECCION = {
    "protanopia": np.array([[0.0, 0.0, 0.0], [0.7, 1.0, 0.0], [0.7, 0.0, 1.0]]),
    "deuteranopia": np.array([[0.0, 0.0, 0.0], [0.7, 1.0, 0.0], [0.7, 0.0, 1.0]]),
    "tritanopia": np.array([[1.0, 0.0, 0.7], [0.0, 1.0, 0.7], [0.0, 0.0, 0.0]]),
}


# ------------------------------------------------------------------ conversiones sRGB
def a_lineal(rgb_uint8):
    """sRGB 0-255 -> RGB lineal 0-1."""
    c = np.asarray(rgb_uint8, dtype=np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def a_srgb(lineal):
    """RGB lineal 0-1 -> sRGB 0-255 (uint8)."""
    c = np.clip(lineal, 0.0, 1.0)
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)
    return np.clip(np.round(c * 255.0), 0, 255).astype(np.uint8)


def _aplicar(matriz, arr):
    return arr @ matriz.T


def _matriz_simulacion(tipo, severidad):
    if tipo not in _MACHADO:
        raise ValueError(f"Tipo desconocido: {tipo}. Use uno de {list(TIPOS)}")
    s = float(np.clip(severidad, 0.0, 1.0))
    return (1.0 - s) * np.eye(3) + s * _MACHADO[tipo]


# ------------------------------------------------------------------ API publica
def simular(rgb_uint8, tipo, severidad=1.0):
    """Como percibe la imagen una persona con el tipo de daltonismo indicado."""
    lin = a_lineal(rgb_uint8)
    return a_srgb(_aplicar(_matriz_simulacion(tipo, severidad), lin))


def corregir(rgb_uint8, tipo, severidad=1.0, intensidad=1.0):
    """Daltoniza la imagen: redistribuye el color perdido hacia los canales visibles.

    intensidad escala la compensacion (0 = sin cambio, 1 = metodo original).
    """
    lin = a_lineal(rgb_uint8)
    sim = _aplicar(_matriz_simulacion(tipo, severidad), lin)
    error = lin - sim
    compensacion = _aplicar(_CORRECCION[tipo], error)
    return a_srgb(lin + float(intensidad) * compensacion)


def a_lab(rgb_uint8):
    """sRGB 0-255 -> CIELAB (iluminante D65)."""
    lin = a_lineal(rgb_uint8)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = _aplicar(m, lin) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def delta_e(rgb_a, rgb_b):
    """Diferencia de color CIE76 por pixel (aprox. 2.3 = diferencia apenas perceptible)."""
    return np.linalg.norm(a_lab(rgb_a) - a_lab(rgb_b), axis=-1)
