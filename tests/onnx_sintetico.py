"""
Genera un modelo ONNX pequeño con la misma interfaz que exporta Custom Vision
(entrada `data` [1,3,224,224] BGR 0-255, salida de probabilidades en el orden de labels.txt)
para probar el flujo sin tener aún el modelo real. No necesita el paquete `onnx`: escribe el protobuf a mano.

Uso:  python tests/onnx_sintetico.py salida.zip
"""
import struct
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from visionvial import config  # noqa: E402


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _campo(num, valor):
    if isinstance(valor, int):
        return _varint(num << 3 | 0) + _varint(valor)
    if isinstance(valor, str):
        valor = valor.encode()
    return _varint(num << 3 | 2) + _varint(len(valor)) + valor


def _tensor(nombre, arr):
    arr = np.asarray(arr, dtype=np.float32)
    return b"".join([*(_campo(1, int(d)) for d in arr.shape), _campo(2, 1), _campo(8, nombre),
                     _campo(9, arr.tobytes())])


def _valor(nombre, forma):
    dims = b"".join(_campo(1, _campo(1, d) if isinstance(d, int) else _campo(2, d)) for d in forma)
    tensor = _campo(1, 1) + _campo(2, dims)
    return _campo(1, nombre) + _campo(2, _campo(1, tensor))


def _nodo(op, entradas, salidas, atributos=b""):
    return b"".join([*(_campo(1, e) for e in entradas), *(_campo(2, s) for s in salidas),
                     _campo(3, f"{op}_{salidas[0]}"), _campo(4, op), atributos])


def _atributo_ints(nombre, valores):
    return _campo(5, _campo(1, nombre) + _campo(20, 7) + b"".join(_campo(8, v) for v in valores))


def _atributo_int(nombre, v):
    return _campo(5, _campo(1, nombre) + _campo(20, 2) + _campo(3, v))


def modelo(etiquetas, semilla=0):
    rng = np.random.default_rng(semilla)
    k = len(etiquetas)
    W = rng.normal(0, 0.05, size=(3, k))
    b = rng.normal(0, 0.5, size=(k,))
    nodos = [
        _nodo("ReduceMean", ["data"], ["media"], _atributo_ints("axes", [2, 3]) + _atributo_int("keepdims", 0)),
        _nodo("MatMul", ["media", "W"], ["logits0"]),
        _nodo("Add", ["logits0", "b"], ["logits"]),
        _nodo("Softmax", ["logits"], ["model_outputs0"], _atributo_int("axis", 1)),
    ]
    grafo = b"".join([*(_campo(1, n) for n in nodos), _campo(2, "visionvial_sintetico"),
                      _campo(5, _tensor("W", W)), _campo(5, _tensor("b", b)),
                      _campo(11, _valor("data", [1, 3, 224, 224])), _campo(12, _valor("model_outputs0", [1, k]))])
    return (_campo(1, 7) + _campo(2, "onnx_sintetico") + _campo(7, grafo)
            + _campo(8, _campo(1, "") + _campo(2, 13)))


def escribir_zip(ruta, etiquetas=None):
    etiquetas = etiquetas or sorted(config.SENALES)
    with zipfile.ZipFile(ruta, "w") as z:
        z.writestr("model.onnx", modelo(etiquetas))
        z.writestr("labels.txt", "\n".join(etiquetas) + "\n")
        z.writestr("metadata_properties.json", '{"CustomVision.Metadata.Version": "sintetico"}')
    return ruta


if __name__ == "__main__":
    salida = sys.argv[1] if len(sys.argv) > 1 else "modelo_sintetico.zip"
    print(escribir_zip(salida, sys.argv[2].split(",") if len(sys.argv) > 2 else None))
