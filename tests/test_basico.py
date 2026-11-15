"""Pruebas rápidas: python -m pytest tests -q"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from visionvial import config, daltonismo as dal  # noqa: E402


def test_correccion_aumenta_contraste_rojo_verde():
    img = np.array([[[220, 30, 30], [30, 160, 40]]], np.uint8)
    for tipo in ("protanopia", "deuteranopia"):
        visto = dal.simular(img, tipo)
        visto_corr = dal.simular(dal.corregir(img, tipo), tipo)
        antes = dal.delta_e(visto[:, :1], visto[:, 1:])[0, 0]
        despues = dal.delta_e(visto_corr[:, :1], visto_corr[:, 1:])[0, 0]
        assert despues > antes


def test_severidad_cero_no_cambia():
    img = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    assert np.array_equal(dal.simular(img, "deuteranopia", 0.0), img)


def test_clasificador_onnx_sintetico():
    import zipfile
    from tests.onnx_sintetico import escribir_zip
    from visionvial.modelo import ClasificadorONNX
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        escribir_zip(tmp / "m.zip")
        with zipfile.ZipFile(tmp / "m.zip") as z:
            z.extractall(tmp)
        clf = ClasificadorONNX(tmp)
        pred = clf.predecir(np.full((300, 400, 3), 128, np.uint8))
        assert set(pred["etiqueta"]) == set(config.SENALES)
        assert abs(pred["probabilidad"].sum() - 1) < 1e-4
