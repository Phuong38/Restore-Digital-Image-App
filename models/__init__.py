import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from cv2 import cv2


def numpy_to_qimage(image: np.ndarray, shape: tuple, is_rgb: bool):
    """
    Convert numpy image to QImage
    :param image: Image as numpy array
    :param shape: QImage shape
    :param is_rgb:
    :return:
    """
    if not is_rgb:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image.copy()
    h, w, ch = image_rgb.shape
    bytes_per_line = ch * w
    convert_to_qt_format = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return convert_to_qt_format.scaled(shape[0], shape[1], Qt.KeepAspectRatio)
