import sys

# sys.path.insert(0, ".")

from PyQt5.QtWidgets import QApplication
from controller.RestoreImageController import RestoreImageController
from models.RestoreImageModel import RestoreImageModel
from views.RestoreImage import RestoreImage


if __name__ == '__main__':
    print('[start app...]')
    app = QApplication(sys.argv)

    model = RestoreImageModel()
    controller = RestoreImageController(model)

    window = RestoreImage(controller, model)
    window.show()
    sys.exit(app.exec())
