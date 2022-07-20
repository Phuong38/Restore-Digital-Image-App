import os

from functools import partial
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import pyqtSignal, pyqtSlot

from views.RestoreImageGUI import Ui_RestoreImage
from controller import RestoreImageController
from models import RestoreImageModel


class RestoreImage(QMainWindow, Ui_RestoreImage):
    resized = pyqtSignal()

    def __init__(self, controller: RestoreImageController, model: RestoreImageModel, parent=None):
        super().__init__(parent)
        self.had_input = False
        self.setupUi(self)
        self.controller = controller
        self.model = model

        # self.setup_layout_handler()
        self.setup_button_actions()
        self.setup_pixels_map()
        self.setup_combobox_actions()
        self.setup_set_score()

    def setup_button_actions(self):
        self.pb_browse.clicked.connect(self.browse_file)
        self.pb_inject_noise.clicked.connect(self.inject_noise_to_image)
        self.pb_denoise_by_model.clicked.connect(self.denoise)
        self.pb_colorize.clicked.connect(self.colorize)
        self.pb_denoise_by_tv.clicked.connect(self.denoise_by_tv)

        self.pb_start_3.clicked.connect(self.run_process)

    def setup_pixels_map(self):
        self.model.input_pixmap.connect(self.set_input_frame)
        self.model.noise_pixmap.connect(self.set_preview_frame)
        self.model.denoised_pixmap.connect(self.set_denoised_frame)
        self.model.denoised_by_tv_pixmap.connect(self.set_denoised_by_tv_frame)
        self.model.gray_pixmap.connect(self.set_gray_frame)
        self.model.colorized_pixmap.connect(self.set_colorized_frame)

    def setup_set_score(self):
        self.model.set_score.connect(self.set_score)

    def browse_file(self):
        file_path = QFileDialog.getOpenFileName(self, 'Open file', os.getcwd(), options=QFileDialog.DontUseNativeDialog)
        self.le_input_path.setText(file_path[0])

        input_source = self.le_input_path.text()
        preview_size = (self.view_input.size().width(), self.view_input.size().height(),)
        self.controller.set_input(input_source, preview_size)
        self.had_input = True

    def inject_noise_to_image(self):
        source_image = self.le_input_path.text()
        noise_image_size = (self.view_noise_image.size().width(), self.view_noise_image.size().height(),)
        self.controller.set_noise_image(source_image, noise_image_size)

    def denoise(self):
        denoise_image_size = (self.view_denoise_image.size().width(), self.view_denoise_image.size().height(),)
        self.controller.set_denoise_image(denoise_image_size)

    def colorize(self):
        colorized_image_size = (self.view_output.size().width(), self.view_output.size().height(),)
        self.controller.set_colorized_image(colorized_image_size)

    def denoise_by_tv(self):
        denoised_by_tv_image_size = (self.view_denoise_by_TV_image.size().width(),
                                     self.view_denoise_by_TV_image.size().height(),)
        self.controller.set_denoised_by_tv_image(denoised_by_tv_image_size)

    def setup_combobox_actions(self):
        combobox = self.s400_style_2
        combobox.currentTextChanged.connect(partial(self.action_update_combobox_option, combobox))
        self.action_update_combobox_option(combobox)

    def action_update_combobox_option(self, conbobox: QtWidgets.QComboBox):
        self.controller.set_sigma(conbobox.currentText())

    def run_process(self):
        if self.le_input_path.text() == "":
            QMessageBox.about(self, "Input not found!", "Please fill input source")
        else:
            input_source = self.le_input_path.text()
            preview_size = (self.view_input.size().width(), self.view_input.size().height(),)
            # if not self.had_input:
            self.controller.set_input(input_source, preview_size)

            self.inject_noise_to_image()
            self.denoise_by_tv()
            self.denoise()
            self.colorize()

    @pyqtSlot(QImage)
    def set_input_frame(self, image):
        self.view_input.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(QImage)
    def set_preview_frame(self, image):
        self.view_noise_image.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(QImage)
    def set_denoised_frame(self, image):
        self.view_denoise_image.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(QImage)
    def set_denoised_by_tv_frame(self, image):
        self.view_denoise_by_TV_image.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(QImage)
    def set_gray_frame(self, image):
        self.view_gray_image.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(QImage)
    def set_colorized_frame(self, image):
        self.view_output.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(str, float)
    def set_score(self, name: str, score: float):
        if name == 'colorized_model':
            self.output_psnr_score.setText(str(score))

        elif name == 'psnr_model':
            self.model_psnr_score.setText(str(score))

        elif name == 'psnr_tv':
            self.tv_psnr_score.setText(str(score))

        elif name == 'ssim_colorized_model':
            self.output_ssim_score.setText(str(score))

        elif name == 'ssim_model':
            self.model_ssim_score.setText(str(score))

        elif name == 'ssim_tv':
            self.tv_ssim_score.setText(str(score))
