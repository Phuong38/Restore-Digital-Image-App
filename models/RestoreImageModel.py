import datetime
import math
import os

import cv2
import numpy as np
import torch
import torchvision.utils
from PIL import Image
from PyQt5.QtCore import pyqtSignal, QObject, QThread
from PyQt5.QtGui import QImage

import skimage
from skimage import metrics
from skimage.restoration import denoise_tv_chambolle

from models import numpy_to_qimage
from models.restoration.restore_image import RestoreImage
from models.colorizers.colorizers_image import ColorizerImage


def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def synthesize_gaussian(image, std):
    ## Give PIL, return the noisy PIL

    img_pil = pil_to_np(image)

    mean = 0
    std = std / 255
    gauss = np.random.normal(loc=mean, scale=std, size=img_pil.shape)
    noisy = img_pil + gauss
    noisy = np.clip(noisy, 0, 1).astype(np.float32)

    return np_to_pil(noisy)


def Representational(r, g, b):
    return 0.299 * r + 0.287 * g + 0.114 * b


def calculate(img):
    b, g, r = cv2.split(img)
    return Representational(r, g, b)


def calculate_psnr_score(orignal_path, compressed_path):
    orignal_img = cv2.imread(orignal_path)
    compressed_img = cv2.imread(compressed_path)
    # print('[PSNR]', orignal_path, orignal_img)
    height, width = orignal_img.shape[:2]

    orignalPixelAt = calculate(orignal_img)
    compressedPixelAt = calculate(compressed_img)

    diff = orignalPixelAt - compressedPixelAt
    error = np.sum(np.abs(diff) ** 2)

    error = error / (height * width)

    return -(10 * math.log10(error / (255 * 255)))


def calculate_ssim_score(reference_image_path: str, generated_image_path: str):
    reference_image = skimage.io.imread(reference_image_path) / 255
    generated_image = skimage.io.imread(generated_image_path) / 255
    return metrics.structural_similarity(reference_image, generated_image, multichannel=True, gaussian_weights=True,
                                         use_sample_covariance=False)



class RestoreImageModel(QObject):
    input_pixmap = pyqtSignal(QImage)
    noise_pixmap = pyqtSignal(QImage)
    gray_pixmap = pyqtSignal(QImage)
    denoised_pixmap = pyqtSignal(QImage)
    denoised_by_tv_pixmap = pyqtSignal(QImage)
    colorized_pixmap = pyqtSignal(QImage)

    set_score = pyqtSignal(str, float)

    def __init__(self):
        super().__init__()
        # self.settings = Settings("app/assets/configs/style_transfer_configs.yaml")

        # Render options and input source

        self.image_name = None
        self.denoised_image = None
        self.image = None
        self.sigma = 10
        self.noise_image = None
        self.options = {}
        self.input_source = None
        self.original_image_path = 'assets/original_image'
        self.denoised_img_by_model_path = 'assets/output_image/by_model'
        self.denoised_img_by_tv_path = 'assets/output_image/by_tv'
        self.colorized_image_path = 'assets/output_image/colorized'
        self.noise_image_path = 'assets/output_image/noise_image'
        self.gray_image_path = 'assets/output_image/gray_image'
        self.preview_size = tuple()
        self.denoised_image_size = tuple()
        self.denoised_by_tv_image_size = tuple()
        self.colorized_image_size = tuple()

        self.input_image = None
        self.current_preview_frame = None

        # Init deeplearning models
        # Restore Image
        self.restore_image = RestoreImage()
        self.colorizer_image = ColorizerImage()

    def set_preview_size(self, preview_size: tuple):
        self.preview_size = preview_size

    def set_denoised_image_size(self, denoised_image_size: tuple):
        self.denoised_image_size = denoised_image_size

    def set_colorized_image_size(self, colorized_image_size: tuple):
        self.colorized_image_size = colorized_image_size

    def set_denoised_by_tv_image_size(self, denoised_by_tv_image_size: tuple):
        self.denoised_by_tv_image_size = denoised_by_tv_image_size

    def set_input_source(self, input_source: str):
        """
        Set input image and show on GUI
        :param input_source:
        :return:
        """

        self.original_image_path = 'assets/original_image'
        self.denoised_img_by_model_path = 'assets/output_image/by_model'
        self.denoised_img_by_tv_path = 'assets/output_image/by_tv'
        self.colorized_image_path = 'assets/output_image/colorized'
        self.noise_image_path = 'assets/output_image/noise_image'
        self.gray_image_path = 'assets/output_image/gray_image'

        try:
            self.input_source = input_source
            self.image_name = input_source.split('/')[-1]

            self.original_image_path = os.path.join(self.original_image_path, self.image_name)
            self.denoised_img_by_model_path = os.path.join(self.denoised_img_by_model_path, self.image_name)
            self.denoised_img_by_tv_path = os.path.join(self.denoised_img_by_tv_path, self.image_name)
            self.colorized_image_path = os.path.join(self.colorized_image_path, self.image_name)
            self.noise_image_path = os.path.join(self.noise_image_path, self.image_name)
            # print('[GRAY Image1]', self.gray_image_path)
            self.gray_image_path = os.path.join(self.gray_image_path, self.image_name)
            # print('[GRAY Image]', self.gray_image_path)

            # Read image
            self.input_image = cv2.imread(self.input_source)

            # Save input image
            cv2.imwrite(self.original_image_path, self.input_image)

            # Set input pixelmap
            self.input_pixmap.emit(
                numpy_to_qimage(image=self.input_image, shape=self.preview_size, is_rgb=False))
        except Exception as e:
            print(f"[RestoreImageModel][set_input_source] Error: {e}")

    def inject_noise(self, input_image, sigma):
        mean = 0
        sigma = sigma

        row, col = input_image.shape
        gaussian = np.random.normal(mean, sigma, (row, col))  # np.zeros((224, 224), np.float32)

        noisy_image = np.zeros(input_image.shape, np.float32)

        if len(input_image.shape) == 2:
            noisy_image = input_image + gaussian
        else:
            noisy_image[:, :, 0] = input_image[:, :, 0] + gaussian
            noisy_image[:, :, 1] = input_image[:, :, 1] + gaussian
            noisy_image[:, :, 2] = input_image[:, :, 2] + gaussian

        cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
        noisy_image = noisy_image.astype(np.uint8)
        return noisy_image

    def set_noise_image(self, source_image: str):
        """
        Inject noise to input image and show on GUI
        :param source_image:
        :return:
        """
        try:

            sigma = self.sigma
            print('sigma: ', sigma)
            gray = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2GRAY)
            self.noise_image = self.inject_noise(gray, sigma=sigma)

            self.gray_pixmap.emit(numpy_to_qimage(image=gray, shape=self.preview_size, is_rgb=False))
            self.noise_pixmap.emit(numpy_to_qimage(image=self.noise_image, shape=self.preview_size, is_rgb=False))

            cv2.imwrite(self.gray_image_path, gray)
            # print('[SAVE GRAY]', self.gray_image_path)
            cv2.imwrite(self.noise_image_path, self.noise_image)
        except Exception as e:
            print(f"[RestoreImageModel][set_noise_image] Error: {e}")

    def set_denoised_image(self):
        """
        denoise image and show on GUI
        :return:
        """
        _noise_image = self.noise_image
        self.denoised_image = self.restore_image.restore_image(_noise_image)
        # print(self.denoised_image.size())
        image_numpy = torchvision.utils.make_grid((self.denoised_image.data.cpu() + 1.0) / 2.0).mul(255).add_(
            0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        # print(image_numpy.shape, image_numpy.dtype)

        gray_image = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(self.denoised_img_by_model_path, gray_image)

        self.denoised_pixmap.emit(
            numpy_to_qimage(image=gray_image, shape=self.denoised_image_size, is_rgb=False))

        psnr_score = calculate_psnr_score(self.gray_image_path, self.denoised_img_by_model_path)
        ssim_score = calculate_ssim_score(self.gray_image_path, self.denoised_img_by_model_path)

        print('[PSNR Model]', round(psnr_score, 2))
        print('[SSIM Model]', round(ssim_score, 2))

        self.set_score.emit('psnr_model', round(psnr_score, 2))
        self.set_score.emit('ssim_model', round(ssim_score, 2))

    def set_denoised_by_tv_image(self):
        _noise_image = self.noise_image
        # img = img_as_float(_noise_image)

        denoise_img = denoise_tv_chambolle(_noise_image, weight=0.1, eps=0.0002, n_iter_max=200, multichannel=False)
        # print(type(denoise_img))
        denoise_img = denoise_img * 255
        denoise_img = denoise_img.astype(np.uint8)

        cv2.imwrite(self.denoised_img_by_tv_path, denoise_img)

        self.denoised_by_tv_pixmap.emit(
            numpy_to_qimage(image=denoise_img, shape=self.denoised_by_tv_image_size, is_rgb=False))

        psnr_score = calculate_psnr_score(self.gray_image_path, self.denoised_img_by_tv_path)
        ssim_score = calculate_ssim_score(self.gray_image_path, self.denoised_img_by_tv_path)

        print('[PSNR TV]', round(psnr_score, 2))
        print('[SSIM TV]', round(ssim_score, 2))

        self.set_score.emit('psnr_tv', round(psnr_score, 2))
        self.set_score.emit('ssim_tv', round(ssim_score, 2))

    def set_colorized_image(self):
        self.colorizer_image.colorizer_image(self.denoised_img_by_model_path, self.colorized_image_path)
        # print('[COLORIZE]', type(self.image))
        self.image = cv2.imread(self.colorized_image_path)

        # Set input pixelmap
        self.colorized_pixmap.emit(
            numpy_to_qimage(image=self.image, shape=self.colorized_image_size, is_rgb=False))

        psnr_score = calculate_psnr_score(self.original_image_path, self.colorized_image_path)
        ssim_score = calculate_ssim_score(self.original_image_path, self.colorized_image_path)

        print('[PSNR Colorized]', round(psnr_score, 2))
        print('[SSIM Colorized]', round(ssim_score, 2))

        self.set_score.emit('colorized_model', round(psnr_score, 2))
        self.set_score.emit('ssim_colorized_model', round(ssim_score, 2))

    def set_sigma(self, sigma: str):
        print('set sigma: ', sigma)
        self.sigma = int(sigma)

    def save_preview_frame(self):
        """
        Save preview frame to output folder
        :return:
        """
        if self.current_preview_frame is None:
            print("[MainModel][save_preview_frame] preview_frame is None")

        else:
            image_name = self.settings.configs['output_folder'] + "/" + str(datetime.datetime.now()) + ".jpg"
            cv2.imwrite(image_name, self.current_preview_frame)
