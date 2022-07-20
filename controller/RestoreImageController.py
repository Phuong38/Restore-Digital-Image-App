from models.RestoreImageModel import RestoreImageModel


class RestoreImageController:
    def __init__(self, main_model: RestoreImageModel):
        super().__init__()
        self.main_model = main_model

    def set_input(self, input_source, preview_size: tuple):
        self.main_model.set_preview_size(preview_size)
        self.main_model.set_input_source(input_source)

    def set_noise_image(self, source_image, image_noise_size: tuple):
        self.main_model.set_preview_size(image_noise_size)
        self.main_model.set_noise_image(source_image)

    def set_denoise_image(self, image_denoise_size: tuple):
        self.main_model.set_denoised_image_size(image_denoise_size)
        self.main_model.set_denoised_image()

    def set_colorized_image(self, colorized_image_size: tuple):
        self.main_model.set_colorized_image_size(colorized_image_size)
        self.main_model.set_colorized_image()

    def set_denoised_by_tv_image(self, denoised_by_tv_image_size: tuple):
        self.main_model.set_denoised_by_tv_image_size(denoised_by_tv_image_size)
        self.main_model.set_denoised_by_tv_image()

    def set_sigma(self, sigma: str):
        self.main_model.set_sigma(sigma)

    def set_preview_size(self, preview_size: tuple):
        self.main_model.set_preview_size(preview_size)

    def run_render(self, input_source, preview_size: tuple):
        self.set_input(input_source, preview_size)
        self.main_model.run()

    def save_current_preview(self):
        self.main_model.save_preview_frame()
