from models.colorizers.base_color import *
from models.colorizers.eccv16 import *
from models.colorizers.siggraph17 import *
from models.colorizers.util import *
import matplotlib.pyplot as plt
import cv2


# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
# img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
# out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
# out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
#
# plt.imsave(f'{opt.save_prefix}_eccv16.png', out_img_eccv16)
# plt.imsave(f'{opt.save_prefix}_siggraph17.png', out_img_siggraph17)


class ColorizerImage:
    def __init__(self):
        self.colorizer_eccv16 = eccv16(pretrained=True).eval()
        self.colorizer_siggraph17 = siggraph17(pretrained=True).eval()
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda:
            self.colorizer_eccv16.cuda()
            self.colorizer_siggraph17.cuda()

    def colorizer_image(self, source_image: str, save_path: str):
        img = load_img(source_image)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
        if self.is_cuda:
            tens_l_rs = tens_l_rs.cuda()

        out_img_eccv16 = postprocess_tens(tens_l_orig, self.colorizer_eccv16(tens_l_rs).cpu())
        out_img_siggraph17 = postprocess_tens(tens_l_orig, self.colorizer_siggraph17(tens_l_rs).cpu())
        plt.imsave(save_path, out_img_eccv16)
        return out_img_eccv16

