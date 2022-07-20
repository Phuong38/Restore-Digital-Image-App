# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys

sys.path.insert(0, ".")
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from models.restoration.mapping_model import Pix2PixHDModel_Mapping
from models.options.test_options import TestOptions


def data_transforms(img, method=Image.BILINEAR, scale=False):
    ow, oh = img.size
    pw, ph = ow, oh
    if scale == True:
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

    h = int(round(oh / 4) * 4)
    w = int(round(ow / 4) * 4)

    if (h == ph) and (w == pw):
        return img

    return img.resize((w, h), method)


def data_transforms_rgb_old(img):
    w, h = img.size
    A = img
    if w < 256 or h < 256:
        A = transforms.Scale(256, Image.BILINEAR)(img)
    return transforms.CenterCrop(256)(A)


def irregular_hole_synthesize(img, mask):
    img_np = np.array(img).astype("uint8")
    mask_np = np.array(mask).astype("uint8")
    mask_np = mask_np / 255
    img_new = img_np * (1 - mask_np) + mask_np * 255

    hole_img = Image.fromarray(img_new.astype("uint8")).convert("RGB")

    return hole_img


def run_restore_image():
    opt = TestOptions().parse(save=False)
    parameter_set(opt)

    model = Pix2PixHDModel_Mapping()

    model.initialize(opt)
    model.eval()

    # pix2pix_model = Pix2PixHDModel()
    # pix2pix_model.initialize(opt)

    # model = InferenceModel()

    if not os.path.exists(f"{opt.outputs_dir}/input_image"):
        os.makedirs(f"{opt.outputs_dir}/input_image")
    if not os.path.exists(f"{opt.outputs_dir}/restored_image"):
        os.makedirs(f"{opt.outputs_dir}/restored_image")
    if not os.path.exists(f"{opt.outputs_dir}/origin"):
        os.makedirs(f"{opt.outputs_dir}/origin")

    dataset_size = 0

    input_loader = os.listdir(opt.test_input)
    dataset_size = len(input_loader)
    input_loader.sort()

    if opt.test_mask != "":
        mask_loader = os.listdir(opt.test_mask)
        dataset_size = len(os.listdir(opt.test_mask))
        mask_loader.sort()

    img_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    mask_transform = transforms.ToTensor()

    start = time.time()

    for i in range(dataset_size):

        input_name = input_loader[i]
        input_file = os.path.join(opt.test_input, input_name)
        if not os.path.isfile(input_file):
            print(f"Skipping non-file {input_name}")
            continue
        input = Image.open(input_file).convert("RGB")

        print(f"Now you are processing {input_name}")

        if opt.NL_use_mask:
            mask_name = mask_loader[i]
            mask = Image.open(os.path.join(opt.test_mask, mask_name)).convert("RGB")
            if opt.mask_dilation != 0:
                kernel = np.ones((3, 3), np.uint8)
                mask = np.array(mask)
                mask = cv2.dilate(mask, kernel, iterations=opt.mask_dilation)
                mask = Image.fromarray(mask.astype('uint8'))
            origin = input
            input = irregular_hole_synthesize(input, mask)
            mask = mask_transform(mask)
            mask = mask[:1, :, :]  ## Convert to single channel
            mask = mask.unsqueeze(0)
            input = img_transform(input)
            input = input.unsqueeze(0)
        else:
            if opt.test_mode == "Scale":
                input = data_transforms(input, scale=True)
            if opt.test_mode == "Full":
                input = data_transforms(input, scale=False)
            if opt.test_mode == "Crop":
                input = data_transforms_rgb_old(input)
            origin = input
            input = img_transform(input)
            input = input.unsqueeze(0)
            mask = torch.zeros_like(input)
            ### Necessary input

            # try:
            with torch.no_grad():
                generated = model.inference(input, mask)
        # except Exception as ex:
        #     print("Skip %s due to an error:\n%s" % (input_name, str(ex)))
        #     continue

        if input_name.endswith(".jpg"):
            input_name = f"{input_name[:-4]}.png"

        vutils.save_image((input + 1.0) / 2.0, f"{opt.outputs_dir}/input_image/{input_name}", nrow=1, padding=0,
                          normalize=True)

        vutils.save_image((generated.data.cpu() + 1.0) / 2.0, f"{opt.outputs_dir}/restored_image/{input_name}", nrow=1,
                          padding=0, normalize=True)

        origin.save(f"{opt.outputs_dir}/origin/{input_name}")

    print('time: ', time.time() - start)


def parameter_set():
    ## Default parameters
    opt = TestOptions().parse(save=False)
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.label_nc = 0
    opt.n_downsample_global = 3
    opt.mc = 64
    opt.k_size = 4
    opt.start_r = 1
    opt.mapping_n_block = 6
    opt.map_mc = 512
    opt.no_instance = True
    opt.checkpoints_dir = "assets/checkpoints/restoration"
    opt.test_mode = "Full"
    # opt.Quality_restore
    opt.test_input = '/media/phuonglt/DATA/VOCdevkit/VOC2012/test_data/test_set_B'
    opt.outputs_dir = '/media/phuonglt/DATA/VOCdevkit/VOC2012/test_data/test_output'
    opt.gpu_ids = []

    ##

    # if opt.Quality_restore:
    opt.name = "mapping_quality"
    # opt.load_pretrainA = '/media/phuonglt/DATA/checkpoint_domianA'
    opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
    opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_quality")
    # opt.load_pretrainB = '/media/phuonglt/DATA/checkpoint_doaminB/test_checkpoint'

    if opt.Scratch_and_Quality_restore:
        opt.NL_res = True
        opt.use_SN = True
        opt.correlation_renormalize = True
        opt.NL_use_mask = True
        opt.NL_fusion_method = "combine"
        opt.non_local = "Setting_42"
        opt.name = "mapping_scratch"
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_scratch")
        if opt.HR:
            opt.mapping_exp = 1
            opt.inference_optimize = True
            opt.mask_dilation = 3
            opt.name = "mapping_Patch_Attention"

    return opt


class RestoreImage:
    def __init__(self):
        self.mask_transform = None
        self.img_transform = None
        self.input_image = None
        self.opt = parameter_set()

        self.model = Pix2PixHDModel_Mapping()
        self.model.initialize(self.opt)
        self.model.eval()

    def restore_image(self, source_image):
        # print('image source: ', type(source_image))
        source_image = Image.fromarray(source_image).convert('RGB')
        self.img_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.mask_transform = transforms.ToTensor()

        # self.original_image = Image.open(source_image).convert("RGB")
        self.input_image = source_image
        if self.opt.test_mode == "Full":
            self.input_image = data_transforms(self.input_image, scale=False)

        self.input_image = self.img_transform(self.input_image)
        self.input_image = self.input_image.unsqueeze(0)
        mask = torch.zeros_like(self.input_image)

        with torch.no_grad():
            generated = self.model.inference(self.input_image, mask)

        vutils.save_image(
            (generated.data.cpu() + 1.0) / 2.0,
            'assets/denoised_image.jpg',
            nrow=1,
            padding=0,
            normalize=True,
        )

        return generated

