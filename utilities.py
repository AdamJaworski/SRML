import os
import pathlib
import random
import cv2
import numpy as np
import torchvision.transforms.functional as FT
from piqa import SSIM
import torch


def convert_image(image, target):

    if target == "torch":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = FT.to_tensor(image)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor

    if target == "cv2":
        image = image.squeeze()

        image = image.float() * 255.0
        image = image.numpy()
        image = np.clip(image, 0, 255)
        image = np.transpose(image, (1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image


def compare_results(run_id):
    full_hd_path = r'./data/gt/full_hd/'
    output_path = r'./data/out/'
    full_hd_bicubic_path = r'./data/gt/full_hd_bicubic/'

    file_list = os.listdir(output_path + run_id)

    random_photos = [x for x in random.choices(file_list, k=10)]
    # random_photos = ['1_58.png', '1_59.png', '1_60.png', '1_1.png', '1_2.png', '1_7.png', '1_227.png']

    ssim = SSIM()
    with torch.no_grad():
        for photo in random_photos:
            gt_photo   = cv2.cvtColor(cv2.imread(full_hd_path + photo.split('_')[1]), cv2.COLOR_BGR2RGB)
            gt_photo   = convert_image(gt_photo, 'torch')
            bicubic    = cv2.cvtColor(cv2.imread(full_hd_bicubic_path + photo.split('_')[1]), cv2.COLOR_BGR2RGB)
            bicubic = convert_image(bicubic, 'torch')
            out_photo  = cv2.cvtColor(cv2.imread(output_path + run_id + '/' + photo), cv2.COLOR_BGR2RGB)
            out_photo = convert_image(out_photo, 'torch')

            out_value     = ssim(gt_photo, out_photo)
            bicubic_value = ssim(gt_photo, bicubic)
            print(f"out value: {out_value:.4}, bicubic value: {bicubic_value:.4} for image: {photo}")


if __name__ == "__main__":
    compare_results('Delta')
