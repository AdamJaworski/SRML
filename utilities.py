import cv2
import numpy as np
import torchvision.transforms.functional as FT


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
