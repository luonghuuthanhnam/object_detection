import random
import torch
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image, ImageFilter
import PIL
import torchvision

def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data

ColorJitter = torchvision.transforms.ColorJitter(brightness=[0.5,1.2], contrast=[0.8,1.2], saturation=(0.8,1.2), hue=(-0.1,0.1))
unloader = torchvision.transforms.ToPILImage()
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-2)
            bbox = target["boxes"]
            bbox[:,[1, 3]] = height - bbox[:, [3, 1]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-2)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target

class RandomScale(object):
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, image, target):
        if random.random() < self.prob:
            pil_img = tensor_to_PIL(image)
            height, width = image.shape[-2:]
            rand_list = np.random.uniform(-0.5, 0.0, 5)
            rand_scale = 1
            for _scale in rand_list:
              if abs(_scale) >= 0.1:
                rand_scale = (1 + _scale)
                break
            height, width = int(height*rand_scale) , int(width*rand_scale)
            pil_img = pil_img.resize((width, height), Image.ANTIALIAS)
            
            image = F.to_tensor(pil_img)
            
            bbox = target["boxes"]
            bbox = bbox*rand_scale
            bbox = bbox.type(torch.int)
            target["boxes"] = bbox

            if "masks" in target:
                print("not support masks")
            if "keypoints" in target:
                print("not support keypoints")
        return image, target

class RandomColor(object):
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = ColorJitter(image)
            if "masks" in target:
                print("not support masks")
            if "keypoints" in target:
                print("not support keypoints")
        return image, target

class GaussNoise(object):
    def __init__(self, prob, mean=0, std=30, auto = True):
        self.prob = prob
        self.std = std
        self.mean = mean
        self.auto = auto
    def __call__(self, image, target):
        if random.random() < self.prob:
            pil_img = tensor_to_PIL(image)
            cv2_img = np.array(pil_img)
            std = self.std
            if self.auto == True:
              std = int(cv2_img.shape[0]/45)
            gaussian = (np.random.normal(self.mean, std, (cv2_img.shape[0],cv2_img.shape[1], 3)))
            cv2_img = cv2_img + gaussian
            cv2_img =  np.clip(cv2_img,0,255)
            pil_img = Image.fromarray((cv2_img).astype(np.uint8))
            image = F.to_tensor(pil_img)
            if "masks" in target:
                print("not support masks")
            if "keypoints" in target:
                print("not support keypoints")
        return image, target

class Blur(object):
    def __init__(self, prob, blur_range = (0,3)):
        self.prob = prob
        self.s_blur, self.e_blur = blur_range
    def __call__(self, image, target):
        if random.random() < self.prob:
            pil_img = tensor_to_PIL(image)
            radius = random.randrange(self.s_blur,self.e_blur)
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius))
            image = F.to_tensor(pil_img)
            if "masks" in target:
                print("not support masks")
            if "keypoints" in target:
                print("not support keypoints")
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
