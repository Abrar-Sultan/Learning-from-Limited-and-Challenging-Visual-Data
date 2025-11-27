# note: this code is adopted from https://github.com/kekmodel/FixMatch-pytorch

import random
import numpy as np
import PIL
from PIL import Image, ImageOps, ImageEnhance, ImageDraw

# All magnitude parameters are scaled between [0, PARAMETER_MAX]
PARAMETER_MAX = 10


def Identity(img, **kwarg):
    '''
    Returns same image
    '''
    return img


def Invert(img, **kwarg):
    '''
    Returns the inverted the image
    '''
    return ImageOps.invert(img) 


def Equalize(img, **kwarg):
    '''
    Equalize the image histogram 
    '''
    return ImageOps.equalize(img) 


def Rotate(img, v, max_v, bias=0):
    '''
    Rotate the image by an angle.
    '''
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Posterize(img, v, max_v, bias=0):
    '''
    Reduce the number of bits for each color channel
    '''
    v = _int_parameter(v, max_v) + bias
    return ImageOps.posterize(img, v)


def Contrast(img, v, max_v, bias=0):
    '''
    Adjust image contrast based on the magnitude value v
    '''
    v = _float_parameter(v, max_v) + bias
    return ImageEnhance.Contrast(img).enhance(v)


def AutoContrast(img, **kwarg):
    '''
    Maximize image contrast automatically
    '''
    return ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    '''
    Adjust image brightness 
    '''
    v = _float_parameter(v, max_v) + bias
    return ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    '''
    Adjust image color saturation
    '''
    v = _float_parameter(v, max_v) + bias
    return ImageEnhance.Color(img).enhance(v)


def Sharpness(img, v, max_v, bias=0):
    '''
    Adjust image shrapnes 
    '''
    v = _float_parameter(v, max_v) + bias
    return ImageEnhance.Sharpness(img).enhance(v)


def TranslateX(img, v, max_v, bias=0):
    '''
    Translate image along x-axis (horizontally)
    '''
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    '''
    Translate image along y-axis (vertically)
    '''
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def ShearX(img, v, max_v, bias=0):
    '''
    Shear image along x-axis
    '''
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    '''
    Shear image along y-axis
    '''
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))



def Solarize(img, v, max_v, bias=0):
    '''
    Invert all pixel values above (256 - v)
    '''
    v = _int_parameter(v, max_v) + bias
    return ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    '''
    Add a constant to pixel values, then apply solarization
    '''
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return ImageOps.solarize(img, threshold)


def Cutout(img, v, max_v, bias=0):
    '''
    Apply cutout
    '''
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg): 
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img


def _float_parameter(v, max_v):
    '''
    Scale v to [0, PARAMETER_MAX]
    '''
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    '''
    Scale v to [0, PARAMETER_MAX] and convert to int
    '''
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs



class RandAugment(object):
    '''
    Randomly apply n augmentations from the pool, each with magnitude sampled from (1, m)
    '''
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 30
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        # Randomly choose n augmentation operations
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(32*0.5))
        return img