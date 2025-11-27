import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class MeanTeacherAugmentation:
    '''
    Applies same augmentation pipeline twice to produce two augmented versions
    of the same image for both labeled and unlabeled training data
    '''
    def __init__(self):  
        self.transform = self.random_transform()  

    def __call__(self, img):
        '''
        Returns two independently augmented versions of the same image
        '''
        augmentation_1 = self.transform(img)
        augmentation_2 = self.transform(img)
        return augmentation_1, augmentation_2
    
    def random_transform(self): 
        '''
        Augmentation pipeline consisting random image translate, random horizontal flip
        '''
        return transforms.Compose([
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
        ])


class RandomTranslateWithReflect:
    '''
    Translate image randomly and fill the uncovered blank area with reflect padding
    this class is from offical mean teacher implementation  https://github.com/CuriousAI/mean-teacher
    '''
    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image