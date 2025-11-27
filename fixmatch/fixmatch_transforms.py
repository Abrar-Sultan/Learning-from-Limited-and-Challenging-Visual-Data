import torch
from torchvision import transforms
from randaugment import RandAugment  

class FixMatchAugmentation:
    '''
    Applies weak augmentation to the labeled images and 
    weak and strong augmentation to the unlabeled images 
    '''
    def __init__(self, use_strong_aug = False):
        self.use_strong_aug = use_strong_aug 

    def __call__(self, image):
        '''
        Returns weakly augmented version of the labeled image (when "use_strong_aug=False" we are using label data) again returns 
        weakly augmented and strongly versions of the same image for the unlabled data otherwise (when "use_strong_aug=True" we are using unlabel data)
        '''
        weak_aug = self.weak_augmentation()(image) 
        weak_aug = self.normalize_img()(weak_aug)

        if self.use_strong_aug:
            strong_aug = self.strong_augmentation()(image) 
            strong_aug = self.normalize_img()(strong_aug)
            return weak_aug, strong_aug 
        else:
            return weak_aug

    def weak_augmentation(self):
        '''
         weak augmentation pipeline consisting random horizontal flip, random crop
        '''
        return transforms.Compose([ 
            transforms.RandomHorizontalFlip(), 
            transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect') 
        ])

    def strong_augmentation(self):
        '''
         strong augmentation pipeline consisting of all the weak augmentaions along with Random augment
        '''
        return transforms.Compose([
            self.weak_augmentation(),
            RandAugment(n=2, m=12)
        ])
    
    def normalize_img(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)) # CIFAR 10
            # transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)) # CIFAR 100
        ])
