import json
import torch
from torchvision import transforms
from fixmatch.randaugment import RandAugment

class ReMixMatchAugmentation:
    '''
    Applies weak augmentation to the labeled images and 
    weak and k strong augmentation to the unlabeled images 
    '''
    def __init__(self, k=2, use_label_img_aug=True, use_cta=False, use_cta_probe=False, cta=None):
        self.k = k
        self.use_label_img_aug = use_label_img_aug
        self.use_cta = use_cta
        self.use_cta_probe = use_cta_probe
        self.cta = cta
        
    def __call__(self, image):
        '''
        Returns weakly augmented version of the labeled image (when "use_label_img_aug=True"). Again when "use_label_img_aug=False"  
        returns weakly augmented and k strongly augmented versions of the same image for the unlabled data using RandAugmment  
        (when "use_cta=False") otherwise use CTAugment for strong augmentation
        '''
        weak_aug = self.weak_augmentation()(image)
        weak_aug = self.normalize_img()(weak_aug) 

        # labeled data
        if self.use_label_img_aug:
            return weak_aug
        
        # unlabeled data (weak + k strong augmentations using RandAugment)
        if not self.use_cta:
            strong_augs = [] 
            for _ in range(self.k):
                strong_aug = self.strong_augmentation_rand_aug()(image)
                strong_aug = self.normalize_img()(strong_aug)
                strong_augs.append(strong_aug)
            # return tuple (weak + list of k strongs)
            strong_aug_stack = torch.stack(strong_augs, dim=0)
            return weak_aug, strong_aug_stack
        
        # unlabeled data (weak + k strong augmentations using CTAugment)
        elif self.cta and not self.use_cta_probe:
            strong_augs = [] 
            for _ in range(self.k):
                policy = self.cta.policy(probe=False)
                strong_aug = self.cta.cta_apply(image, policy)
                strong_aug = self.normalize_img()(strong_aug)
                strong_augs.append(strong_aug)
            strong_aug_stack = torch.stack(strong_augs, dim=0)
            return weak_aug, strong_aug_stack
        
        # for CTAugment update. Apply a randomly sampled policy to the image and returns both 
        # augmented image and the augmentation policy
        else:
            policy = self.cta.policy(probe=True)
            probe = self.cta.cta_apply(image, policy)
            probe = self.normalize_img()(probe)
            return probe, json.dumps(policy)
            
    def weak_augmentation(self):
        '''
         weak augmentation pipeline consisting random horizontal flip, random crop
        '''
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect')
        ])

    def strong_augmentation_rand_aug(self):
        '''
         strong augmentation pipeline consisting of all the weak augmentaions along with RandAugment
        '''
        return transforms.Compose([
            self.weak_augmentation(),
            RandAugment(n=2, m=10)   
        ])
    
    def normalize_img(self):
        return transforms.Compose([ 
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)), # CIFAR 10
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))  # CIFAR 100
        ])
    

        


    
