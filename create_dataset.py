import os 
import glob
import albumentations as A 
from torch.utils.data import Dataset 
import cv2 
import numpy as np 



mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def image_transforms(height, width, phase):
    image_transforms = {
        'train': A.Compose([
        A.Resize(height, width, always_apply = True),
        A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
        A.Flip(p=0.5)
        ]), 
        'valid': A.Compose([
        A.Resize(height, width, always_apply = True),
        A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]),
        'test': A.Compose([
        A.Resize(height, width, always_apply = True),
        A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]),
    }
    return image_transforms[phase]

class Custom_dataset(Dataset):
    def __init__(self, image_list, label_list, transform = None):
        self.image_list = image_list
        self.label_list = label_list 
        self.transform = transform 
        assert len(image_list) == len(label_list), "The number of images is not equal to the number of labels"
    
    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index]) #if cv2: cannot use pytorch.transpose
        label = self.label_list[index]

        if self.transform is not None:
            # print('self.transform: ', self.transform)
            image = self.transform(image = image)['image']
        image=np.transpose(image,(2, 1, 0))
        return image, label

    def __len__(self):
        return len(self.image_list)

 
class Sports_Dataset(object):
    def __init__(self, dataset_dir, labels, transform, phase = 'train'):
        self.dataset_dir = dataset_dir
        self.labels = sorted(labels)
        self.transform = transform
        self.phase = phase 
        
        self.image_list = []
        self.label_list = []
        for label in labels:
            temp = os.path.join(self.dataset_dir, phase, label)
            assert os.path.exists(temp), f"not exist {temp}"
            self.image_list.extend(glob.glob(temp + '/*.*'))
            self.label_list.extend([self.labels.index(label)] * len(os.listdir(temp)))
            assert len(self.image_list) == len(self.label_list), f"The number of images is not equal to the number of labels"

    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index]) #if cv2: cannot use pytorch.transpose
        label = self.label_list[index]

        if self.transform is not None:
            # print('self.transform: ', self.transform)
            image = self.transform(image = image)['image']
        image=np.transpose(image,(2, 1, 0))
        return image, label

    def __len__(self):
        return len(self.image_list)

 