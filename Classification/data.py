import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class EmbryoDataset(Dataset):
    def __init__(self, dirs_info, transform=None):
        self.transform = transform
        self.classes = os.listdir(dirs_info[0]['dir'])  
        self.image_paths = []  
        self.image_labels = []  
        
        for dir_info in dirs_info:
            self._process_dir(dir_info['dir'], dir_info['num_images'])

    def _process_dir(self, directory, num_images):
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(directory, class_name)
            image_count = 0  
            for filename in os.listdir(class_dir):
                if num_images is not None and image_count >= num_images:
                    break  
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.image_labels.append(class_idx)
                    image_count += 1 

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.image_labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


