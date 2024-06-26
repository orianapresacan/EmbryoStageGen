import os
import numpy as np
import cv2
import albumentations
import glob
import random
from PIL import Image
from torch.utils.data import Dataset


class EmbryoBase(Dataset):
    def __init__(self, data_csv, data_root,
                 size=256, random_crop=False, interpolation="bicubic",
                 shift_segmentation=False, subclass_name=None):
        
        self.shift_segmentation = shift_segmentation
        self.data_csv = data_csv
        self.data_root = data_root
        
        with open(self.data_csv, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }
        size = None if size is not None and size<=0 else size
        self.size = size
        
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)

        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example


class EmbryoTrain(EmbryoBase):
    def __init__(self, size=256, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv='/global/D1/homes/oriana/LDM/data/embryo/embryo_train_2cell.txt',
                         data_root='/global/D1/homes/oriana/LDM/data/embryo/2cell',
                         size=size, random_crop=random_crop, interpolation=interpolation,
                         subclass_name="Train")


# def write_lines(file, lines):
#     with open(file, 'w') as f:
#         for line in lines:
#             f.write(os.path.basename(line))
#             f.write('\n')


# def generateKvasirCSV(dir, output, train=1):
#     files = glob.glob(os.path.join(dir, '*.png'))
#     sorted_files = sorted(files)  # Sort the list of file paths

#     length = len(sorted_files)

#     train_data = sorted_files[:int(train * length)]
#     write_lines(f'{output}/embryo_train_Blastocyst.txt', train_data)
    
# generateKvasirCSV('/global/D1/homes/oriana/LDM/data/embryo/Blastocyst', '/global/D1/homes/oriana/LDM/data/embryo/')