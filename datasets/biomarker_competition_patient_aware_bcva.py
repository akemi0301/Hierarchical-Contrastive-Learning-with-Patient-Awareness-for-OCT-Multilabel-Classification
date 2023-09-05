import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import os
import random
class PatientAware_BCVA_Dataset(data.Dataset):
    def __init__(self,df, img_dir, transforms):
        self.img_dir = img_dir
        self.transforms = transforms
        self.df = pd.read_csv(df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        bcva=self.df.iloc[idx,1]

        path_1 = self.img_dir + self.df.iloc[idx, 0]
        if path_1 == "":
            print(f"path_1:{path_1}")
            exit()
        path_2 = path_1
        if random.random() > 0.6:
            try:
                idx_2 = random.randint(idx-2, idx+2)
                while self.df.iloc[idx_2, 1] != bcva:
                    idx_2 = random.randint(idx-2, idx+2)

                path_2 = self.img_dir + self.df.iloc[idx_2, 0]
            except Exception as e:
                print(f"index cause exception:{idx}")

        image_1 = self.processing_image(path_1)
        image_2 = self.processing_image(path_2)

        image = self.transforms(image_1, image_2)
        return  path_1, path_2, image, bcva

    def processing_image(self, path):
        image = Image.open(path).convert("L")
        image = np.array(image)
        image = Image.fromarray(image)
        return image
