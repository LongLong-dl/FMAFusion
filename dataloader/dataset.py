import os
import csv

import numpy as np
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset

import cv2


class Nutrition_RGBD_CSV(Dataset):
    def __init__(self, imagery_root, split_csv_path, transform=None):
        self.imagery_root = imagery_root
        self.transform = transform

        self.dish_ids = []
        self.total_calories = []
        self.total_mass = []
        self.total_fat = []
        self.total_carb = []
        self.total_protein = []

        with open(split_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dish_id = row['dish_id']
                self.dish_ids.append(dish_id)
                self.total_calories.append(np.array(float(row['calories']), dtype=np.float32))
                self.total_mass.append(np.array(float(row['mass']), dtype=np.float32))
                self.total_fat.append(np.array(float(row['fat']), dtype=np.float32))
                self.total_carb.append(np.array(float(row['carb']), dtype=np.float32))
                self.total_protein.append(np.array(float(row['protein']), dtype=np.float32))

    def __getitem__(self, index):
        dish_id = self.dish_ids[index]
        dish_dir = os.path.join(self.imagery_root, 'realsense_overhead', dish_id)

        rgb_path = os.path.join(dish_dir, 'rgb.png')
        rgbd_path = os.path.join(dish_dir, 'depth_color.png')

        img_rgb = cv2.imread(rgb_path)
        img_rgbd = cv2.imread(rgbd_path)

        if img_rgb is None or img_rgbd is None:
            raise FileNotFoundError(
                f"Image missing in {dish_dir}: rgb={img_rgb is None}, rgbd={img_rgbd is None}")

        img_rgb = Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
        img_rgbd = Image.fromarray(cv2.cvtColor(img_rgbd, cv2.COLOR_BGR2RGB))

        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
            img_rgbd = self.transform(img_rgbd)

        return (img_rgb,
                dish_id,
                self.total_calories[index],
                self.total_mass[index],
                self.total_fat[index],
                self.total_carb[index],
                self.total_protein[index],
                img_rgbd)

    def __len__(self):
        return len(self.dish_ids)
