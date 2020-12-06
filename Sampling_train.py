import os
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder
import cv2
from utils import StrtoLabel
from utils import trans_square

# 维度：31+10+24=65
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

upper_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z']
# print(len(upper_char))  # 24

Province_symbol = ['藏', '川', '鄂', '甘', '赣', '贵', '桂', '黑', '沪', '吉', '冀', '津', '晋', '京',
                   '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '陕', '苏', '皖', '湘', '新', '渝', '豫',
                   '粤', '云', '浙']
# print(len(Province_symbol))  # 31


class Sampling(data.Dataset):
    def __init__(self, root):
        self.transform = data_transforms
        self.imgs = []
        self.labels = []
        for filenames in os.listdir(root):
            x = os.path.join(root, filenames)
            y = filenames.split('_')[0]

            self.imgs.append(x)
            self.labels.append(y)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]

        img = Image.open(img_path)
        # img = trans_square(img)
        img = self.transform(img)

        label = self.labels[index]
        # print(label)  # f3iX

        label = StrtoLabel(label)  # 将字母转成数字表示，方便做one-hot
        # print(label)

        label = self.one_hot(label)
        # print(label)

        return img, label

    def one_hot(self, x):

        z = np.zeros(shape=[7, 65])
        for i in range(7):
            index = int(x[i])
            z[i][index] = 1

        return z


if __name__ == '__main__':
    samping = Sampling("./blue_plate")
    dataloader = data.DataLoader(samping, 10, shuffle=True)
    for i, (img, label) in enumerate(dataloader):
        print(i)
        print(img.shape)  # torch.Size([10, 3, 140, 440])
        print(label.shape)  # torch.Size([10, 3, 140, 440])
