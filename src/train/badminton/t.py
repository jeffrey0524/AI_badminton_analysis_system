import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision import transforms, datasets

import json
from PIL import Image
import pickle
import cv2 as cv
tensor = torch.randn(1, 34)  # 假設有一個1x34的張量
subset = tensor[:, 3]  # 取得前11個元素
print(subset)