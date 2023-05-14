import pickle
import cv2
import os
import numpy as np
from time import time
Path1 = 'part1/pickles'
file1 = os.listdir(Path1)
file1 = sorted(file1)
# dataset = np.zeros((1, 360, 640, 11))
# dataset = np.zeros((16, 360, 640, 11), dtype=np.uint8)
dataset = []
i = 0
# print(dataset)
for file in file1:
    picklePath = Path1 + '/' + file
    print(picklePath)
    with open(picklePath, 'rb') as f:
        pic = pickle.load(f)
        
        # dataset = np.vstack((dataset, pic))
        # print(pic)
        dataset.append(pic)
        print(np.shape(pic))
dataset = np.concatenate(dataset, axis= 0)
# print(dataset)
print(np.shape(dataset))
print("dumping pickle...")
a = time()
picklePath = 'part1/output/DatasetCube.pickle'
with open(picklePath, 'wb') as f:
        pic = pickle.dump(dataset, f)
print(f"dumping pickle spent {time() - a} secs")

print("loading pickle...")
a = time()
with open(picklePath, 'rb') as f:
        dataset = pickle.load(f)
print(f"loading pickle spent {time() - a} secs")
print(np.array(dataset).shape)