import numpy as np
import cv2
import time
import os
from src.pre_process import get_field


aaa = time.time()
Path1 = 'part1/train'
file1 = os.listdir(Path1)

for i in range(len(file1)):#針對每個資料夾做處理
    file = file1[i]
    print(f"pre-processing for {file} ...")
    pre_process_frames = Path1 + '/' + file
    des = os.listdir(pre_process_frames)
    desVideo = None
    desCSV = None
    video2imgPath = pre_process_frames + '/' + file
    if not os.path.exists(video2imgPath):
        os.makedirs(video2imgPath)
    for f in des:
        x = f.split('.', -1)
        f = pre_process_frames + '/' + f
        if x[-1] == 'mp4':
            desVideo = f
    background, savePath = get_field(desVideo, video2imgPath)
    # cv2.imshow("123", background)
    # cv2.waitKey(0)