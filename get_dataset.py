import numpy as np
import pandas as pd
import random
import os
import cv2
import time
import pickle
from src.pre_process import get_field

def get_cubeRange(hitFrame, non_hit_num):
    range_list = []
    for i in hitFrame:
        range_list.append(get_random_range(i))
    range_list = np.array(range_list)
    # print(range_list)
    range_diff = range_list[1 : len(range_list), 0] - range_list[0 : len(range_list) - 1, 1]
    # print(range_diff)
    diff_index = np.where(range_diff > 0)
    diff_index = np.array(diff_index)
    diff_index = diff_index.reshape(diff_index.size)
    # print(diff_index)
    if len(diff_index) == 0:
        return np.zeros(1)
    diff_index = random.choice(diff_index)
    non_hit_frame = int((range_list[diff_index, 1] + range_list[diff_index + 1, 0]) / 2)
    for i in range(non_hit_num):
        range_list = np.append(range_list, np.array(get_random_range(non_hit_frame)).reshape(1, 2), 0)
    return range_list

def get_random_range(i : int):
    upward = random.randint(3, 7)
    downward = 11 - upward
    s = i - upward
    e = i + downward
    if s < 1:
        s = 1
        e = 11
    return s, e

aaa = time.time()
Path1 = 'part1/train'
file1 = os.listdir(Path1)
datasetCSV = open("part1/sample.csv", 'w')
h = 360
w = 640
c = 0
datasetCube = np.zeros((1, h, w, 11), np.uint8)
for i in range(len(file1)):#針對每個資料夾做處理
    s = time.time()
    file = file1[i]
    # file = "00289"
    # print(i)
    print(f"pre-processing for {file} ...")
    pre_process_frames = Path1 + '/' + file
    non_hit_num = 2
    des = os.listdir(pre_process_frames)
    desVideo = None
    desCSV = None
    for f in des:
        x = f.split('.', -1)
        f = pre_process_frames + '/' + f
        if x[-1] == 'mp4':
            desVideo = f
        if x[-1] == 'csv':
            desCSV =f
    ansCSV = pd.read_csv(desCSV)
    ansCSV.replace('\s+', '', regex=True, inplace=True)#去除空格
    datasetCSV.write(ansCSV.to_csv(index=False, header=False))
    if len(ansCSV) >= 3:
        for num in range(non_hit_num):
            datasetCSV.write('0,0,A,0,0,0,0,0,0,0,0,0,0,X\n')
    else:# 拍數過少的影片不添加沒打到的cube
        non_hit_num = 0
    hitFrame = np.array(ansCSV[ansCSV.columns[1]])
    video2imgPath = pre_process_frames + '/' + file
    if not os.path.exists(video2imgPath):
        os.makedirs(video2imgPath)
        background, savePath = get_field(desVideo, video2imgPath)     
    cubeRange = get_cubeRange(hitFrame, non_hit_num)
    if cubeRange.all() == np.zeros(1):
        continue
    print(f"hit number : {len(cubeRange) - non_hit_num}, non-hit number : {non_hit_num}")
    # print(cubeRange)
    for (start, end) in cubeRange:#one hit for a loop
        # print(start, end)
        oneHitFrames = np.zeros((1, h, w, 11), dtype=np.uint8)
        for j in range(0, 11):#read 11 frames for a hit
            imgPath = f"{pre_process_frames}/{file}/{start + j}.jpg"
            img = cv2.imread(imgPath, 0)
            img = cv2.resize(img, (h, w))
            oneHitFrames[:, :, :, j] = img.reshape(1, h, w)
        datasetCube = np.vstack((datasetCube, oneHitFrames))#stack one-hit cube under datasetcube
    print(datasetCube[1:].shape)
    print(f"{file} spend {time.time()- s} secs")
    print("=====================================我是分隔線=====================================")
    print(i)
    if (i+1) % 50 == 0:
        c += 1
        picklePath = f"part1/pickles/dataset{c}.pickle"
        with open(picklePath, 'wb') as f:
            datasetCube = datasetCube[1:]
            pickle.dump(datasetCube, f)
        datasetCube = np.zeros((1, h, w, 11), np.uint8)
        print(f"auto save during pre-process folder {i+1}")

 
# datasetCube = datasetCube[1:]
# with open(picklePath, 'wb') as f:
#     pickle.dump(datasetCube, f)


print(f"Output dataset shape : {datasetCube.shape}. Spent {time.time() - aaa} secs")
print(f"dataset.pickle had saved to \"{picklePath}\"")
