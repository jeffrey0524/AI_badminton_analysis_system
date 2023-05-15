from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[1]
import os
from random import randint
import pandas as pd
import numpy as np
import cv2
import pickle
from datetime import datetime
import gc

# import csv_change
import pre_process

# global veriable define
pickle_path = 'src/pickle'
frame_w = 1280
frame_h = 720
frame_channal = 3
frame_color = cv2.IMREAD_COLOR # cv2.IMREAD_COLOR  cv2.IMREAD_GRAYSCALE
nonhit_labal = [0, 0, 'X', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'X']

def get_train_cube(hit_frame_list,img_dir_path):
    long = len(hit_frame_list)
    if long >= 4:
        train_cube = np.concatenate([get_train_cube(hit_frame_list[:int(long/4)],img_dir_path[:int(long/4)]), 
                                     get_train_cube(hit_frame_list[int(1*long/4):int(2*long/4)],img_dir_path[int(1*long/4):int(2*long/4)]), 
                                     get_train_cube(hit_frame_list[int(2*long/4):int(3*long/4)],img_dir_path[int(2*long/4):int(3*long/4)]), 
                                     get_train_cube(hit_frame_list[int(3*long/4):int(4*long/4)],img_dir_path[int(3*long/4):int(4*long/4)])], axis=0)
        # print(np.shape(train_cube))
        return train_cube
    elif long>1:
        train_cube = np.concatenate([get_train_cube(hit_frame_list[:int(long/2)],img_dir_path[:int(long/2)]), 
                                     get_train_cube(hit_frame_list[int(long/2):],img_dir_path[int(long/2):])], axis=0)
        # print(np.shape(train_cube))
        return train_cube
    else:
        # print(hit_frame_list)
        return __one_cube(hit_frame_list[0],img_dir_path[0])

def __get_onevideo_train_cube(hit_frame_list,img_dir_path):
    long = len(hit_frame_list)
    if long >= 4:
        train_cube = np.concatenate([__get_onevideo_train_cube(hit_frame_list[:int(long/4)],img_dir_path), 
                                     __get_onevideo_train_cube(hit_frame_list[int(1*long/4):int(2*long/4)],img_dir_path), 
                                     __get_onevideo_train_cube(hit_frame_list[int(2*long/4):int(3*long/4)],img_dir_path), 
                                     __get_onevideo_train_cube(hit_frame_list[int(3*long/4):int(4*long/4)],img_dir_path)], axis=0)
        return train_cube
    elif long>1:
        train_cube = np.concatenate([__get_onevideo_train_cube(hit_frame_list[:int(long/2)],img_dir_path), 
                                     __get_onevideo_train_cube(hit_frame_list[int(long/2):],img_dir_path)], axis=0)
        print(np.shape(train_cube))
        return train_cube
    else:
        # print(hit_frame_list)
        return __one_cube(hit_frame_list[0],img_dir_path)

def __one_cube(hit_frame,img_dir_path):
    hit_frame = hit_frame-5
    if os.path.exists(f'{img_dir_path}/{hit_frame}.jpg'):
        img0 = cv2.imread(f'{img_dir_path}/{hit_frame}.jpg', frame_color).reshape(frame_h,frame_w,frame_channal,1)
    else:
        img0 = np.full((frame_h,frame_w,frame_channal,1),0)
        print('file:',f'{img_dir_path}/{hit_frame}.jpg')
        # print('hit',hit_frame,'not exeit')
    hit_frame += 1
    if os.path.exists(f'{img_dir_path}/{hit_frame}.jpg'):
        img1 = cv2.imread(f'{img_dir_path}/{hit_frame}.jpg', frame_color).reshape(frame_h,frame_w,frame_channal,1)
    else:
        img1 = np.full((frame_h,frame_w,frame_channal,1),0)
        # print('hit',hit_frame,'not exeit')
    hit_frame += 1
    if os.path.exists(f'{img_dir_path}/{hit_frame}.jpg'):
        img2 = cv2.imread(f'{img_dir_path}/{hit_frame}.jpg', frame_color).reshape(frame_h,frame_w,frame_channal,1)
    else:
        img2 = np.full((frame_h,frame_w,frame_channal,1),0)
        # print('hit',hit_frame,'not exeit')
    hit_frame += 1
    if os.path.exists(f'{img_dir_path}/{hit_frame}.jpg'):
        img3 = cv2.imread(f'{img_dir_path}/{hit_frame}.jpg', frame_color).reshape(frame_h,frame_w,frame_channal,1)
    else:
        img3 = np.full((frame_h,frame_w,frame_channal,1),0)
        # print('hit',hit_frame,'not exeit')
    hit_frame += 1
    if os.path.exists(f'{img_dir_path}/{hit_frame}.jpg'):
        img4 = cv2.imread(f'{img_dir_path}/{hit_frame}.jpg', frame_color).reshape(frame_h,frame_w,frame_channal,1)
    else:
        img4 = np.full((frame_h,frame_w,frame_channal,1),0)
        # print('hit',hit_frame,'not exeit')
    hit_frame += 1
    if os.path.exists(f'{img_dir_path}/{hit_frame}.jpg'):
        img5 = cv2.imread(f'{img_dir_path}/{hit_frame}.jpg', frame_color).reshape(frame_h,frame_w,frame_channal,1)
    else:
        img5 = np.full((frame_h,frame_w,frame_channal,1),0)
        # print('hit',hit_frame,'not exeit')
    hit_frame += 1
    if os.path.exists(f'{img_dir_path}/{hit_frame}.jpg'):
        img6 = cv2.imread(f'{img_dir_path}/{hit_frame}.jpg', frame_color).reshape(frame_h,frame_w,frame_channal,1)
    else:
        img6 = np.full((frame_h,frame_w,frame_channal,1),0)
        # print('hit',hit_frame,'not exeit')
    hit_frame += 1
    if os.path.exists(f'{img_dir_path}/{hit_frame}.jpg'):
        img7 = cv2.imread(f'{img_dir_path}/{hit_frame}.jpg', frame_color).reshape(frame_h,frame_w,frame_channal,1)
    else:
        img7 = np.full((frame_h,frame_w,frame_channal,1),0)
        # print('hit',hit_frame,'not exeit')
    hit_frame += 1
    if os.path.exists(f'{img_dir_path}/{hit_frame}.jpg'):
        img8 = cv2.imread(f'{img_dir_path}/{hit_frame}.jpg', frame_color).reshape(frame_h,frame_w,frame_channal,1)
    else:
        img8 = np.full((frame_h,frame_w,frame_channal,1),0)
        # print('hit',hit_frame,'not exeit')
    hit_frame += 1
    if os.path.exists(f'{img_dir_path}/{hit_frame}.jpg'):
        img9 = cv2.imread(f'{img_dir_path}/{hit_frame}.jpg', frame_color).reshape(frame_h,frame_w,frame_channal,1)
    else:
        img9 = np.full((frame_h,frame_w,frame_channal,1),0)
        # print('hit',hit_frame,'not exeit')
    hit_frame += 1
    if os.path.exists(f'{img_dir_path}/{hit_frame}.jpg'):
        img10 = cv2.imread(f'{img_dir_path}/{hit_frame}.jpg', frame_color).reshape(frame_h,frame_w,frame_channal,1)
    else:
        img10 = np.full((frame_h,frame_w,frame_channal,1),0)
        # print('hit',hit_frame,'not exeit')
    one_cube = np.concatenate([img0,img1,img2,img3,img4,img5,img6,img7,img8,img9,img10],axis=3)
    return one_cube.reshape(1,frame_h,frame_w,frame_channal,11)

def __read_csv_random_shot(csv_path:str) -> list[int]:
    df = pd.read_csv(csv_path)
    hit_list = df[df.columns[1]].values.tolist()
    for i in range(0,len(hit_list)):
        hit_list[i] += randint(-5,5)
    return hit_list

def __save_pickle(cube, labal_csv):
    with open(f'{pickle_path}/cube.pickle', 'wb') as f:
        pickle.dump(cube, f)
    with open(f'{pickle_path}/csv.pickle', 'wb') as f:
        pickle.dump(labal_csv, f)
    pass

def __load_pickle():
    with open(f'{pickle_path}/cube.pickle', 'rb') as f:
        cube = pickle.load(f)
    with open(f'{pickle_path}/csv.pickle', 'rb') as f:
        labal_csv = pickle.load(f)
    return cube, labal_csv

def get_data_list(dir_path:str):
    paths = os.listdir(dir_path)
    paths.sort()
    result = []
    for path in paths:
        csv_path = f'{dir_path}/{path}/{path}_S2.csv'
        img_dir_path = f'{dir_path}/{path}/{path}'
        if not os.path.isdir(img_dir_path):
            background, img_dir_path = pre_process.get_field(f'{dir_path}/{path}/{path}.mp4', img_dir_path)
        hitframes = __read_csv_random_shot(csv_path)
        labal_list = pd.read_csv(csv_path).values.tolist()
        for i in range(len(labal_list)):
            labal_list[i][2] = labal_list[i][2][0]     # remove ' '
        #? mix with a not hit data
        difference = []
        for i in range(len(hitframes)-1):
            difference.append(hitframes[i+1] - hitframes[i])
            continue
        if len(difference) >=1:
            if max(difference)>20:
                nonhit_frame = randint(hitframes[difference.index(max(difference))]+10, hitframes[difference.index(max(difference))+1]-10)
                hitframes.append(nonhit_frame)
                labal_list.append(nonhit_labal)
        for i in range(len(hitframes)):
            if labal_list[i][1] != 0: labal_list[i][1] -= hitframes[i]-5
            #? list to tensor
            # list[1, 8, 'B', 2, 2, 2, 736, 386, 633, 565, 724, 415, 2, 'X'] =>
            # tensor[0,0,0,0,0,0,0,0,1,0,0, 0,1,0, 0,1, 0,1, 0,1, [20]736, 386, 633, 565, 724, 415, 0,1,0,0,0,0,0,0,0, 0,0,1]
            import torch
            tensor = torch.zeros(38)
            tensor[labal_list[i][1]] = 1
            if labal_list[i][3] == 1: tensor[14] = 1
            if labal_list[i][3] == 2: tensor[15] = 1
            if labal_list[i][4] == 1: tensor[16] = 1
            if labal_list[i][4] == 2: tensor[17] = 1
            if labal_list[i][5] == 1: tensor[18] = 1
            if labal_list[i][5] == 2: tensor[19] = 1
            tensor[20] = labal_list[i][6]
            tensor[21] = labal_list[i][7]
            tensor[22] = labal_list[i][8]
            tensor[23] = labal_list[i][9]
            tensor[24] = labal_list[i][10]
            tensor[25] = labal_list[i][11]
            tensor[25 + labal_list[i][12]] = 1
            if labal_list[i][13] == 'A': tensor[-3] = 1
            elif labal_list[i][13] == 'B': tensor[-2] = 1
            else: tensor[-1] = 1
            if labal_list[i][2] == 'A': tensor[11] = 1
            elif labal_list[i][2] == 'B': tensor[12] = 1
            else:  # nohit
                tensor[13] = 1
                tensor[0] = 0
                tensor[25 + labal_list[i][12]] = 0
                tensor[-1] = 0
            # print(tensor)
            result.append([img_dir_path,hitframes[i], tensor])
    # [[img_path, rendom hitframe, labal],[img_path, rendom hitframe, labal]]
    return result

def list_to_cube(data_list): # :list[[str,int,list],[str,int,list]]
    img_path_list = []
    hit_frame_list = []
    for i in data_list:
        img_path_list.append(i[0])
        hit_frame_list.append(i[1])
    return get_train_cube(hit_frame_list,img_path_list)

if __name__ == '__main__':
    aatime = datetime.now()
    # print('start_time',aatime)
    dir_path = PROJECT_DIR/'part1/train'
    dir_path = PROJECT_DIR/'test_part1/train'
    # cube, labal_csv = __load_pickle()
    # cube, labal_csv = data_process(dir_path)
    # print('cube shape = ',np.shape(cube))
    # print(len(labal_csv))
    # pd.DataFrame(labal_csv).to_csv('./labal.csv')
    # print('spend_time = ',datetime.now()-aatime)
    # __save_pickle(cube, labal_csv)

    data_list = get_data_list(dir_path)
    # print(len(data_list))
    print(data_list[0:5])
    train_cube = list_to_cube(data_list[0:3])
    print(train_cube.shape)

    pass