import os
from random import randint
import pandas as pd
import numpy as np
import cv2
import pickle
from datetime import datetime

from src import csv_change
from src import pre_process

# global veriable define
pickle_path = 'src/pickle'
frame_w = 1280
frame_h = 720
frame_channal = 3
nonhit_labal = [0, 0, 'A', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'X']

def data_process(dir_path:str):
    paths = os.listdir(dir_path)
    paths.sort()
    csv_list = []
    train_cube = np.full((0,frame_h,frame_w,frame_channal,11),0)
    for path in paths:
        print('video:',path)
        csv_path = f'{dir_path}/{path}/{path}_S2.csv'
        #? deal with csv
        hit_frame_list = __read_csv_random_shot(f'{dir_path}/{path}/{path}_S2.csv')
        labal = pd.read_csv(csv_path).values.tolist()
        for i in range(len(labal)):
            labal[i][2] = labal[i][2][0]     # remove ' '
        csv_list += labal
        #? mix with a not hit data
        hit_frame_list.append(__get_nonhit_frame(hit_frame_list))
        csv_list.append(nonhit_labal)
        #? deal wiht cube
        video_path = f'{dir_path}/{path}/{path}.mp4'
        # background, img_dir_path = pre_process.get_field(video_path)
        img_dir_path = f'part1/train/{path}/{path}'
        print('pre_process done',datetime.now())
        # imgs = os.listdir(img_dir_path)
        for hit_frame in hit_frame_list:
            one_cube = np.full((frame_h,frame_w,frame_channal,0),0)
            # print('one_cube start',datetime.now())
            for hit in range(hit_frame-5,hit_frame+6):
                # print(f'{img_dir_path}/{hit}.jpg')
                if os.path.exists(f'{img_dir_path}/{hit}.jpg'):
                    img = np.expand_dims(cv2.imread(f'{img_dir_path}/{hit}.jpg'), axis = 3)
                else:
                    img = np.full((frame_h,frame_w,frame_channal,1),0)
                    print('hit',hit,'not exeit')
                one_cube = np.concatenate((one_cube,img),axis=3)
                continue
            # print('one_cube done',datetime.now())
            train_cube = np.concatenate((train_cube, np.expand_dims(one_cube, axis = 0)), axis=0)
            continue
        print('data_process done',datetime.now())
        print(np.shape(train_cube))
        continue
    return train_cube,csv_list

def __get_nonhit_frame(hit_frame_list):
    # after - before than detect which is longest than random to take a 11frame cube
    difference = []
    for i in range(len(hit_frame_list)-1):
        difference.append(hit_frame_list[i+1] - hit_frame_list[i])
        continue
    # print(difference)
    nonhit_frame = randint(hit_frame_list[difference.index(max(difference))]+10, hit_frame_list[difference.index(max(difference))+1]-10)
    return nonhit_frame

def __read_csv_random_shot(csv_path:str) -> list[int]:
    df = pd.read_csv(csv_path)
    hit_list = df[df.columns[1]].values.tolist()
    for i in range(0,len(hit_list)):
        hit_list[i] += randint(-5,5)
    return hit_list

def save_pickle(cube, labal_csv):
    with open(f'{pickle_path}/cube.pickle', 'wb') as f:
        pickle.dump(cube, f)
    with open(f'{pickle_path}/csv.pickle', 'wb') as f:
        pickle.dump(labal_csv, f)
    pass

def load_pickle():
    with open(f'{pickle_path}/cube.pickle', 'rb') as f:
        cube = pickle.load(f)
    with open(f'{pickle_path}/csv.pickle', 'rb') as f:
        labal_csv = pickle.load(f)
    return cube, labal_csv

if __name__ == '__main__':
    print('start_time',datetime.now())
    dir_path = 'part1/train'
    # dir_path = 'test_part1/train'
    cube, labal_csv = data_process(dir_path)
    # cube, labal_csv = load_pickle()
    print(np.shape(cube))
    print(labal_csv)
    save_pickle(cube, labal_csv)

    # path = '00001'
    # hit_frame_list = __read_csv_random_shot(f'{dir_path}/{path}/{path}_S2.csv')
    # print(hit_frame_list)
    # nonhit_frame = __get_nonhit_frame(hit_frame_list)
    # print(nonhit_frame)

    # video_path = f'{dir_path}/00001/00001.mp4'
    # background, path = pre_process.get_field(video_path)
    # print(path)
    # cv2.imshow(background)
    # cv2.waitKey(0)
    # cv2.imwrite('./test_part1/back.img', background)
    
    # hit_frame_list,img_dir_path = pre_process(dir_path)
    # imgs = os.listdir(img_dir_path)
    # for img in imgs:
    #     print(img)

    pass