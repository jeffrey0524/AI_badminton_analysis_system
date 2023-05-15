import os
import pickle
import pandas as pd
import torch
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[0]

from src import pre_process
from src import img_process
from src import csv_change
# from src import dataprocess

def preprocess(dir_path,path):
    img_dir_path = f'{dir_path}/{path}/{path}'
    if not os.path.isdir(img_dir_path):
        background, img_dir_path = pre_process.get_field(f'{dir_path}/{path}/{path}.mp4', img_dir_path)
    return img_dir_path

def process(dir_path,path):
    pickle_path = f'{dir_path}/{path}/{path}.pickle'
    if os.path.isfile(pickle_path):
        with open(pickle_path, 'rb') as f:
            predict_output = pickle.load(f)
    else:
        predict_output, hit_number = img_process.moment(f'{dir_path}/{path}/{path}.mp4')
        with open(pickle_path, 'wb') as f:
            pickle.dump(predict_output, f)
    return predict_output

def get_cube(predict_output,savePath):
    cube_list = dataprocess.get_train_cube(predict_output,savePath)
    return cube_list # (n,720,1280,3,11)

def pridict(cube):
    return []

def cube_to_csv(dir_path,path,data_list,hit_list):
    #? tensor to csv?
    if len(data_list) != len(hit_list): 
        print(f'len error: data_list = {len(data_list)}, hit_list = {len(hit_list)}')
    for i in range(len(hit_list)):
        # is it nonhit?
        if data_list[i][13] == 1:
            continue
        else:
            cube_label = [f'{path}.mp4',i,hit_list[i]-5+data_list[i][0]]
            for data in data_list[i][1:]:
                cube_label.append(data)
    return cube_label

def write_csv(dir_path,data_list):
    #   VideoName	ShotSeq	HitFrame	Hitter	RoundHead	…	BallType	Winner
    #   00001.mp4	1	    17	        A	    2	        …	1	        X
    #   00001.mp4	2	    24	        B	    2	        …	2	        X
    #   00001.mp4	3	    36	        A	    2	        …	3	        A
    #   00002.mp4	1	    11	        A	    2	        …	1	        X
    #   00002.mp4	2	    26	        B	    2	        …	5	        X
    #   00002.mp4	3	    40	        A	    1	        …	8	        A
    csv_labal = ['VideoName', 'ShotSeq', 'HitFrame', 'Hitter', 'RoundHead', 
             'Backhand', 'BallHeight','LandingX','LandingY','HitterLocationX',
             'HitterLocationY','DefenderLocationX','DefenderLocationY', 'BallType', 'Winner']
    df = pd.DataFrame(data_list)
    df.columns = csv_labal
    df.to_csv(f'{dir_path}.csv')
    pass

if __name__ == '__main__':
    dir_path = PROJECT_DIR/'part1/val'
    dir_path = PROJECT_DIR/'part2/test'
    dir_path = PROJECT_DIR/'test_part1/val'
    paths = os.listdir(dir_path)
    paths.sort()
    # print(paths)
    list_csv = []
    for path in paths:
        print('video:',path)
        savePath = preprocess(dir_path,path)
        predict_output = process(dir_path,path)
        # print(predict_output)
        cube_list = get_cube(predict_output,savePath)
        pridicted_list = []
        for cube in cube_list:
            pridicted_list.append(pridict(cube))
        list_csv += cube_to_csv(dir_path,path,pridicted_list,predict_output)
    write_csv(dir_path,list_csv)
