import os
import pickle

from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[2]
if __name__ == '__main__':
    import sys
    sys.path.append(str(PROJECT_DIR))

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
    pass

def write_csv(dir_path,path,data):
    pass

if __name__ == '__main__':
    dir_path = 'part1/val'
    # dir_path = 'test_part1/val'
    paths = os.listdir(dir_path)
    paths.sort()
    # print(paths)
    for path in paths:
        print('video:',path)
        savePath = preprocess(dir_path,path)
        predict_output = process(dir_path,path)
        # print(predict_output)
        # cube_list = get_cube(predict_output,savePath)
        # for cube in cube_list:
        #     data = pridict(cube)
        #     write_csv(dir_path,path,data)
