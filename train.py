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
nonhit_labal = [0, 0, 'X', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'X']

if __name__ == '__main__':
    dir_path = 'part1/train'
    # dir_path = 'test_part1/train'    
    pass