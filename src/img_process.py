import sys
import getopt
import numpy as np
import os
from glob import glob
import piexif
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
from src.TrackNetv2.predict.TrackNet import TrackNet
import keras.backend as K
from keras import optimizers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import cv2
from sklearn.cluster import DBSCAN
from os.path import isfile, join
from PIL import Image
import time
BATCH_SIZE=1
HEIGHT=288
WIDTH=512
sigma=2.5
mag=1

def genHeatMap(w, h, cx, cy, r, mag):
	if cx < 0 or cy < 0:
		return np.zeros((h, w))
	x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
	heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
	heatmap[heatmap <= r**2] = 1
	heatmap[heatmap > r**2] = 0
	return heatmap*mag

def custom_time(time):
	remain = int(time / 1000)
	ms = (time / 1000) - remain
	s = remain % 60
	s += ms
	remain = int(remain / 60)
	m = remain % 60
	remain = int(remain / 60)
	h = remain
	#Generate custom time string
	cts = ''
	if len(str(h)) >= 2:
		cts += str(h)
	else:
		for i in range(2 - len(str(h))):
			cts += '0'
		cts += str(h)
	
	cts += ':'

	if len(str(m)) >= 2:
		cts += str(m)
	else:
		for i in range(2 - len(str(m))):
			cts += '0'
		cts += str(m)

	cts += ':'

	if len(str(int(s))) == 1:
		cts += '0'
	cts += str(s)

	return cts

def custom_loss(y_true, y_pred):
	loss = (-1)*(K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1)))
	return K.mean(loss)

def get_ball_position(videoName : str):
    print(f'Beginning predicting {videoName}......')
    start = time.time()
    modelName = "/home/aivc2/AI_bedminton_dataset/src/TrackNetv2/predict/model_33"
    model = load_model(modelName, custom_objects={'custom_loss':custom_loss})
    cap = cv2.VideoCapture(videoName)
    success, image1 = cap.read()
    
    success, image2 = cap.read()
    success, image3 = cap.read()
    ratio = image1.shape[0] / HEIGHT

    size = (int(WIDTH*ratio), int(HEIGHT*ratio))
    fps = 30
    count = 3
    predict_position = []
    while success:
        unit = []
        #Adjust BGR format (cv2) to RGB format (PIL)
        x1 = image1[...,::-1]
        x2 = image2[...,::-1]
        x3 = image3[...,::-1]
        #Convert np arrays to PIL images
        x1 = array_to_img(x1)
        x2 = array_to_img(x2)
        x3 = array_to_img(x3)
        #Resize the images
        x1 = x1.resize(size = (WIDTH, HEIGHT))
        x2 = x2.resize(size = (WIDTH, HEIGHT))
        x3 = x3.resize(size = (WIDTH, HEIGHT))
        #Convert images to np arrays and adjust to channels first
        x1 = np.moveaxis(img_to_array(x1), -1, 0)		
        x2 = np.moveaxis(img_to_array(x2), -1, 0)		
        x3 = np.moveaxis(img_to_array(x3), -1, 0)
        #Create data
        unit.append(x1[0])
        unit.append(x1[1])
        unit.append(x1[2])
        unit.append(x2[0])
        unit.append(x2[1])
        unit.append(x2[2])
        unit.append(x3[0])
        unit.append(x3[1])
        unit.append(x3[2])
        unit=np.asarray(unit)	
        unit = unit.reshape((1, 9, HEIGHT, WIDTH))
        unit = unit.astype('float32')
        unit /= 255
        y_pred = model.predict(unit, batch_size=BATCH_SIZE)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype('float32')
        h_pred = y_pred[0]*255
        h_pred = h_pred.astype('uint8')
        frame_time = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
        if np.amax(h_pred) <= 0:
            predict_position.append([0, 0])
        else:	
            #h_pred
            (cnts, _) = cv2.findContours(h_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = [cv2.boundingRect(ctr) for ctr in cnts]
            max_area_idx = 0
            max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
            for i in range(len(rects)):
                area = rects[i][2] * rects[i][3]
                if area > max_area:
                    max_area_idx = i
                    max_area = area
            target = rects[max_area_idx]
            (cx_pred, cy_pred) = (int(ratio*(target[0] + target[2] / 2)), int(ratio*(target[1] + target[3] / 2)))
            predict_position.append([cx_pred, cy_pred])	
            
                
        image1 = image2
        image2 = image3
        success, image3 = cap.read()
        count += 1
    end = time.time()
    print('Prediction time:', end-start, 'secs')
    print('Done......')
    return np.array(predict_position)

def get_clustering_center(arr:np.array): 
    X = np.array(arr).reshape(-1, 1)
    dbscan = DBSCAN(eps=6, min_samples=1)
    dbscan.fit(X)
    labels = set(dbscan.labels_)
    centers = []
    for label in labels:
        indices = np.where(dbscan.labels_ == label)[0]
        center = np.mean(X[indices])
        centers.append(int(center))
    centers.pop(-1)
    return centers

def moment(path):
    ball_position = get_ball_position(path)
    for i in range(len(ball_position) - 2 , -1, -1):
        if ball_position[i, 0] == 0 and ball_position[i, 1] == 0:
            ball_position[i] = ball_position[i + 1] 
    vector_vol = ball_position[1 : len(ball_position), :] - ball_position[0 : len(ball_position) - 1]
    vector_acc = vector_vol[1 : len(vector_vol), :] - vector_vol[0 : len(vector_vol) - 1]
    mag_acc = np.array([np.linalg.norm(acc) for acc in vector_acc])
    delta_acc = mag_acc[1 : len(mag_acc)] - mag_acc[0 : len(mag_acc) - 1]#算加速度純量變化量
    delta_acc[delta_acc < 0] = 0#把負的改成0
    mean = np.mean(delta_acc)#算平均的值
    predict_output = np.array(np.where(delta_acc > mean)) + 3#挑出變化量大於平均值的index，再將index轉為frame
    predict_output = get_clustering_center(predict_output)#dbscan找群集中心
    return predict_output, len(predict_output)

# if __name__ == "__main__":
# print(os.path.dirname(os.path.abspath(__file__)))
if __name__ == "__main__":
    video_list = ["00001", "00002","00003", "00004", "00005", "00006", "00007", "00008", "00009", "00010", "00011"]
    for video in video_list:
        videoName = f"src_videos/{video}.mp4"
        # videoName ="src_videos/00001.mp4"
        # print(videoName)
        # answerCSV = f"part1/train/{video}/{video}_S2.csv"
        predict_output, hit_number = moment(videoName)
        print(f"{video}" + ".mp4 predict outcome:")
        # print(f"Ground Truth Hit Frame = {list(ans)}, Hit Number = {len(ans)}")
        print(f"Predicted Hit Frame = {predict_output}, Hit Number = {hit_number}")
        print("============================================我是分隔線============================================\n")

