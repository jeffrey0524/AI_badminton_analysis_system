import cv2
import numpy as np
import time
import os

def get_background_img(videoPath:str):
    cap = cv2.VideoCapture(videoPath)
    ret, frame = cap.read()
    BGshape = frame.shape[:2] + (1, 3)
    background = np.ones(BGshape, dtype=np.uint8)
    a = time.time()
    count = 0
    while True:
        ret, frame = cap.read()
        if ret == True:  
            count += 1
            if count == 10:
                frame = frame.reshape(BGshape)
                background = np.concatenate((background, frame), axis = 2)
                count = 0
        else:
            break
    a = time.time()
    most_common = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=2, arr=background)
    most_common = most_common.astype(np.uint8)
    cap.release()
    return most_common

def remove_background(frame, background):
    background = cv2.GaussianBlur(background, (5, 5), 0)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY) 
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    diff = cv2.absdiff(background, gray_frame)  
    hist = cv2.calcHist([diff], [0], None, [256], [0, 256])
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    max_bin = np.argmax(cdf >= 0.98)
    diff[diff < max_bin] = 0
    diff[diff >= max_bin] = 255
    diff = cv2.medianBlur(diff, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel, iterations=2)  
    diff_3 = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    output = cv2.bitwise_and(frame, diff_3)     
    return (output, diff)

def get_field(video:str, savePath:str):
    # videoName = video.split(".", -1)
    # videoName = str(videoName[-2]).split("/", -1) 
    # # print(videoName)
    # savePath = f'./background_remove_output/{videoName[-1]}'
    # print(savePath)
    if not os.path.isdir(savePath):
        os.mkdir(savePath)
    # print(savePath)
    cap = cv2.VideoCapture(video)
    background = get_background_img(video)
    background_part = background[int(0.1 * 720):, int(0.15 * 1280):int(0.85 * 1280), :]
    counter = 0
    while True:
            ret, frame = cap.read()
            if ret == True:  
                counter += 1
                frame_part = frame[int(0.1 * 720):, int(0.15 * 1280):int(0.85 * 1280), :]
                output = np.zeros_like(frame)
                output[int(0.1 * 720):, int(0.15 * 1280):int(0.85 * 1280), :], diff = remove_background(frame_part, background_part)
                cv2.imwrite(f"{savePath}/{counter}.jpg", output)
            else:
                break
    return background, savePath

# if __name__ == "__main__":
#     video_list = ["00001", "00002","00003", "00004", "00005", "00006", "00007", "00008", "00009", "00010", "00011"]
#     for v in video_list:          
#         bg, savePath = get_field(f"src_videos/{v}.mp4")