import cv2
import numpy as np
import time


def get_background(videoPath:str):
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
    # print(f"{videoPath} concatenation done. Spent {time.time() - a}")
    a = time.time()
    most_common = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=2, arr=background)
    # print(f"{videoPath} find background done. Spent {time.time() - a}")
    most_common = most_common.astype(np.uint8)
    cap.release()
    return most_common

def saveContour(img:np.ndarray, contours):
    contourColor = (0, 255, 255)
    contourThick = 2
    contours = list(contours)
    contours.sort(key = lambda c: cv2.contourArea(c), reverse = True)
    max_contour = contours[0]
    rect = np.zeros((4, 2), dtype = np.int16) 
    s = max_contour.sum(axis = 2)
    # print(max_contour)
    # print(max_contour.shape)
    rect[0] = max_contour[np.argmin(s)]#左上
    rect[2] = max_contour[np.argmax(s)]#右下
    rect[3] = [min(max_contour[:, :, 0]), max(max_contour[:, :, 1])]#左下
    rect[1] = [rect[2, 0] - (rect[0, 0] - rect[3, 0]) , min(max_contour[:, :, 1])]#右上
    cv2.drawContours(img, max_contour, -1, contourColor, contourThick)
    for i in range(4):
        # print(rect[i])
        cv2.circle(img, rect[i], 8, (0, 0, 255), -1)
        cv2.putText(img, f"pt{i}", rect[i], cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return img

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
    diff = cv2.medianBlur(diff, 5)  
    diff_3 = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    output = cv2.bitwise_and(frame, diff_3)     
    return output, diff

# videoPaths = ['00001.mp4', '00002.mp4', '00003.mp4', '00004.mp4', '00005.mp4', '00007.mp4']
videoPaths = ['00001.mp4']
if __name__ == "__main__":
    for video in videoPaths:
        cap = cv2.VideoCapture(video)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"./background_remove_output/video/background_remove{video[0:5]}.mp4", fourcc, 30, (1280, 720))
        start = time.time()
        # background = get_background(video)
        background = cv2.imread(f"./background_remove_output/background_img/background{video[0:5]}.jpg")
        while True:
            ret, frame = cap.read()
            if ret == True:  
                output, diff = remove_background(frame, background)
                    
                # contours,hierarchy = cv2.findContours(diff ,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                # output = saveContour(output, contours)
                # cv2.imshow("video", output)
                # if cv2.waitKey(1) == ord('q'):
                #     break
                
                out.write(output)
            else:
                break

        out.release()
        print(f"{video} done. Spent {time.time() - start} s")



