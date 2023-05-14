    import cv2
import numpy as np
import time

def saveContour(img:np.ndarray, contours):
    contourColor = (0, 255, 255)
    contourThick = 2
    contours = list(contours)
    contours.sort(key = lambda c: cv2.contourArea(c), reverse = True)
    max_contour = contours[0]
    rect = np.zeros((4, 2), dtype = np.int16) 
    s = max_contour.sum(axis = 2)
    # print(max_contour.shape)
    rect[0] = max_contour[np.argmin(s)]#左下
    rect[2] = max_contour[np.argmax(s)]#右上
    rect[3] = [min(max_contour[:, :, 0]), max(max_contour[:, :, 1])]
    rect[1] = [rect[2, 0] - (rect[0, 0] - rect[3, 0]) , min(max_contour[:, :, 1])]
    cv2.drawContours(img, max_contour, -1, contourColor, contourThick)
    for i in range(4):
        # print(rect[i])
        cv2.circle(img, rect[i], 8, (0, 0, 255), -1)
        cv2.putText(img, f"pt{i}", rect[i], cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return img


videoPaths = ['00001.mp4', '00002.mp4', '00003.mp4', '00004.mp4', '00005.mp4', '00007.mp4']
# videoPaths = ['00007.mp4']
for video in videoPaths:
    background = cv2.imread(f"./background_remove_output/background_img/BG{video[0:5]}.jpg")
    
    gray_img = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img[int(0.33 * 720) :, :]
    # gray_img = adjust_contrast(gray_img, 200)
    
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    max_bin = np.argmax(cdf >= 0.95)
    gray_img[gray_img < max_bin] = 0
    gray_img[gray_img >= max_bin] = 255
    
    
    kernel = np.ones((2,10), np.uint8)
    gray_img = cv2.dilate(gray_img, kernel)
    gray_img = cv2.erode(gray_img, kernel)
    
    # contours,hierarchy = cv2.findContours(gray_img ,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # background[int(0.33 * 720) :, :] = saveContour(background[int(0.33 * 720) :, :], contours)
    cv2.imwrite(f"background_remove_output/background_img/background{video[0:5]}.jpg", background)



