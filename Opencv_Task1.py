import PIL.Image
import cv2
import pytesseract
from PIL import Image
import numpy as np

path = 'C:\Preetham_Kolli\CSE\Python\AI_ML\pythonProject\OpenCvPython\Resources\shapes_task_1.jpg'
img = cv2.imread(path)
imgContour = img.copy()


def getContours(img):
    # config = ('-l eng --oem 1 --psm 3')
    #
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # ret, threshing = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    #
    # rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    #
    # dilation = cv2.dilate(threshing, rect_kernel, iterations=1)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]


    kernel = np.ones((5, 5), np.uint8)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    imgDilated = cv2.dilate(imgCanny, kernel, iterations=1)
    myconfig = r"--psm 11 --oem 3"

    contours, hierarchy = cv2.findContours(imgDilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #sorted_contours = sorted(contours,key=cv2.contourArea,reverse=True)[1:]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        #print(x,y,w,h)
        cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_img = imgGray[y:y + h, x:x + w]
        #ret, img1 = cv2.threshold(np.array(cropped_img), 125, 255, cv2.THRESH_BINARY)
        #img2 = Image.fromarray(img1.astype(np.uint8))
        # cv2.imshow("Shape", cropped_img)
        # cv2.waitKey(0)
        # ret, thresh1 = cv2.threshold(cropped_img, 120, 255, cv2.THRESH_BINARY)
        # text = pytesseract.image_to_string(thresh1, config='--psm 6')
        text = pytesseract.image_to_string(cropped_img  ,config=myconfig)
        # if shape == text: print("Shape found")
        print(text)


#shape = input('Enter the shape name: ')
getContours(imgContour)

cv2.imshow("Output",imgContour)
cv2.waitKey(0)


*******************************************


Chapter-8:

import cv2

import numpy as np

path = 'C:\Preetham_Kolli\CSE\Python\AI_ML\pythonProject\OpenCvPython\Resources\shapes.png'

img = cv2.imread(path)
imgContour = img.copy()

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img):

    # Here, we retrieve the contours(boundary of a shape) from the image.
    # We will use the findContour func.
    # RETR_EXTERNAL: Retrieves the extreme outer contours.
    # CHAIN_APPROX_NONE: Retrieves all the information about the contours.
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(area)
        if area>500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            #print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            print(len(approx))
            objCor = len(approx)
            x,y,w,h = cv2.boundingRect(approx)

            if objCor ==3: objectType = "Tri"
            elif objCor == 4:
                aspRatio = w/float(h)
                if aspRatio >0.95 and aspRatio <1.05: objectType="Square"
                else: objectType="Rectangle"
            elif objCor>4: objectType="Circle"
            else: objectType = "None"

            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(imgContour,objectType,(x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny = cv2.Canny(imgBlur,50,50)
imgBlank = np.zeros_like(img) # For getting a complete black image(using the input image).

getContours(imgCanny)

# Instead of using horizontal and verical functions, use the below functions for stack of horizontal and vertical images.
imgStack = stackImages(0.6,([img,imgGray,imgBlur],
                             [imgCanny,imgContour,imgBlank] ))

cv2.imshow("Stack",imgStack)
cv2.waitKey(0)
