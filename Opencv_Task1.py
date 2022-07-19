# Question: Write a program to take a user defined text input and detect the respective shape in shapes.jpg (use OpenCV only)
# If the user enters ‘square’ or any specified alias of it, the square is highlighted amongst all the shapes present in the input file.

# Importing the necessary packages.
import PIL.Image
import cv2
import pytesseract
import numpy as np

#Using the image shapes.jpg.
path = 'C:\Preetham_Kolli\CSE\Python\AI_ML\pythonProject\OpenCvPython\Resources\shapes_task_1.jpg'
img = cv2.imread(path)

# Creating a copy of the image.
imgContour = img.copy()


# Function which returns the dilated image.
def dilateImage(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
    return dilation


# Main function which highlights the user input shape.
def getContours(img,user_shape):

    imgDilated = dilateImage(img)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extracting the contours from the image.
    contours, hierarchy = cv2.findContours(imgDilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)

        # Function which approximates the shapes and contains total number of boundary points.
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # Extracts the top-left point(x,y) and the width,height of the bounded contour
        x, y, w, h = cv2.boundingRect(approx)

        # Finding the mid-point of bounded contour(Because text is present in centre of shape)
        M = cv2.moments(cnt)
        if M['m00'] != 0.0:
            x1 = int(M['m10'] / M['m00'])
            y1 = int(M['m01'] / M['m00'])


        # Cropping the gray image for text processing
        # If shape is triangle, cropping parameters needed to be altered slightly to extract the correct text.
        if(len(approx)==3):
            cropped_img = imgGray[y1 - int(h / 4):y1 + int(h / 4), x1 - int((w / 3) -3)-2:x1 + int(w / 3) + 1]
        else:
            cropped_img = imgGray[y1 - int(h / 4):y1 + int(h / 4), x1 - int(w / 3):x1 + int(w / 3) + 1]

        #Setting threshold of the gray-scale cropped image
        thr = cv2.threshold(cropped_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        #Extracting the text from the final processed cropped image
        text = pytesseract.image_to_string(thr)

        # Removing the last character(escape sequence) from the text
        text = text.rstrip(text[-1])

        # If user input shape = text, highlight the shape.
        if user_shape == text:
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)



# Accepting multi-line input from user.
# ******** AFTER ENTERING INPUT, PRESS ENTER TWICE FOR IT TO BE ACCPTED. **********


lines = []
print("Enter the shape name: ")
while True:
    line = input()
    if line:
        lines.append(line)
    else:
        break;

# Final input string
user_shape = '\n'.join(lines)

# Calling the function.
getContours(imgContour,user_shape)

# Displaying the image.
cv2.imshow("Output", imgContour)
cv2.waitKey(0)


# Below function can be used for verifying the output.

# def verifyShapes():
#     shapes = ['trapezium', 'Circle', 'oval or\nellipse', 'square', 'triangle', 'rectangle', 'pentagon', 'hexagon',
#               'heptagon', 'octagon']
#     for i in range(len(shapes)):
#         print("Shape: ", shapes[i])
#         getContours(imgContour, shapes[i])
#         cv2.imshow("Output", imgContour)
#         cv2.waitKey(0)
#
#
# verifyShapes()



