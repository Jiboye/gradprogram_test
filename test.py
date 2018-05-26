import numpy as np
import cv2
from matplotlib import pyplot as plt

image = cv2.imread('test1.jpg')
image2 = cv2.imread('test2.jpg')
if (image.shape[0] > 1000):
	small = cv2.resize(image,(0,0),fx = 0.25,fy = 1)
	if(image.shape[1] > 1000):
		small = cv2.resize(image,(0,0),fx = 0.25,fy = 0.25)

height = small.shape[0]
width = small.shape[1] 
x1 = (int)(width - width*0.9)
y1 = (int)(height - height * 0.85)
x2 = (int)(width - width * 0.1)
y2 = (int)(height - height * 0.01)

cv2.rectangle(small,((x1),(y1)),((x2),(y2) ),(0,0,255),2)
human = small[y1:y2,x1:x2]
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

grayimg = cv2.cvtColor(human,cv2.COLOR_BGR2GRAY)
cv2.imshow('human',human)
blurred = cv2.blur(grayimg, (3, 3))
canny = cv2.Canny(blurred, 50, 255)

thresh = 170
im_bw = cv2.threshold(grayimg,thresh,255,cv2.THRESH_BINARY)[1]






low = np.array([15,200,150],dtype ="uint8")
up = np.array([32,229,183],dtype ="uint8")
mask = cv2.inRange(small,low,up)
cv2.imshow('mask',mask)

cv2.imshow("canny:", canny)
cv2.imwrite('canny.jpg',canny)

plt.subplot(121),plt.imshow(human,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(canny,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

cv2.imshow("small:", small)
cv2.waitKey(0)
cv2.destroyAllWindows()


