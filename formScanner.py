# import cv2
# import pytesseract
# import numpy as np

# import os

# percentage=25

# pixelThreshold=500

# roi=[[(100, 972), (676, 1072), 'text', 'Name'],
#     [(744, 972), (1332, 1068), 'text', 'Phone'],
#     [(100, 1412), (688, 1508), 'text', 'Email'],
#     [(740, 1412), (1328, 1516), 'text', 'Id'],
#     [(100, 1584), (684, 1688), 'text', 'City'],
#     [(744, 1576), (1340, 1684), 'text', 'Country'],
#     [(744, 1144), (796, 1200), 'box', 'allergic'],
#     [(100, 1152), (160, 1200), 'box', ' decide']]


# pytesseract.pytesseract.tesseract_cmd='c:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# imgQ=cv2.imread('query.png')
# h,w,c=imgQ.shape

# ## ORB-> Oriented FAST and  Rotated BRIEF 
# orb=cv2.ORB_create(1000)
# ##kp->key points unique points or elements in img
# ##des1->descripters are the representation of keys points 
# kp1,des1=orb.detectAndCompute(imgQ,None)
# # imgKp1=cv2.drawKeypoints(imgQ,kp1,None)
# # print(des1)

# path='UserForms'
# myPicList=os.listdir(path)
# print(myPicList)

# for j,y in enumerate(myPicList):

#     img = cv2.imread(path +"/"+y)

#     kp2,des2 = orb.detectAndCompute(img,None)
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING)
#     matches = bf.match(des2,des1)
#     matches.sort(key= lambda x: x.distance)
#     good = matches[:int(len(matches)*(percentage/100))]
  
#     imgMatch = cv2.drawMatches(img,kp2,imgQ,kp1,good[:100],None,flags=2)
 

#     # #########################################################################
#     # item.distance: This attribute gives us the distance between the descriptors. A lower distance indicates a better match.
#     # item.trainIdx: This attribute gives us the index of the descriptor in the list of train descriptors (in our case, it’s the list of descriptors in the img2).
#     # item.queryIdx: This attribute gives us the index of the descriptor in the list of query descriptors (in our case, it’s the list of descriptors in the img1).
#     # item.imgIdx: This attribute gives us the index of the train image.
#     # #################################################################################

#     # print(good[0],kp2[good[0].queryIdx].pt,kp1[good[0].trainIdx].pt)
#     srcPoints=np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
#     dstPoints=np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)

#     # print(srcPoints)
#     M, _ = cv2.findHomography(srcPoints,dstPoints,cv2.RANSAC,5.0)
#     # print(M)
    
#     imgScan = cv2.warpPerspective(img,M,(w,h))
#     # imgScan = cv2.resize(imgScan, (w //4, h //4))

#     # cv2.imshow(y, imgScan)

#     imgShow=imgScan.copy()
#     imgMask=np.zeros_like(imgShow)

#     myData=[]
#     print(f'extracting data from form {j}')

#     for x,r in enumerate(roi):
#         cv2.rectangle(imgMask,(r[0][0],r[0][1]),(r[1][0],r[1][1]),(0,255,0),cv2.FILLED)
#         imgShow=cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)

#         imgCrop=imgScan[r[0][1]:r[1][1],r[0][0]:r[1][0]]
#         # cv2.imshow(str(r[3]),imgCrop)

#         if str(r[2])==str('text'):
#             s=pytesseract.image_to_string(imgCrop).replace("\n","")
#             s=s.replace("\x0c","")
#             print(f'{r[3]}:{pytesseract.image_to_string(imgCrop)}')
#             myData.append(s)

#         if r[2]=='box':
#             imgGray=cv2.cvtColor(imgCrop,cv2.COLOR_BGR2GRAY)
#             ###inverse ->bright region gives zero and dark region gives one
#             imgThresh=cv2.threshold(imgGray,170,255,cv2.THRESH_BINARY_INV)[1]
#             totalPixels=cv2.countNonZero(imgThresh)
#             # print(r[3],totalPixels)
#             if totalPixels>pixelThreshold: totalPixels=1
#             else: totalPixels=0
#             print(f'{r[3]}:{totalPixels}')
#             myData.append(totalPixels)

#         cv2.putText(imgShow,str(myData[x]),(r[0][0],r[0][1]),
#             cv2.FONT_HERSHEY_PLAIN,2.5,(0,0,255),4)

#     with open('dataOutput.csv','a+') as f:
#         for data in myData:
#             f.write((str(data)+','))
#         f.write('\n')
    
#     imgShow = cv2.resize(imgShow, (w //4, h //4))
#     print(myData)

# cv2.waitKey(0)




import cv2
import pytesseract
import numpy as np
from regionSelector import *

import os

percentage=25

pixelThreshold=500

obj=ROI()

roi=obj.RI()


pytesseract.pytesseract.tesseract_cmd='c:\\Program Files\\Tesseract-OCR\\tesseract.exe'

imgQ=cv2.imread('query.png')
h,w,c=imgQ.shape

## ORB-> Oriented FAST and  Rotated BRIEF 
orb=cv2.ORB_create(1000)
##kp->key points unique points or elements in img
##des1->descripters are the representation of keys points 
kp1,des1=orb.detectAndCompute(imgQ,None)
# imgKp1=cv2.drawKeypoints(imgQ,kp1,None)
# print(des1)

path='UserForms'
myPicList=os.listdir(path)
print(myPicList)

for j,y in enumerate(myPicList):

    img = cv2.imread(path +"/"+y)

    kp2,des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2,des1)
    matches.sort(key= lambda x: x.distance)
    good = matches[:int(len(matches)*(percentage/100))]
  
    imgMatch = cv2.drawMatches(img,kp2,imgQ,kp1,good[:100],None,flags=2)
 

    # #########################################################################
    # item.distance: This attribute gives us the distance between the descriptors. A lower distance indicates a better match.
    # item.trainIdx: This attribute gives us the index of the descriptor in the list of train descriptors (in our case, it’s the list of descriptors in the img2).
    # item.queryIdx: This attribute gives us the index of the descriptor in the list of query descriptors (in our case, it’s the list of descriptors in the img1).
    # item.imgIdx: This attribute gives us the index of the train image.
    # #################################################################################

    # print(good[0],kp2[good[0].queryIdx].pt,kp1[good[0].trainIdx].pt)
    srcPoints=np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dstPoints=np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    # print(srcPoints)
    M, _ = cv2.findHomography(srcPoints,dstPoints,cv2.RANSAC,5.0)
    # print(M)
    
    imgScan = cv2.warpPerspective(img,M,(w,h))
    # imgScan = cv2.resize(imgScan, (w //4, h //4))

    # cv2.imshow(y, imgScan)

    imgShow=imgScan.copy()
    imgMask=np.zeros_like(imgShow)

    myData=[]
    print(f'extracting data from form {j}')

    for x,r in enumerate(roi):

        imgCrop=imgScan[r[0][1]:r[1][1],r[0][0]:r[1][0]]

        if str(r[2])==str('text'):
            s=pytesseract.image_to_string(imgCrop).replace("\n","")
            s=s.replace("\x0c","")
            print(f'{r[3]}:{pytesseract.image_to_string(imgCrop)}')
            myData.append(s)

        if r[2]=='box':
            imgGray=cv2.cvtColor(imgCrop,cv2.COLOR_BGR2GRAY)
            ###inverse ->bright region gives zero and dark region gives one
            imgThresh=cv2.threshold(imgGray,170,255,cv2.THRESH_BINARY_INV)[1]
            totalPixels=cv2.countNonZero(imgThresh)
            # print(r[3],totalPixels)
            if totalPixels>pixelThreshold: totalPixels=1
            else: totalPixels=0
            print(f'{r[3]}:{totalPixels}')
            myData.append(totalPixels)

        cv2.putText(imgShow,str(myData[x]),(r[0][0],r[0][1]),
            cv2.FONT_HERSHEY_PLAIN,2.5,(0,0,255),4)

    with open('dataOutput.csv','a+') as f:
        for data in myData:
            f.write((str(data)+','))
        f.write('\n')
    
    imgShow = cv2.resize(imgShow, (w //4, h //4))
    cv2.imshow('final',imgShow)
    print(myData)

cv2.waitKey(0)
