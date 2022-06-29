
import cv2
import pytesseract
import numpy as np
from regionSelector import *
from csv import DictWriter
import csv 

import os

class Scan:
    def __init__(self,qP,dP):
        self.queryPath=qP
        self.dataPath=dP
        self.percentage=25
        self.pixelThreshold=500

    def scanForm(self):
        pytesseract.pytesseract.tesseract_cmd='c:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        imgQ=cv2.imread(self.queryPath)
        h,w,c=imgQ.shape

        obj=ROI(self.queryPath)

        roi=obj.RI()
        print(roi)
        with open('dataOutput.csv','a+',newline='') as f:
            fields=[i[3] for i in roi]
            csvWriter=DictWriter(f,fieldnames=fields)                   
            csvWriter.writeheader()
        ## ORB-> Oriented FAST and  Rotated BRIEF 
        orb=cv2.ORB_create(1000)
        ##kp->key points unique points or elements in img
        ##des1->descripters are the representation of keys points 
        kp1,des1=orb.detectAndCompute(imgQ,None)
        # imgKp1=cv2.drawKeypoints(imgQ,kp1,None)
        # print(des1)

        path=self.dataPath
        myPicList=os.listdir(path)
        print(myPicList)

        for j,y in enumerate(myPicList):

            img = cv2.imread(path +"/"+y)

            kp2,des2 = orb.detectAndCompute(img,None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.match(des2,des1)
            matches.sort(key= lambda x: x.distance)
            good = matches[:int(len(matches)*(self.percentage/100))]
        
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
                    print(f'{r[3]}:{s}')
                    myData.append(s)

                if r[2]=='box':
                    imgGray=cv2.cvtColor(imgCrop,cv2.COLOR_BGR2GRAY)
                    ###inverse ->bright region gives zero and dark region gives one
                    imgThresh=cv2.threshold(imgGray,170,255,cv2.THRESH_BINARY_INV)[1]
                    totalPixels=cv2.countNonZero(imgThresh)
                    # print(r[3],totalPixels)
                    if totalPixels>self.pixelThreshold: totalPixels=1
                    else: totalPixels=0
                    print(f'{r[3]}:{totalPixels}')
                    myData.append(totalPixels)

                cv2.putText(imgShow,str(myData[x]),(r[0][0],r[0][1]),
                    cv2.FONT_HERSHEY_PLAIN,2.5,(0,0,255),4)

            with open('dataOutput.csv','a+',newline='') as f:
                fields=[i[3] for i in roi]
                csvWriter=DictWriter(f,fieldnames=fields)
                print('my data is ',myData)
                dic={}
                for i,d in enumerate(myData):
                    dic[fields[i]]=d 
                csvWriter.writerow(dic)
 
            
            imgShow = cv2.resize(imgShow, (w //4, h //4))
            cv2.imshow('final',imgShow)
            print(myData)

        cv2.waitKey(0)
