
######### Region Selector ###############################
"""
This script allows to collect raw points from an image.
The inputs are two mouse clicks one in the x,y position and
the second in w,h of a rectangle.
Once a rectangle is selected the user is asked to enter the type
and the Name:
Type can be 'Text' or 'CheckBox'
Name can be anything
"""

import cv2
import random
class ROI:
    def __init__(self,p):        
        self.path = p
        self.scale = 0.25
        self.circles = []
        self.counter = 0
        self.coutner2 = 0
        self.point1=[]
        self.point2=[]
        self.mypoints = []
        self.mycolor=[]

    def mousePoints(self,event,x,y,flags,params):
        self.counter,self.point1,self.point2,self.coutner2,self.circles,self.mycolor
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.counter==0:
                self.point1=int(x//self.scale),int(y//self.scale)
                self.counter +=1
                self.mycolor = (random.randint(0,2)*200,random.randint(0,2)*200,random.randint(0,2)*200 )
            elif self.counter ==1:
                self.point2=int(x//self.scale),int(y//self.scale)
                type = input('Enter Type ')
                name = input ('Enter Name ')
                self.mypoints.append([self.point1,self.point2,type,name])
                self.counter=0
            self.circles.append([x,y,self.mycolor])
            self.coutner2 += 1
    
    def RI(self):
        img = cv2.imread(self.path)
        img = cv2.resize(img, (0, 0), None, 0.25, 0.25)

        while True:
            # To Display points
            for x,y,color in self.circles:
                cv2.circle(img,(x,y),3,color,cv2.FILLED)
            cv2.imshow("Original Image", img)
            cv2.setMouseCallback("Original Image", self.mousePoints)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                return self.mypoints
                break

