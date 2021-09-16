import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from keras.models import load_model
from tensorflow.keras.preprocessing import image


imgurl = './inputs/images.jpg'
img = cv2.imread(imgurl)
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
(l,w)= img2.shape
height5= int( 1000 /30)
wd5 = int(1000/20)
img2 = cv2.resize(img2,(1000,1000))
arr = np.uint8(img)
plt.imshow(img2,cmap='gray')
ret,thresh1 = cv2.threshold(img2,127,255,cv2.THRESH_BINARY)


scale = 1
delta = 0
ddepth = cv.CV_16S


grad_x = cv.Sobel(thresh1, ddepth, 1, 0, ksize=3, scale=scale, delta=delta,borderType=cv.BORDER_DEFAULT)
grad_y = cv.Sobel(thresh1, ddepth, 0, 1, ksize=3, scale=scale,delta=delta,borderType=cv.BORDER_DEFAULT)
    
    
abs_grad_x = cv.convertScaleAbs(grad_x)
abs_grad_y = cv.convertScaleAbs(grad_y)
    
    
grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

kernel = np.ones((5,5), np.uint8)
dil = cv2.dilate(grad, kernel, iterations=1)
ln ,wd = grad.shape
   
kernel = np.ones((6,6), np.uint8)
dil = cv2.dilate(grad, kernel, iterations=1)
plt.imshow(dil , cmap='gray')   
plt.plot()
    
img=dil.copy()
    
    
    

def findnextrowline(point):
    while(point<1000):
        for j in range(1000):
            if(img[point][j]>200):
                break
            elif(j+1 == 1000):
                return point
        point = point+1
    return 999
    

def firstcolumn_point():
    for i in range(1000):
        for j in range(1000):
            if(img[i][j]>200):
                return i

def updatepoint2(point):
    p=point
    while(p<1000):
        for j in range(1000):
            if(img[p][j]>240):
                return p
        p = p+1
    return 1000
        
    
        
rowpoints=[]   
p = firstcolumn_point()
rowpoints.append(p)
k=1
j=0
while(p < 999):
    p=p+ wd5
    update=findnextrowline(p)    
    rowpoints.append(update)
    p=updatepoint2(update)
    rowpoints.append(p)
    k=0

def updatepoint(point,f,s):
    p=point
    while(p<1000):
        for j in range(f,s):
            if(img[j][p]>240):
                return p
        p = p+1
    return 1000
def findnextcolline(point,f,s):
    while(point<1000):
        for j in range(f,s):
            if(img[j][point]>200):
                break
            elif(j+1 == s):
                return point
        point = point+1
    return 999

def findfirstpoint(f,s):
    for i in range(1000):
        ke=f+1
        while(ke<s+1):
            if(img[ke][i]>200):
                return i
            else:
                ke=ke+1
    return 0

k=0
finalcolpoints=[]
length = len(rowpoints)-2

for i in range(0,length,2):
    first=rowpoints[i]
    second=rowpoints[i+1]
    k = findfirstpoint(first,second)
    tempcol=[]
    tempcol.append(k)
    while(k<999):
        k = k+height5
        # update=find_end(first,second,k)
        update=findnextcolline(k,first,second)
        
        tempcol.append(update)   
        
        k=updatepoint(update,first,second)
        # k = upval(first,second,update)
#         if(k<999):
        tempcol.append(k)
    finalcolpoints.append(tempcol)

for i in range(1000):
    for j in range(1000):
        if(img[i][j]==255):
            img[i][j]=0
        else:
            img[i][j]=255

imgbefore = img.copy()  
cv2.imwrite('./helpers/predictionimg.png',imgbefore)
url2 = './helpers/predictionimg.png'
imgr = img.copy()
for i in rowpoints:
    img=cv2.line(img,(0,i),(999,i),0,4)
# print(rowpoints)

k=0
for i in range(0,length,2):
    first=rowpoints[i]
    if(i+1>=length):
        second=999
    else:
        second=rowpoints[i+1]
    
    tempcol=finalcolpoints[k]
    k=k+1
    for j in tempcol:
        cv2.line(img,(j,first),(j,second),0,2)


cv2.imwrite('./helpers/finaloutput.png',img)

def preprocesing(img1):
    kernal = np.ones((5,5),np.uint8)
    kl2 = np.ones((3,3),np.uint8)
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img1 = cv2.dilate(img1,kernal,iterations=2)
    img1 = cv2.erode(img1,kl2,iterations=1)
    img1 = cv2.cvtColor(img1 , cv2.COLOR_GRAY2BGR)
    img1=cv2.resize(img1,(28,28),interpolation=cv2.INTER_AREA)
    return img1


#-------------classifying-------------

model = load_model('Mymodel.h5')

alphabets=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
imgfirst = cv2.imread(url2)
text = []
for i,j in zip(range(0,len(rowpoints),2),finalcolpoints):
    p1 = rowpoints[i]
    p2=rowpoints[i+1]
    rowtext=[]
    for k in  range(0,len(j)-1,2):
        p3 = j[k]
        p4 = j[k+1]
        imgpredict = imgfirst[p1:p2,p3:p4]
        imgr = imgpredict.copy()
        img4= preprocesing(imgpredict)
        x=image.img_to_array(img4)
        x = np.expand_dims(x,axis=0)
        model.predict(x)
        prediction=np.argmax(model.predict(x),axis=1)
        rowtext.append(alphabets[prediction[0]])
    text.append(rowtext)


for i,n in zip(range(len(text)),text) :
    print(i,n)

cv2.imwrite('test.png',imgr)