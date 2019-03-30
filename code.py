'''
PCA for Face Recognition
'''


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import copy




## Generating Training image database

T = np.zeros([36000,20])
for i in range(1,21):
    img=cv2.imread('TrainDatabase/' + str(i) + '.jpg',0)
    img=np.ravel(img)
    T[:,i-1]=img

M=np.zeros([36000,1])    
for i in range(36000):
    M[i,0]=np.mean(T[i,:])

## Mean subtraction

A=np.zeros([36000,20])   
for i in range(20):
    A[:,i]=T[:,i]-M[:,0]
    
## Finding Evalues and Evectors

L=np.matmul(np.transpose(A),A)
D,V = np.linalg.eig(L)

eig_faces=np.matmul(A,np.transpose(V))

# Projecting training images onto the eigen faces

projected_images=np.zeros([20,20])

for i in range(20):
    projected_images[:,i]=np.matmul(np.transpose(eig_faces),A[:,i])
    
## Creating vectorized test image matrix

Test=np.zeros([36000,10])

for i in range(1,11):
    img=cv2.imread('TestDatabase/' + str(i) + '.jpg',0)
    img=np.ravel(img)
    Test[:,i-1]=img

## Choose a test sample by giving a column number '4' in this example
    
test=Test[:,4]

men_sub=test-M[:,0]

projected_test_image=np.matmul(np.transpose(eig_faces),men_sub)

## creating difference matrix

diff=np.zeros([20,20])
for i in range(20):
    diff[:,i]=projected_test_image[:]-projected_images[:,i]

## Creating a list of eucledian distatnces
    
euc_dist=np.zeros(20)    
for i in range(20):
    euc_dist[i]=np.linalg.norm(diff[:,i],2)
    
## Recognizing the face 
    
n = np.argmin(euc_dist)+1

## n corresponds to the index number in the training images of the 
# recognized face

