# A simple code demonstrating the use of PCA for face recognition
1. Make sure you have two folders named 'TrainDatabase' and 'TestDatabase' included in the same directory as the one where the python code file is present - "code.py"
2. The 'TrainDatabase' and 'TestDatabase' folders should contain the training images and the test images of the faces respectively. They should be named as consecutive integers (For eg: 1.jpg, 2.jpg  etc) 
3. In line 58 of the code which is 
   test=Test[:,4]
 '4' means that we wish to run facial recognition on the image named '4' in the 'TestDatabase' folder
  You can change this as per your requirement.
4. The final result will be stored in the variable 'n'; which is again an integer which corresponds to the face in the 'TrainDatabase' folder named by that particular number 'n'.
