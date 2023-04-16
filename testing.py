## importing necessary libraries and modules
import cv2
import numpy as np
from keras.models import load_model

## loading the model 
model=load_model("C:/Users/eTraders/model2-010.model")

## Creating dictionaries 
labels_dict={0:'without mask',1:'mask'} ## to map the prediction to labels
color_dict={0:(0,0,255),1:(0,255,0)} ## to assign different colors to the bounding boxes around faces 
#                                       red for w/o mask and green for mask


size = 4 ## resize the image for faster face detection
webcam = cv2.VideoCapture(0) # Initialized to capture video frames from the cam

## Loading the pre-trained haar cascade classifier for face detection
classifier = cv2.CascadeClassifier('C:/Users/eTraders/PycharmProjects/open-cv/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

## loop to continuously capture video frames and process them
while True:
    (rval, im) = webcam.read() ## for frame reading
    im=cv2.flip(im,1,1) ## for flipping to act as a mirror

    ## Resize the image for faster detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    ## detect faces and return list of bounding boxes around the faces
    faces = classifier.detectMultiScale(mini)

    ## Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        ## Extracting the ROI from the original frame
        face_img = im[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(150,150))
        normalized=resized/255.0 ## to make values lie between 0 and 1
        reshaped=np.reshape(normalized,(1,150,150,3)) ## reshape to match the input shape expected by CNN model
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        
        ## obtaining the predicted class label
        label=np.argmax(result,axis=1)[0]
      
        ## drawing bounding box around detected face with a thickness of pixel 
        cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)

        ## drawing filled rect on top of bounding box
        cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)

        ## displaying the predicted label string
        cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    ## Show the image
    cv2.imshow('LIVE',   im)
    key = cv2.waitKey(10)

    ## if Esc key is press then break out of the loop 
    if key == 27: #The Esc key
        break

## Stop video
webcam.release()

## Close all started windows
cv2.destroyAllWindows()