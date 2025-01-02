from keras.layers import Input
from tensorflow.keras.optimizers import SGD
#from keras.optimizers import gradient_descent_v2
from model.slowfast import SlowFast_body, bottleneck
import cv2
import numpy as np
import os
import time
from PIL import Image
from PIL import ImageDraw
import telepot
bot = telepot.Bot('7058162745:AAGH8Z293zXor3p3aMFvwtG4SLPxmBCiwKA')

def resnet50(inputs, **kwargs):
    model = SlowFast_body(inputs, [3, 4, 6, 3], bottleneck, **kwargs)
    return model
font = cv2.FONT_HERSHEY_SIMPLEX
def frames_from_video(video_dir, nb_frames = 25, img_size = 224):

    # Opens the Video file
    cap = cv2.VideoCapture(video_dir)
    i=0
    frames = []
    while(cap.isOpened() and i<nb_frames):
       
        ret, frame = cap.read()
        
        if ret == False:
            break
        frame = cv2.resize(frame, (img_size, img_size))
        
        frames.append(frame)
        i+=1
        #cv2.putText(frame, 'TEXT ON VIDEO',(30, 30),font, 1,(0, 255, 255),2,cv2.LINE_4)
        #cv2.imshow('Action Recognition', frame)
        #cv2.waitKey(0)
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
        time.sleep(1)
    cap.release()
    cv2.destroyAllWindows()
    return np.array(frames) / 255.0

def predictions(video_dir, model, nb_frames = 25, img_size = 224):

    X = frames_from_video(video_dir, nb_frames, img_size)
    X = np.reshape(X, (1, nb_frames, img_size, img_size, 3))
    
    predictions = model.predict(X)
    preds = predictions.argmax(axis = 1)

    classes = []
    with open(os.path.join('output', 'classes.txt'), 'r') as fp:
        for line in fp:
            classes.append(line.split()[1])
    out=''
    for i in range(len(preds)):
        print('Prediction - {} -- {}'.format(preds[i], classes[preds[i]]))
        out=str(classes[preds[i]])
        bot.sendMessage('6556078091', str(classes[preds[i]]+' detected'))
    cap = cv2.VideoCapture(video_dir)
    i=0
    frames = []
    while(cap.isOpened() and i<nb_frames):
        ret, frame = cap.read()
        
        if ret == False:
            break
        frame = cv2.resize(frame, (img_size, img_size))
        
        frames.append(frame)
        i+=1
        cv2.putText(frame,str(out),(30, 30),font, 1,(0, 255, 255),2,cv2.LINE_4)
        pass
        
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
        time.sleep(1)
    cap.release()
    cv2.destroyAllWindows()

# Load the model with pre-configured parameters
x = Input(shape = (25, 224, 224, 3))
model = resnet50(x, num_classes=14)

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(learning_rate=0.01, momentum=0.9), 
              metrics=['accuracy'])

model.summary()

# We load the weights directly
model.load_weights("model_new.h5")
def analyse(fileName):  
    predictions(video_dir = fileName, model = model, nb_frames = 25, img_size = 224)
