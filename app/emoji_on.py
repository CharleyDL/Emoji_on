#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Created By   : Charley ∆. Lebarbier
# Date Created : Thursday 22 Feb. 2023
# ==============================================================================
# Use the webcam stream to detect emotions and covers them with a corresponding 
# emoji.
# ==============================================================================


import cv2
import keras
import numpy as np
import tensorflow as tf


def mask_emotion(frame, x, y, w, h, emotion) -> None:
    """
    Get a frame from webcam and put an emoji on the face according to
    the emotion detected.
    @Params
        frame           - required : frame get from the webcam
        x               - required : x
        y               - required : y
        w               - required : width
        h               - required : heigth
        emotion         - required : emotion detected by DeepFace
    """

    ## -- Dict with emoji for each emotion
    emoji_dict = {
        'Angry': cv2.imread('../data/emoji/angry.png'),
        'Disgust': cv2.imread('../data/emoji/disgust.png'),
        'Fear': cv2.imread('../data/emoji/fear.png'),
        'Happy': cv2.imread('../data/emoji/happy.png'),
        'Neutral': cv2.imread('../data/emoji/neutral.png'),
        'Sad': cv2.imread('../data/emoji/sad.png'),
        'Surprize': cv2.imread('../data/emoji/surprise.png'),
    }

    emoji_mask = cv2.resize(emoji_dict[emotion], (w, h))
    ret, mask = cv2.threshold(emoji_mask, 1, 255, cv2.THRESH_BINARY)

    roi = frame[y:y+h, x:x+w]
    roi[np.where(mask)] = 0
    roi += emoji_mask


def emotion_detect(frame, model):
    """
    Use our pre-training prediction model to detect the emotion of a human 
    face at each frame.  
    @Params :  
        frame - required : take the frame from the webcam stream
        model - required : take the prediction model
    """

    ## -- Import face classifier
    face_class = cv2.CascadeClassifier(cv2.data.haarcascades\
        + 'haarcascade_frontalface_default.xml')

    ## -- Define emotion labels
    emotions_dict = { 0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad",
                      5:"Surprize", 6:"Neutral" }

    ## -- Initiate the Face detection parameters
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_class.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    ## -- Region of Interest (ROI) for the face
    for (x, y, w, h) in faces:
        try:
            ## -- Prepare the frame for the model
            roi_gray = gray[y:y+h, x:x+w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, 
                (48, 48)), -1), 0)
            cropped_img = cropped_img / 255.0    # Image Normalization

            ## -- Predict the emotion with model
            prediction = model.predict(cropped_img)
            analyze = emotions_dict[int(np.argmax(prediction))]     # Get emotion in the dict

            ## -- Display the emotion in text behind the head
            cv2.putText(frame, analyze, (x+35, y-50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 0), 2, cv2.LINE_AA)

            ## -- Translate into emotion word
            if analyze == 'Angry':
                frame = mask_emotion(frame, x, y, w, h, 'Angry')
            elif analyze == 'Disgust':
                frame = mask_emotion(frame, x, y, w, h, 'Disgust')
            elif analyze == 'Fear':
                frame = mask_emotion(frame, x, y, w, h, 'Fear')
            elif analyze == 'Happy':
                frame = mask_emotion(frame, x, y, w, h, 'Happy')
            elif analyze == 'Neutral':
                frame = mask_emotion(frame, x, y, w, h, 'Neutral')
            elif analyze == 'Sad':
                frame = mask_emotion(frame, x, y, w, h, 'Sad')
            elif analyze == 'Surprize':
                frame = mask_emotion(frame, x, y, w, h, 'Surprize')

        except Exception as e:
            print(e)


#### ---- MAIN PROGRAM ---- ####
if __name__ == '__main__':

    ## -- Import the model
    model = keras.models.load_model('../model_save/cnn_dataAug.h5')

    ## -- Activate Webcam and Get its stream
    cap = cv2.VideoCapture(0)

    ## -- Activate the program
    while True:
        ## -- Get frame
        ret, frame = cap.read()

        ## -- Detect the face and the emotion
        emotion_detect(frame, model)

        ## -- Add instruction for leave the program
        cv2.putText(frame, "'Q' to leave", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, (0, 0, 180), 1, cv2.LINE_4)

        ## -- Display the result
        cv2.imshow('Video', cv2.resize(frame, (800,600), interpolation=cv2.INTER_CUBIC))

        ## -- Leave the app
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ## -- Close the stream and window
    cap.release()
    cv2.destroyAllWindows()