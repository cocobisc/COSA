import IPython
import json, os, re, sys, time
import cv2
import numpy as np
import imutils
from keras import Model
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
import keras
from keras.models import Model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    orig = cv2.imread(img_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds, orig

if __name__ == '__main__':
    classf = open("class.txt", "r")
    classes = []
    lines = classf.readlines()

    for line in lines:
        classes.append(line[:-1])
    # print(classes)
    classf.close()
    # print(cv2.imread('test/monkeys/n903.jpg'))
    model_path = 'asdfsdaf.h5'
    print('Loading model:', model_path)
    t0 = time.time()
    model = load_model(model_path)
    t1 = time.time()
    print('Loaded in:', t1-t0)
    print(model.input_shape[1])
    print(model.input_shape[2])
    test_path = 'test/monkeys/n003.jpg'
    print('Generating predictions on image:', test_path)
    preds,orig = predict(test_path, model)
    print(orig)
    label = classes[np.argmax(preds, axis=1)[0]]
    proba = np.max(preds)
    label = "{}: {:.2f}%".format(label, proba * 100)
    print(orig.shape)
    cv2.putText(orig, label, (int(orig.shape[1]*0.1), int(orig.shape[0]*0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 0, 255), 4)
    cv2.rectangle(orig, (0,0), (orig.shape[1], orig.shape[0]), (255, 0, 255), 10)
    cv2.imwrite("labeled image.jpg", orig)
    cv2.imshow("Output", orig)
    cv2.waitKey(0)