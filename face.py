from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
import json
import math
import csv
import pandas as pd
import matplotlib.pyplot as plt

class Face:

    def __init__(self):
        print("TestFace object")
        pass

    def analysis(self, filename):
        # try:
            # initialize dlib's face detector (HOG-based) and then create
            # the facial landmark predictor
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            image = cv2.imread(filename)
            image = imutils.resize(image, width=500)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)
            for (i, rect) in enumerate(rects):
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                landmarks = predictor(gray, rect)

                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                file = filename.replace("(","")
                file = file.replace(")","")
                file = file.replace(" ","")
                f = "keypoints/"+file
                print(f)
                dict_data  = {"droop":"doop","filename":f} # droop / nodroop label
                csv_columns = ['droop','filename']
            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
                dict_data['x'+str(i)] = landmarks.part(i).x
                dict_data['y'+str(i)] = landmarks.part(i).y
                csv_columns.append('x'+str(i))
                csv_columns.append('y'+str(i))
       
            cv2.imwrite(f, image)
            # print(dict_data)
            print(filename)
            csv_file = "droop_dataset.csv"
            with open(csv_file, 'a+') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writerow(dict_data)
        # except Exception as e:
        #     print(e.args)

    def runAnalysis(self):
        for i in range(1,100):
            file = str("droop (" + str(i) +").jpg")
            # select images dir
            filename = os.path.join('droop/', '{}'.format(file))
            print(filename)
            analysis(filename)

face = Face()
face.runAnalysis()