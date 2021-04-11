#!/usr/bin/env python

import numpy as np
import cv2
from edge_impulse_linux.runner import ImpulseRunner
import time
import psutil

class ImageImpulseRunner(ImpulseRunner):
    def __init__(self, model_path: str):
        super(ImageImpulseRunner, self).__init__(model_path)
        self.closed = True
        self.labels = []
        self.dim = (0, 0)
        self.videoCapture = cv2.VideoCapture()
        self.isGrayscale = False

    def init(self):
        if psutil.OSX or psutil.MACOS:
            print('Make sure that video devices access is granted for your application. runnin')
            print('If your video device is not responding, try running "tccutil reset Camera" to reset the camera access privileges')

        model_info = super(ImageImpulseRunner, self).init()
        width = model_info['model_parameters']['image_input_width'];
        height = model_info['model_parameters']['image_input_height'];

        if width == 0 or height == 0:
            raise Exception('Model file "' + self._model_path + '" is not suitable for image recognition')

        self.dim = (width, height)
        self.labels = model_info['model_parameters']['labels']
        self.isGrayscale =  model_info['model_parameters']['image_channel_count'] == 1
        return model_info

    def __enter__(self):
        self.videoCapture = cv2.VideoCapture()
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self.videoCapture.release()
        self.closed = True

    def classify(self, data):
        return super(ImageImpulseRunner, self).classify(data)

    def classifier(self, videoDeviceId = 0):
        self.videoCapture = cv2.VideoCapture(videoDeviceId)
        while not self.closed:
            success, img = self.videoCapture.read()
            if success:
                features = []
                if self.isGrayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    resizedImg = cv2.resize(img, self.dim, interpolation = cv2.INTER_AREA)
                    pixels = np.array(resizedImg).flatten().tolist()

                    for p in pixels:
                        features.append((p << 16) + (p << 8) + p)
                else:
                    resizedImg = cv2.resize(img, self.dim, interpolation = cv2.INTER_AREA)
                    pixels = np.array(resizedImg).flatten().tolist()

                    for ix in range(0, len(pixels), 3):
                        b = pixels[ix + 0]
                        g = pixels[ix + 1]
                        r = pixels[ix + 2]
                        features.append((r << 16) + (g << 8) + b)

                res = self.classify(features)
                yield res, img
