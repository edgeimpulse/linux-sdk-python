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

                EI_CLASSIFIER_INPUT_WIDTH = self.dim[0]
                EI_CLASSIFIER_INPUT_HEIGHT = self.dim[1]

                in_frame_cols = img.shape[1]
                in_frame_rows = img.shape[0]

                factor_w = EI_CLASSIFIER_INPUT_WIDTH / in_frame_cols
                factor_h = EI_CLASSIFIER_INPUT_HEIGHT / in_frame_rows

                largest_factor = factor_w if factor_w > factor_h else factor_h

                resize_size_w = int(largest_factor * in_frame_cols)
                resize_size_h = int(largest_factor * in_frame_rows)
                resize_size = (resize_size_w, resize_size_h)

                resized = cv2.resize(img, resize_size, interpolation = cv2.INTER_AREA)

                crop_x = int((resize_size_w - resize_size_h) / 2) if resize_size_w > resize_size_h else 0
                crop_y = int((resize_size_h - resize_size_w) / 2) if resize_size_h > resize_size_w else 0

                crop_region = (crop_x, crop_y, EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT)

                cropped = resized[crop_region[1]:crop_region[1]+crop_region[3], crop_region[0]:crop_region[0]+crop_region[2]]

                if self.isGrayscale:
                    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    pixels = np.array(cropped).flatten().tolist()

                    for p in pixels:
                        features.append((p << 16) + (p << 8) + p)
                else:
                    pixels = np.array(cropped).flatten().tolist()

                    for ix in range(0, len(pixels), 3):
                        b = pixels[ix + 0]
                        g = pixels[ix + 1]
                        r = pixels[ix + 2]
                        features.append((r << 16) + (g << 8) + b)

                res = self.classify(features)
                yield res, cropped
