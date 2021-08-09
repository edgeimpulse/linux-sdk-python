#!/usr/bin/env python

import numpy as np
import cv2
from edge_impulse_linux.runner import ImpulseRunner
import math
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

    # This returns images in RGB format (not BGR)
    def get_frames(self, videoDeviceId = 0):
        if psutil.OSX or psutil.MACOS:
            print('Make sure to grant the this script access to your webcam.')
            print('If your webcam is not responding, try running "tccutil reset Camera" to reset the camera access privileges.')

        self.videoCapture = cv2.VideoCapture(videoDeviceId)
        while not self.closed:
            success, img = self.videoCapture.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if success:
                yield img

    # This returns images in RGB format (not BGR)
    def classifier(self, videoDeviceId = 0):
        if psutil.OSX or psutil.MACOS:
            print('Make sure to grant the this script access to your webcam.')
            print('If your webcam is not responding, try running "tccutil reset Camera" to reset the camera access privileges.')

        self.videoCapture = cv2.VideoCapture(videoDeviceId)
        while not self.closed:
            success, img = self.videoCapture.read()
            if success:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                features, cropped = self.get_features_from_image(img)

                res = self.classify(features)
                yield res, cropped

    # This expects images in RGB format (not BGR)
    def get_features_from_image(self, img, crop_direction_x='center', crop_direction_y='center'):
        features = []

        EI_CLASSIFIER_INPUT_WIDTH = self.dim[0]
        EI_CLASSIFIER_INPUT_HEIGHT = self.dim[1]

        in_frame_cols = img.shape[1]
        in_frame_rows = img.shape[0]

        factor_w = EI_CLASSIFIER_INPUT_WIDTH / in_frame_cols
        factor_h = EI_CLASSIFIER_INPUT_HEIGHT / in_frame_rows

        largest_factor = factor_w if factor_w > factor_h else factor_h

        resize_size_w = int(math.ceil(largest_factor * in_frame_cols))
        resize_size_h = int(math.ceil(largest_factor * in_frame_rows))
        resize_size = (resize_size_w, resize_size_h)

        resized = cv2.resize(img, resize_size, interpolation = cv2.INTER_AREA)

        if (crop_direction_x == 'center'):
            crop_x = int((resize_size_w - resize_size_h) / 2) if resize_size_w > resize_size_h else 0
        elif (crop_direction_x == 'left'):
            crop_x = 0
        elif (crop_direction_x == 'right'):
            crop_x = resize_size_w - EI_CLASSIFIER_INPUT_WIDTH
        else:
            raise Exception('Invalid value for crop_direction_x, should be center, left or right')

        if (crop_direction_y == 'center'):
            crop_y = int((resize_size_h - resize_size_w) / 2) if resize_size_h > resize_size_w else 0
        elif (crop_direction_y == 'top'):
            crop_y = 0
        elif (crop_direction_y == 'bottom'):
            crop_y = resize_size_h - EI_CLASSIFIER_INPUT_HEIGHT
        else:
            raise Exception('Invalid value for crop_direction_y, should be center, top or bottom')

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

        return features, cropped
