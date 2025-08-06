#!/usr/bin/env python

import numpy as np
import sys
try:
    import cv2
except ImportError:
    print('Missing OpenCV, install via `pip3 install "opencv-python>=4.5.1.48,<5"`')
    exit(1)

from edge_impulse_linux.runner import ImpulseRunner
import math

class ImageImpulseRunner(ImpulseRunner):
    def __init__(self, model_path: str):
        super(ImageImpulseRunner, self).__init__(model_path)
        self.closed = True
        self.labels = []
        self.dim = (0, 0)
        self.videoCapture = cv2.VideoCapture()
        self.isGrayscale = False
        self.resizeMode = ''

    def init(self, debug=False):
        model_info = super(ImageImpulseRunner, self).init(debug)
        width = model_info['model_parameters']['image_input_width']
        height = model_info['model_parameters']['image_input_height']

        if width == 0 or height == 0:
            raise Exception('Model file "' + self._model_path + '" is not suitable for image recognition')

        self.dim = (width, height)
        self.labels = model_info['model_parameters']['labels']
        self.isGrayscale = model_info['model_parameters']['image_channel_count'] == 1
        self.resizeMode = model_info['model_parameters'].get('image_resize_mode', 'not-reported')
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
        if sys.platform == "darwin":
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
        if sys.platform == "darwin":
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

    # This expects images in RGB format (not BGR), DEPRECATED, use get_features_from_image_auto_studio_settings
    def get_features_from_image(self, img, crop_direction_x='center', crop_direction_y='center'):
        features = []

        EI_CLASSIFIER_INPUT_WIDTH = self.dim[0]
        EI_CLASSIFIER_INPUT_HEIGHT = self.dim[1]

        in_frame_cols = img.shape[1]
        in_frame_rows = img.shape[0]

        factor_w = EI_CLASSIFIER_INPUT_WIDTH / in_frame_cols
        factor_h = EI_CLASSIFIER_INPUT_HEIGHT / in_frame_rows

        # Maintain the same aspect ratio by scaling by the same factor for both dimensions
        largest_factor = factor_w if factor_w > factor_h else factor_h

        resize_size_w = int(math.ceil(largest_factor * in_frame_cols))
        resize_size_h = int(math.ceil(largest_factor * in_frame_rows))
        # One dim will match the classifier size, the other will be larger
        resize_size = (resize_size_w, resize_size_h)

        resized = cv2.resize(img, resize_size, interpolation=cv2.INTER_AREA)

        if (crop_direction_x == 'center'):
            crop_x = int((resize_size_w - EI_CLASSIFIER_INPUT_WIDTH) / 2)  # 0 when same
        elif (crop_direction_x == 'left'):
            crop_x = 0
        elif (crop_direction_x == 'right'):
            crop_x = resize_size_w - EI_CLASSIFIER_INPUT_WIDTH  # can't be negative b/c one size will match input and the other will be larger
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

        cropped = resized[crop_y: crop_y + EI_CLASSIFIER_INPUT_HEIGHT,
                          crop_x: crop_x + EI_CLASSIFIER_INPUT_WIDTH]

        if self.isGrayscale:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            pixels = np.array(cropped).flatten().tolist()

            for p in pixels:
                features.append((p << 16) + (p << 8) + p)
        else:
            pixels = np.array(cropped).flatten().tolist()

            for ix in range(0, len(pixels), 3):
                r = pixels[ix + 0]
                g = pixels[ix + 1]
                b = pixels[ix + 2]
                features.append((r << 16) + (g << 8) + b)

        return features, cropped

    def get_features_from_image_auto_studio_settings(self, img):
        if self.resizeMode == '':
            raise Exception(
                'Runner has not initialized, please call init() first')
        if self.resizeMode == 'not-reported':
            self.resizeMode = 'squash'
        return get_features_from_image_with_studio_mode(img, self.resizeMode, self.dim[0], self.dim[1], self.isGrayscale)


def resize_image(image, size):
    """Resize an image to the given size using a common interpolation method.

    Args:
        image: The input image as a NumPy array.
        size: A tuple (width, height) specifying the desired output size.

    Returns:
        The resized image as a NumPy array.
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def resize_with_letterbox(image, target_width, target_height):
    """Resize an image while maintaining aspect ratio using letterboxing.

    Args:
        image: The input image as a NumPy array.
        target_size: A tuple (width, height) specifying the desired output size.

    Returns:
        The resized image as a NumPy array and the letterbox dimensions.
    """

    height, width = image.shape[:2]

    # Calculate scale factors to preserve aspect ratio
    scale_x = target_width / width
    scale_y = target_height / height
    scale = min(scale_x, scale_y)

    # Calculate new dimensions and padding
    new_width = int(width * scale)
    new_height = int(height * scale)
    top_pad = (target_height - new_height) // 2
    bottom_pad = target_height - new_height - top_pad
    left_pad = (target_width - new_width) // 2
    right_pad = target_width - new_width - left_pad

    # Resize image and add padding
    resized_image = resize_image(image, (new_width, new_height))
    padded_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)

    return padded_image


def get_features_from_image_with_studio_mode(img, mode, output_width, output_height, is_grayscale):
    """
    Extract features from an image using different resizing modes suitable for Edge Impulse Studio.

    Args:
        img (numpy.ndarray): The input image as a NumPy array.
        mode (str): The resizing mode to use. Options are 'fit-shortest', 'fit-longest', and 'squash'.
        output_width (int): The desired output width of the image.
        output_height (int): The desired output height of the image.
        is_grayscale (bool): Whether the output image should be converted to grayscale.

    Returns:
        tuple: A tuple containing:
            - features (list): A list of pixel values in the format (R << 16) + (G << 8) + B for color images,
              or (P << 16) + (P << 8) + P for grayscale images.
            - resized_img (numpy.ndarray): The resized image as a NumPy array.
    """
    features = []

    in_frame_cols = img.shape[1]
    in_frame_rows = img.shape[0]

    if mode == 'fit-shortest':
        aspect_ratio = output_width / output_height
        if in_frame_cols / in_frame_rows > aspect_ratio:
            # Image is wider than target aspect ratio
            new_width = int(in_frame_rows * aspect_ratio)
            offset = (in_frame_cols - new_width) // 2
            cropped_img = img[:, offset:offset + new_width]
        else:
            # Image is taller than target aspect ratio
            new_height = int(in_frame_cols / aspect_ratio)
            offset = (in_frame_rows - new_height) // 2
            cropped_img = img[offset:offset + new_height, :]

        resized_img = cv2.resize(cropped_img, (output_width, output_height), interpolation=cv2.INTER_AREA)
    elif mode == 'fit-longest':
        resized_img = resize_with_letterbox(img, output_width, output_height)
    elif mode == 'squash':
        resized_img = resize_image(img, (output_width, output_height))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if is_grayscale:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        features = (resized_img.astype(np.uint32) * 0x010101).flatten().tolist()
    else:
        # Use numpy's vectorized operations for RGB feature encoding
        pixels = resized_img.astype(np.uint32)
        features = ((pixels[..., 0] << 16) | (pixels[..., 1] << 8) | pixels[..., 2]).flatten().tolist()

    return features, resized_img