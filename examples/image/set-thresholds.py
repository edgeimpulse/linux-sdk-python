#!/usr/bin/env python

import device_patches       # Device specific patches for Jetson Nano (needs to be before importing cv2)  # noqa: F401

try:
    import cv2
except ImportError:
    print('Missing OpenCV, install via `pip3 install "opencv-python>=4.5.1.48,<5"`')
    exit(1)
import os
import sys
import getopt
import json
from edge_impulse_linux.image import ImageImpulseRunner

runner = None

def help():
    print('python set-thresholds.py <path_to_model.eim> <path_to_image.jpg>')

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
            sys.exit()

    if len(args) != 2:
        help()
        sys.exit(2)

    model = args[0]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    print('MODEL: ' + modelfile)

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            # model_info = runner.init(debug=True) # to get debug print out

            print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
            if not 'thresholds' in model_info['model_parameters']:
                print('This model does not expose any thresholds, build a new Linux deployment (.eim file) to get configurable thresholds')
                exit(1)

            print('Thresholds:')
            for threshold in model_info['model_parameters']['thresholds']:
                print('    -', json.dumps(threshold))

            # Example output for an object detection model:
            # Thresholds:
            #    - {"id": 3, "min_score": 0.20000000298023224, "type": "object_detection"}

            img = cv2.imread(args[1])
            if img is None:
                print('Failed to load image', args[1])
                exit(1)

            # imread returns images in BGR format, so we need to convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # this mode uses the same settings used in studio to crop and resize the input
            features, cropped = runner.get_features_from_image_auto_studio_settings(img)

            print("Which threshold would you like to change? (id)")
            while True:
                try:
                    threshold_id = int(input('Enter threshold ID: '))
                    if threshold_id not in [t['id'] for t in model_info['model_parameters']['thresholds']]:
                        print('Invalid threshold ID, try again')
                        continue
                    break
                except ValueError:
                    print('Invalid input, please enter a number')

            print("Enter a new threshold value (between 0.0 and 1.0):")
            while True:
                try:
                    new_threshold = float(input('New threshold value: '))
                    if new_threshold < 0.0 or new_threshold > 1.0:
                        print('Invalid threshold value, must be between 0.0 and 1.0')
                        continue
                    break
                except ValueError:
                    print('Invalid input, please enter a number')

            # dynamically override the thresold from 0.2 -> 0.8
            runner.set_threshold({
                'id': threshold_id,
                'min_score': new_threshold,
            })

            res = runner.classify(features)
            print('classify response', json.dumps(res, indent=4))

        finally:
            if (runner):
                runner.stop()

if __name__ == "__main__":
   main(sys.argv[1:])
