#!/usr/bin/env python

import device_patches       # Device specific patches for Jetson Nano (needs to be before importing cv2)

import cv2
import os
import sys, getopt
import signal
import time
from edge_impulse_linux.image import ImageImpulseRunner

runner = None

def now():
    return round(time.time() * 1000)

def get_webcams():
    port_ids = []
    for port in range(5):
        print("Looking for a camera in port %s:" %port)
        camera = cv2.VideoCapture(port)
        if camera.isOpened():
            ret = camera.read()[0]
            if ret:
                backendName =camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) found in port %s " %(backendName,h,w, port))
                port_ids.append(port)
            camera.release()
    return port_ids

def sigint_handler(sig, frame):
    print('Interrupted')
    if (runner):
        runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

def help():
    print('python classify.py <path_to_model.eim> <Camera port ID, only required when more than 1 camera is present>')

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

    if len(args) == 0:
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
            labels = model_info['model_parameters']['labels']
            if len(args)>= 2:
                videoCaptureDeviceId = int(args[1])
            else:
                port_ids = get_webcams()
                if len(port_ids) == 0:
                    raise Exception('Cannot find any webcams')
                if len(args)<= 1 and len(port_ids)> 1:
                    raise Exception("Multiple cameras found. Add the camera port ID as a second argument to use to this script")
                videoCaptureDeviceId = int(port_ids[0])

            camera = cv2.VideoCapture(videoCaptureDeviceId)
            ret = camera.read()[0]
            if ret:
                backendName = camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) in port %s selected." %(backendName,h,w, videoCaptureDeviceId))
                camera.release()
            else:
                raise Exception("Couldn't initialize selected camera.")

            next_frame = 0 # limit to ~10 fps here

            for img in runner.get_frames(videoCaptureDeviceId):
                if (next_frame > now()):
                    time.sleep((next_frame - now()) / 1000)

                # make two cuts from the image, one on the left and one on the right
                features_l, cropped_l = runner.get_features_from_image(img, 'left')
                features_r, cropped_r = runner.get_features_from_image(img, 'right')

                # classify both
                res_l = runner.classify(features_l)
                res_r = runner.classify(features_r)

                cv2.imwrite('debug_l.jpg', cv2.cvtColor(cropped_l, cv2.COLOR_RGB2BGR))
                cv2.imwrite('debug_r.jpg', cv2.cvtColor(cropped_r, cv2.COLOR_RGB2BGR))

                def print_classification(res, tag):
                    if "classification" in res["result"].keys():
                        print('%s: Result (%d ms.) ' % (tag, res['timing']['dsp'] + res['timing']['classification']), end='')
                        for label in labels:
                            score = res['result']['classification'][label]
                            print('%s: %.2f\t' % (label, score), end='')
                        print('', flush=True)
                    elif "bounding_boxes" in res["result"].keys():
                        print('%s: Found %d bounding boxes (%d ms.)' % (tag, len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
                        for bb in res["result"]["bounding_boxes"]:
                            print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))

                    if "visual_anomaly_grid" in res["result"].keys():
                        print('Found %d visual anomalies (%d ms.)' % (len(res["result"]["visual_anomaly_grid"]), res['timing']['dsp'] + res['timing']['classification']))
                        for grid_cell in res["result"]["visual_anomaly_grid"]:
                            print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (grid_cell['label'], grid_cell['value'], grid_cell['x'], grid_cell['y'], grid_cell['width'], grid_cell['height']))

                print_classification(res_l, 'LEFT')
                print_classification(res_r, 'RIGHT')

                next_frame = now() + 100

        finally:
            if (runner):
                runner.stop()

if __name__ == "__main__":
   main(sys.argv[1:])
