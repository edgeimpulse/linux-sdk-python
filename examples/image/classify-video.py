#!/usr/bin/env python

import device_patches       # Device specific patches for Jetson Nano (needs to be before importing cv2)

import cv2
import os
import time
import sys, getopt
import numpy as np
from edge_impulse_linux.image import ImageImpulseRunner

runner = None
# if you don't want to see a video preview, set this to False
show_camera = True

if (sys.platform == 'linux' and not os.environ.get('DISPLAY')):
    show_camera = False


def help():
    print('python classify-video.py <path_to_model.eim> <path_to_video.mp4> [frame_interval]')

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

    if len(args) < 2 or len(args) > 3:
        help()
        sys.exit(2)

    model = args[0]
    video_file = args[1]


    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)
    print('MODEL: ' + modelfile)

    # jump 1 frame at a time by default
    frame_interval = 1

    if len(args) == 3:
        try:
            frame_interval = int(args[2])
        except ValueError:
            frame_interval = None
    print('Stepping through %d frames at a time' % frame_interval)


    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
            labels = model_info['model_parameters']['labels']

            vidcap = cv2.VideoCapture(video_file)
            #fps = vidcap.get(cv2.CAP_PROP_FPS)
            #total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
            #total_duration = total_frames / fps
            #print("File information: FPS: %.2f Frames %d, duration: %.2f" % (fps, total_frames, total_duration))

            frame_nr = 0
            start_time = time.time()

            def getFrame(id):
                #debug_time = time.time()
                #print("frame id: %d" % id)
                vidcap.set(cv2.CAP_PROP_POS_FRAMES,id)

                hasFrames,image = vidcap.read()
                if hasFrames:
                    #print("get frame took: %.2f sec" % (time.time() - debug_time))
                    return image
                else:
                    print('Failed to load frame', video_file)
                    exit(1)



            img = getFrame(frame_nr)

            while img.size != 0:

                # imread returns images in BGR format, so we need to convert to RGB
                #debug_time  = time.time()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #print("color convert took: %.2f sec" % (time.time() - debug_time))

                # get_features_from_image also takes a crop direction arguments in case you don't have square images
                #debug_time  = time.time()
                features, cropped = runner.get_features_from_image(img)
                #print("get features took: %.2f sec" % (time.time() - debug_time))

                # the image will be resized and cropped, save a copy of the picture here
                # so you can see what's being passed into the classifier
                #cv2.imwrite('debug.jpg', cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

                res = runner.classify(features)

                #debug_time  = time.time()
                if "classification" in res["result"].keys():
                    print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
                    for label in labels:
                        score = res['result']['classification'][label]
                        print('%s: %.2f\t' % (label, score), end='')
                    print('', flush=True)

                elif "bounding_boxes" in res["result"].keys():
                    print('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
                    for bb in res["result"]["bounding_boxes"]:
                        print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))
                        img = cv2.rectangle(cropped, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (255, 0, 0), 1)
                #print("process results took: %.2f sec" % (time.time() - debug_time))

                if (show_camera):
                    #debug_time = time.time()
                    cv2.imshow('edgeimpulse', cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) == ord('q'):
                        break
                    #print("show results took: %.2f sec" % (time.time() - debug_time))


                frame_nr += frame_interval
                img = getFrame(frame_nr)
        finally:
            if (runner):
                runner.stop()

if __name__ == "__main__":
   main(sys.argv[1:])
