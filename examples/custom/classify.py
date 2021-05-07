import os
import sys, getopt
import signal
import time
from edge_impulse_linux.runner import ImpulseRunner
import io

runner = None

def signal_handler(sig, frame):
    print('Interrupted')
    if (runner):
        runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def help():
    print('python classify.py <path_to_model.eim> <path_to_features.txt>')

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

    if len(args) <= 1:
        help()
        sys.exit(2)

    model = args[0]


    features_file = io.open(args[1], 'r', encoding='utf8')
    features = features_file.read().strip().split(",")
    if '0x' in features[0]:
        features = [float(int(f, 16)) for f in features]
    else:
        features = [float(f) for f in features]


    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    print('MODEL: ' + modelfile)


    runner = ImpulseRunner(modelfile)
    try:
        model_info = runner.init()
        print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')

        res = runner.classify(features)
        print("classification:")
        print(res["result"])
        print("timing:")
        print(res["timing"])

    finally:
        if (runner):
            runner.stop()

if __name__ == '__main__':
    main(sys.argv[1:])
