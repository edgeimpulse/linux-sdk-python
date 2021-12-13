import os

def get_device():
    # On Jetson Nano `OPENBLAS_CORETYPE=ARMV8` needs to be set, otherwise including OpenCV
    # throws an illegal instruction error
    if (os.path.exists('/proc/device-tree/model')):
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            if ('NVIDIA Jetson Nano' in model):
                return 'jetson-nano'
    return 'unknown'

device = get_device()
if (device == 'jetson-nano'):
    os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'
