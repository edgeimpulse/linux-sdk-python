
import numpy as np
import pyaudio
import time
from six.moves import queue
from edge_impulse_linux.runner import ImpulseRunner as ImpulseRunner
CHUNK_SIZE = 1024
OVERLAP = 0.25

def now():
    return round(time.time() * 1000)

class Microphone():
    def __init__(self, rate, chunk_size, device_id = None, channels = 1):
        self.buff = queue.Queue()
        self.chunk_size = chunk_size
        self.data = []
        self.rate = rate
        self.closed = True
        self.channels = channels
        self.interface = pyaudio.PyAudio()
        self.device_id = device_id
        self.zero_counter = 0

        while not self.device_id or not self.checkDeviceModelCompatibility(self.device_id):
            input_devices = self.listAvailableDevices()
            input_device_id = int(input("Type the id of the audio device you want to use: \n"))
            for device in input_devices:
                if device[0] == input_device_id:
                      if self.checkDeviceModelCompatibility(input_device_id):
                          self.device_id = input_device_id

        print('selected Audio device: %i'% self.device_id)

    def checkDeviceModelCompatibility(self, device_id):
        supported = False
        try:
            supported = self.interface.is_format_supported(self.rate,
                        input_device=device_id,
                        input_channels=self.channels,
                        input_format=pyaudio.paInt16)
        except:
            supported = False
        finally:
            return supported



    def listAvailableDevices(self):
        if not self.interface:
            self.interface = pyaudio.PyAudio()

        info = self.interface.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        input_devices = []
        for i in range (0,numdevices):
            if self.interface.get_device_info_by_host_api_device_index(0,i).get('maxInputChannels')>0:
                input_devices.append((i, self.interface.get_device_info_by_host_api_device_index(0,i).get('name')))

        if len(input_devices) == 0:
            raise Exception('There are no audio devices available');

        for i in range (0, len(input_devices)):
            print("%i --> %s" % input_devices[i])

        return input_devices

    def __enter__(self):
        if not self.interface:
            self.interface = pyaudio.PyAudio()

        self.stream = self.interface.open(
            input_device_index = self.device_id,
            format = pyaudio.paInt16,
            channels = self.channels,
            rate = self.rate,
            input = True,
            frames_per_buffer = self.chunk_size,
            stream_callback = self.fill_buffer
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self.stream.stop_stream()
        self.stream.close()
        self.closed = True
        self.interface.terminate()

    def fill_buffer(self, in_data, frame_count, time_info, status_flags):
        zeros=bytes(self.chunk_size*2)
        if in_data != zeros:
            self.zero_counter = 0
        else:
            self.zero_counter+=1

        if self.zero_counter > self.rate / self.chunk_size:
            self.closed = True
            raise Exception('There is no audio data comming from the audio interface')

        self.buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self.buff.get()

            if chunk is None:
                return

            data = [chunk]
            while True:
                try:
                    chunk = self.buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)

class AudioImpulseRunner(ImpulseRunner):
    def __init__(self, model_path: str):
        super(AudioImpulseRunner, self).__init__(model_path)
        self.closed = True
        self.sampling_rate = 0
        self.window_size = 0
        self.labels = []

    def init(self):
        model_info = super(AudioImpulseRunner, self).init()
        if model_info['model_parameters']['frequency'] == 0:
            raise Exception('Model file "' + self._model_path + '" is not suitable for audio recognition')

        self.window_size = model_info['model_parameters']['input_features_count']
        self.sampling_rate = model_info['model_parameters']['frequency']
        self.labels = model_info['model_parameters']['labels']

        return model_info

    def __enter__(self):
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self.closed = True

    def classify(self, data):
        return super(AudioImpulseRunner, self).classify(data)

    def classifier(self, device_id = None):
        with Microphone(self.sampling_rate, CHUNK_SIZE, device_id=device_id) as mic:
            generator = mic.generator()
            features = np.array([], dtype=np.int16)
            while not self.closed:
                for audio in generator:
                    data = np.frombuffer(audio, dtype=np.int16)
                    features = np.concatenate((features, data), axis=0)
                    while len(features) >= self.window_size:
                        begin = now()
                        res = self.classify(features[:self.window_size].tolist())
                        features = features[int(self.window_size * OVERLAP):]
                        yield res, audio
