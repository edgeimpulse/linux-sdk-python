import subprocess
import os.path
import tempfile
import shutil
import time
import signal
import socket
import json

def now():
    return round(time.time() * 1000)

class ImpulseRunner:
    def __init__(self, model_path: str):
        self._model_path = model_path
        self._tempdir = None
        self._runner = None
        self._client = None
        self._ix = 0

    def init(self):
        if (not os.path.exists(self._model_path)):
            raise Exception('Model file does not exist: ' + self._model_path)

        if (not os.access(self._model_path, os.X_OK)):
            raise Exception('Model file "' + self._model_path + '" is not executable')

        self._tempdir = tempfile.mkdtemp()
        socket_path = os.path.join(self._tempdir, 'runner.sock')
        self._runner = subprocess.Popen([self._model_path, socket_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        while not os.path.exists(socket_path) or not self._runner.poll() is None:
            time.sleep(0.1)

        if not self._runner.poll() is None:
            raise Exception('Failed to start runner (' + str(self._runner.poll()) + ')')

        self._client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._client.connect(socket_path)

        return self.hello()

    def stop(self):
        if (self._tempdir):
            shutil.rmtree(self._tempdir)

        if (self._client):
            self._client.close()

        if (self._runner):
            os.kill(self._runner.pid, signal.SIGINT)
            # todo: in Node we send a SIGHUP after 0.5sec if process has not died, can we do this somehow here too?

    def hello(self):
        msg = { "hello": 1 }
        return self.send_msg(msg)

    def classify(self, data):
        msg = { "classify": data }
        return self.send_msg(msg)

    def send_msg(self, msg):
        t_send_msg = now()

        if not self._client:
            raise Exception('ImpulseRunner is not initialized')

        self._ix = self._ix + 1
        ix = self._ix

        msg["id"] = ix
        self._client.send(json.dumps(msg).encode('utf-8'))

        t_sent_msg = now()

        # i'm not sure if this is right, we should switch to async i/o for this like in Node
        # I think that would perform better
        data = self._client.recv(1 * 1024 * 1024)

        t_received_msg = now()

        braces_open = 0
        braces_closed = 0
        line = ''
        resp = None

        for c in data.decode('utf-8'):
            if c == '{':
                line = line + c
                braces_open = braces_open + 1
            elif c == '}':
                line = line + c
                braces_closed = braces_closed + 1
                if (braces_closed == braces_open):
                    resp = json.loads(line)
            elif braces_open > 0:
                line = line + c

            if (not resp is None):
                break

        if (not resp or resp["id"] != ix):
            raise Exception('Wrong id, expected: ' + str(ix) + ' but got ' + resp["id"])

        if not resp["success"]:
            raise Exception(resp["error"])

        del resp["id"]
        del resp["success"]

        t_parsed_msg = now()
        # print('sent', t_sent_msg - t_send_msg, 'received', t_received_msg - t_send_msg, 'parsed', t_parsed_msg - t_send_msg)
        return resp
