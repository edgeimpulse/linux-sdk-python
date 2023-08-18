# First, install the dependencies via:
#    $ pip3 install requests

import json
import time, hmac, hashlib
import requests
import re, uuid
import math

# Your API & HMAC keys can be found here (go to your project > Dashboard > Keys to find this)
HMAC_KEY = "fed53116f20684c067774ebf9e7bcbdc"
API_KEY = "ei_fd83..."

# empty signature (all zeros). HS256 gives 32 byte signature, and we encode in hex, so we need 64 characters here
emptySignature = ''.join(['0'] * 64)

# use MAC address of network interface as deviceId
device_name =":".join(re.findall('..', '%012x' % uuid.getnode()))

# here we have new data every 16 ms
INTERVAL_MS = 16

if INTERVAL_MS <= 0:
    raise Exception("Interval in miliseconds cannot be equal or lower than 0.")

# here we'll collect 2 seconds of data at a frequency defined by interval_ms
freq =1000/INTERVAL_MS
values_list=[]
for i in range (2*int(round(freq,0))):
    values_list.append([math.sin(i * 0.1) * 10,
                math.cos(i * 0.1) * 10,
                (math.sin(i * 0.1) + math.cos(i * 0.1)) * 10])

data = {
    "protected": {
        "ver": "v1",
        "alg": "HS256",
        "iat": time.time() # epoch time, seconds since 1970
    },
    "signature": emptySignature,
    "payload": {
        "device_name":  device_name,
        "device_type": "LINUX_TEST",
        "interval_ms": INTERVAL_MS,
        "sensors": [
            { "name": "accX", "units": "m/s2" },
            { "name": "accY", "units": "m/s2" },
            { "name": "accZ", "units": "m/s2" }
        ],
        "values": values_list
    }
}



# encode in JSON
encoded = json.dumps(data)

# sign message
signature = hmac.new(bytes(HMAC_KEY, 'utf-8'), msg = encoded.encode('utf-8'), digestmod = hashlib.sha256).hexdigest()

# set the signature again in the message, and encode again
data['signature'] = signature
encoded = json.dumps(data)

# and upload the file
res = requests.post(url='https://ingestion.edgeimpulse.com/api/training/data',
                    data=encoded,
                    headers={
                        'Content-Type': 'application/json',
                        'x-file-name': 'idle01',
                        'x-api-key': API_KEY
                    })
if (res.status_code == 200):
    print('Uploaded file to Edge Impulse', res.status_code, res.content)
else:
    print('Failed to upload file to Edge Impulse', res.status_code, res.content)