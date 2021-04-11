# Edge Impulse Linux SDK for Python

This library lets you run machine learning models and collect sensor data on Linux machines using Python. This SDK is part of [Edge Impulse](https://www.edgeimpulse.com) where we enable developers to create the next generation of intelligent device solutions with embedded machine learning. [Start here to learn more and train your first model](https://docs.edgeimpulse.com).

## Installation guide

1. Install a recent version of [Python 3](https://www.python.org/downloads/).
1. Install the SDK:

    **Raspberry Pi**

    ```
    $ sudo apt-get install libatlas-base-dev libportaudio0 libportaudio2 libportaudiocpp0 portaudio19-dev
    $ pip3 install edge_impulse_linux -i https://pypi.python.org/simple
    ```

    **Other platforms**

    ```
    $ pip3 install edge_impulse_linux
    ```

1. Clone this repository to get the examples:

    ```
    $ git clone https://github.com/edgeimpulse/linux-sdk-python
    ```

## Collecting data

Before you can classify data you'll first need to collect it. If you want to collect data from the camera or microphone on your system you can use the Edge Impulse CLI, and if you want to collect data from different sensors (like accelerometers or proprietary control systems) you can do so in a few lines of code.

### Collecting data from the camera or microphone

To collect data from the camera or microphone, follow the [getting started guide](https://docs.edgeimpulse.com/docs/edge-impulse-for-linux) for your development board.

### Collecting data from other sensors

To collect data from other sensors you'll need to write some code to collect the data from an external sensor, wrap it in the Edge Impulse Data Acquisition format, and upload the data to the Ingestion service. [Here's an end-to-end example](https://github.com/edgeimpulse/linux-sdk-python/blob/master/custom/collect.py).

## Classifying data

To classify data (whether this is from the camera, the microphone, or a custom sensor) you'll need a model file. This model file contains all signal processing code, classical ML algorithms and neural networks - and typically contains hardware optimizations to run as fast as possible. To grab a model file:

1. Train your model in Edge Impulse.
1. Install the [Edge Impulse for Linux CLI](https://docs.edgeimpulse.com/docs/edge-impulse-for-linux).
1. Download the model file via:

    ```
    $ edge-impulse-linux-runner --download modelfile.eim
    ```

    This downloads the file into `modelfile.eim`. (Want to switch projects? Add `--clean`)

Then you can start classifying realtime sensor data. We have examples for:

* [Audio](https://github.com/edgeimpulse/linux-sdk-python/blob/master/examples/audio/classify.py) - grabs data from the microphone and classifies it in realtime.
* [Camera](https://github.com/edgeimpulse/linux-sdk-python/blob/master/examples/image/classify.py) - grabs data from a webcam and classifies it in realtime.
* [Custom data](https://github.com/edgeimpulse/linux-sdk-python/blob/master/examples/custom/classify.py) - classifies custom sensor data.

## Troubleshooting

### [Errno -9986] Internal PortAudio error (macOS)

If you see this error you can re-install portaudio via:

```
brew uninstall --ignore-dependencies portaudio
brew install portaudio --HEAD​
```

### Abort trap (6) (macOS)

This error shows when you want to gain access to the camera or the microphone on macOS from a virtual shell (like the terminal in Visual Studio Code). Try to run the command from the normal Terminal.app.
