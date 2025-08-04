# Edge Impulse Linux SDK for Python

This library lets you run machine learning models and collect sensor data on Linux machines using Python. This SDK is part of [Edge Impulse](https://www.edgeimpulse.com) where we enable developers to create the next generation of intelligent device solutions with embedded machine learning. [Start here to learn more and train your first model](https://docs.edgeimpulse.com).

## Installation guide

1. Install a recent version of [Python 3](https://www.python.org/downloads/) and `pip` tools.
1. Install the SDK:

    **Raspberry Pi**

    ```
    $ sudo apt-get install libatlas-base-dev libportaudio0 libportaudio2 libportaudiocpp0 portaudio19-dev libopenjp2-7 libgtk-3-0 libswscale-dev libavformat58 libavcodec58
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

4. Install pip dependencies:

    ```
    $ pip3 install -r requirements.txt
    ```

    For the computer vision examples you'll want `opencv-python>=4.5.1.48,<5`
    Note on macOS on apple silicon, you will need to use a later version,
    4.10.0.84 tested and installs cleanly

## Collecting data

Before you can classify data you'll first need to collect it. If you want to collect data from the camera or microphone on your system you can use the Edge Impulse CLI, and if you want to collect data from different sensors (like accelerometers or proprietary control systems) you can do so in a few lines of code.

### Collecting data from the camera or microphone

To collect data from the camera or microphone, follow the [getting started guide](https://docs.edgeimpulse.com/docs/edge-impulse-for-linux) for your development board.

### Collecting data from other sensors

To collect data from other sensors you'll need to write some code to collect the data from an external sensor, wrap it in the Edge Impulse Data Acquisition format, and upload the data to the Ingestion service. [Here's an end-to-end example](https://github.com/edgeimpulse/linux-sdk-python/blob/master/examples/custom/collect.py).

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
* [Camera (full frame)](https://github.com/edgeimpulse/linux-sdk-python/blob/master/examples/image/classify-full-frame.py) - grabs data from a webcam and classifies it twice (once cut from the left, once cut from the right). This is useful if you have a wide-angle lense and don't want to miss any events.
* [Still image](https://github.com/edgeimpulse/linux-sdk-python/blob/master/examples/image/classify-image.py) - classifies a still image from your hard drive.
* [Video](https://github.com/edgeimpulse/linux-sdk-python/blob/master/examples/image/classify-video.py) - grabs frames from a video source from your hard drive and classifies it.
* [Custom data](https://github.com/edgeimpulse/linux-sdk-python/blob/master/examples/custom/classify.py) - classifies custom sensor data.

## Troubleshooting

### Collecting print out from the model

To display the logging messages (ie, you may be used to in other deployments), init the runner like so
```
# model_info = runner.init(debug=True) # to get debug print out
```
This will pipe stdout and stderr into the same of your own process


### [Errno -9986] Internal PortAudio error (macOS)

If you see this error you can re-install portaudio via:

```
brew uninstall --ignore-dependencies portaudio
brew install portaudio --HEAD​
```

### Abort trap (6) (macOS)

This error shows when you want to gain access to the camera or the microphone on macOS from a virtual shell (like the terminal in Visual Studio Code). Try to run the command from the normal Terminal.app.
