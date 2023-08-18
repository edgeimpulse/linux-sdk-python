import os
import time
import numpy as np
import re
import uuid
import requests
import hmac
import math
import hashlib
import json
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    filename="sensor_app.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler())  # To also print to console


IS_MACOS = os.uname().sysname == "Darwin"
HMAC_KEY = "54598668df4fb5ed5d55247b38dd55a3"
API_KEY = "ei_6be3855c55f764ecec5834853c4fd051c684ef4e1cd26e6a074dfeede5636c64"
emptySignature = "".join(["0"] * 64)
device_name = ":".join(re.findall("..", "%012x" % uuid.getnode()))
INTERVAL_MS = 16
freq = 1000 / INTERVAL_MS

BUFFER_SIZE = 5 * int(round(freq, 0))


class MockSerial:
    class Serial:
        def __init__(self, *args, **kwargs):
            pass

        def write(self, data):
            pass

        def close(self):
            pass


class Sensor:
    def __init__(self, mock):
        if mock:
            self.setup_mock()
        else:
            self.setup_real()

    def setup_mock(self):
        print("Running on macOS: Mocking hardware-specific libraries.")

        class MockI2C:
            pass

        class MockADS:
            P0 = 0
            P1 = 1

            class ADS1115:
                data_rate = 860

                def __init__(self, *args, **kwargs):
                    pass

        class MockAnalogIn:
            def __init__(self, *args, **kwargs):
                self.t = 0  # time variable
                self.fs = 1000 / DataHandler.INTERVAL_MS  # sampling frequency

            @property
            def value(self):
                A = 1024  # arbitrary amplitude for the sine wave
                f = 1  # 1 Hz for 60 BPM
                val = A * math.sin(2 * math.pi * f * self.t / self.fs)
                self.t += 1
                if self.t >= self.fs:
                    self.t = 0  # reset the timer after one second
                return val

        self.i2c = MockI2C()
        self.ADS = MockADS
        self.AnalogIn = MockAnalogIn
        self.serial = MockSerial.Serial()

    def setup_real(self):
        import busio
        import board
        import adafruit_ads1x15.ads1115 as ADS
        from adafruit_ads1x15.analog_in import AnalogIn
        import serial

        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.ADS = ADS
        self.AnalogIn = AnalogIn
        self.serial = serial.Serial(self.detect_serial_port(), 115200)

    def try_handshake(self, port):
        """Try to handshake with a device on a given serial port."""
        try:
            with serial.Serial(port, 115200, timeout=1) as ser:
                ser.write(b"IDENTIFY\n")  # Asking the device to identify itself
                response = ser.readline().decode("utf-8").strip()
                return (
                    response == "ECG_PPG_SENSOR"
                )  # Assuming this is the response you expect
        except:
            return False

    def detect_serial_port(self):
        """Detect the correct serial port by trying a handshake."""
        potential_ports = [
            "/dev/ttyAMA0",
            "/dev/ttyUSB0",
            "/dev/ttyUSB1",
            "/dev/ttyACM0",
        ]
        for port in potential_ports:
            if os.path.exists(port) and self.try_handshake(port):
                return port
        logging.error("No suitable serial port found.")
        raise Exception("No suitable serial port found.")


class DataHandler:
    emptySignature = "".join(["0"] * 64)
    device_name = ":".join(re.findall("..", "%012x" % uuid.getnode()))
    INTERVAL_MS = 16

    def __init__(self, AnalogIn, ads=None, ecg_pin=None, ppg_pin=None):
        if ads and ecg_pin and ppg_pin:
            self.ecg_channel = AnalogIn(ads, ecg_pin)
            self.ppg_channel = AnalogIn(ads, ppg_pin)
        else:
            self.ecg_channel = None
            self.ppg_channel = None

    def plot_data(self, data):
        # Use classic style for ECG-like appearance
        plt.style.use("classic")

        ecg_values = [x[0] for x in data]
        ppg_values = [x[1] for x in data]

        fig, axs = plt.subplots(2, 1, figsize=(10, 6))

        # Mock BPM and HRV values for demonstration
        mock_bpm = 60
        mock_hrv = 50

        # Display BPM and HRV above the graph
        fig.suptitle(f"BPM: {mock_bpm}   HRV: {mock_hrv}ms", fontsize=14)

        # Plot ECG data
        axs[0].plot(ecg_values, label="ECG", color="lime")
        axs[0].legend()
        axs[0].set_title("ECG Data")
        axs[0].grid(True, which="both", linestyle="--", linewidth=0.5)
        axs[0].set_facecolor("white")

        # Plot PPG data
        axs[1].plot(ppg_values, color="red", label="PPG")
        axs[1].legend()
        axs[1].set_title("PPG Data")
        axs[1].grid(True, which="both", linestyle="--", linewidth=0.5)
        axs[1].set_facecolor("white")

        # Display the plot
        plt.tight_layout()
        plt.show()

    def send_to_edge_impulse(self, values_list):
        if self.INTERVAL_MS <= 0:
            raise Exception("Interval in milliseconds cannot be equal or lower than 0.")

        data = {
            "protected": {"ver": "v1", "alg": "HS256", "iat": time.time()},
            "signature": self.emptySignature,
            "payload": {
                "device_name": self.device_name,
                "device_type": "LINUX_TEST",
                "interval_ms": self.INTERVAL_MS,
                "sensors": [
                    {"name": "ECG", "units": "mV"},
                    {"name": "PPG", "units": "arb. units"},
                ],
                "values": values_list,
            },
        }

        encoded = json.dumps(data)
        signature = hmac.new(
            bytes(HMAC_KEY, "utf-8"),
            msg=encoded.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()
        data["signature"] = signature
        encoded = json.dumps(data)

        res = requests.post(
            url="https://ingestion.edgeimpulse.com/api/training/data",
            data=encoded,
            headers={
                "Content-Type": "application/json",
                "x-file-name": "idle",
                "x-api-key": API_KEY,
            },
        )

        if res.status_code == 200:
            print("Uploaded file to Edge Impulse", res.status_code, res.content)
        else:
            print("Failed to upload file to Edge Impulse", res.status_code, res.content)


def is_raspberry_pi():
    """Return True if we are running on a Raspberry Pi."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("Hardware"):
                    if "BCM" in line:
                        return True
    except:
        pass
    return False


if __name__ == "__main__":

    def run_sensor(mock):
        if mock:
            sensor = Sensor(mock=True)
            handler = DataHandler(
                sensor.AnalogIn, sensor.ADS, sensor.ADS.P0, sensor.ADS.P1
            )
        else:
            sensor = Sensor(mock=False)
            handler = DataHandler(
                sensor.AnalogIn, sensor.ADS, sensor.ADS.P0, sensor.ADS.P1
            )

        return sensor, handler

    def on_quit():
        root.quit()
        root.destroy()

    # Create main window
    root = tk.Tk()
    root.title("Sensor Data")
    root.geometry("300x200")
    root.configure(bg="black")

    # Add buttons
    btn_mock = tk.Button(
        root,
        text="Use Mock Data",
        command=lambda: run_sensor(True),
        bg="white",
        fg="black",
    )
    btn_real = tk.Button(
        root,
        text="Use Real Data",
        command=lambda: run_sensor(False),
        bg="white",
        fg="black",
    )
    btn_quit = tk.Button(root, text="Quit", command=on_quit, bg="red", fg="white")

    btn_mock.pack(pady=20)
    btn_real.pack(pady=20)
    btn_quit.pack(pady=20)

    root.mainloop()
    print("Collecting data...")
    if is_raspberry_pi():
        sensor, handler = run_sensor(False)
    else:
        sensor, handler = run_sensor(True)

    if (
        IS_MACOS or True
    ):  # This ensures mock data is always chosen regardless of the platform
        handler = DataHandler(sensor.AnalogIn, sensor.ADS, sensor.ADS.P0, sensor.ADS.P1)
    else:
        handler = DataHandler(None)
        values_list = [
            [math.sin(i * 0.1) * 10, math.cos(i * 0.1) * 10]
            for i in range(2 * int(1000 / DataHandler.INTERVAL_MS))
        ]

    last_serial_write_time = time.time()
    collected_data = []

    logging.info("Starting the data collection loop...")

try:
    while True:
        if handler.ecg_channel and handler.ppg_channel:
            ecg_val = handler.ecg_channel.value
            ppg_val = handler.ppg_channel.value
            logging.debug(f"ECG: {ecg_val}, PPG: {ppg_val}")  # Log the collected values

            collected_data.append([ecg_val, ppg_val])

            if len(collected_data) >= BUFFER_SIZE:
                logging.info(f"Buffer size reached {BUFFER_SIZE}. Sending data...")
                handler.send_to_edge_impulse(collected_data)
                handler.plot_data(collected_data)  # This will plot the data.
                collected_data = []
        else:
            logging.warning(
                "ECG and PPG channels not available. Waiting for 1 second..."
            )
            time.sleep(1)
except Exception as e:
    logging.error(f"Error occurred in the loop: {e}")
except KeyboardInterrupt:
    logging.info("Terminated by user")
    sensor.serial.close()
