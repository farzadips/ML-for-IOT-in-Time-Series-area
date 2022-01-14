import cherrypy
import adafruit_dht
import pyaudio
from board import D4
import datetime
import base64
import json



class Calculator(object):
    exposed = True

    def __init__(self):
        #self.dht_device = adafruit_dht.DHT11(D4)
        audio = pyaudio.PyAudio()
        self.stream = audio.open(format=pyaudio.paInt16, rate=48000, channels=1, input_device_index=3, input=True, frames_per_buffer=4800)
        self.stream.stop_stream()

    def GET(self, *path, **query):
        # datetime:
        now = datetime.datetime.now()
        timestamp = int(now.timestamp())
        # temprature:
        # temprature = self.dht_device.temprature
        temprature = 24
        # humidity:
        # humidity = self.dht_device.humidity
        humidity = 80
        # is recording
        frames = []
        self.stream.start_stream()
        for ii in range(10):
            data = self.stream.read(4800)
            frames.append(data)
        self.stream.stop_stream()
        audio_bytes = b''.join(frames)
        # conversion to bas64:
        audio_b64bytes = base64.b64encode(audio_bytes)
        audio_string = audio_b64bytes.decode()

        # pack data into SENML + JSON STRING
        body = {
                    "bn": "raspberrypi.local",
                    "bt": timestamp,
                    "e": [
                        {"n": "temperature", "u": "Cel", "t": 0, "v": temprature},
                        {"n": "humidity", "u": "%RH", "t": 0, "v": humidity},
                        {"n": "audio", "u": "/", "t": 0, "vd": audio_string}
                    ],
        }

        # convert bosy to JSON
        body = json.dumps(body)

        # send response to the client
        return body

    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass


if __name__ == '__main__':
    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(Calculator(), '', conf)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()
