from DoSomething import DoSomething
from board import D23
import adafruit_dht
import argparse
import json
import time
from datetime import datetime
import random
import base64
import pyaudio


class makeClient(DoSomething):
	def notify(self,topic, msg):
	# Define the actions to be performed after client receives a message
		
		#Conversion from JSON to dict
		input_json = json.loads(msg)
		print('hi')
		if float(input_json['audio_duration']) == 1:
			#audio = pyaudio.PyAudio()
			#self.stream = audio.open(format=pyaudio.paInt16, rate=48000, channels=1, input_device_index=3, input=True, frames_per_buffer=4800)
			#self.stream.stop_stream()
			print(f'rtopic:{topic}, received command to record audio')
			# is recording
			#self.stream.start_stream()
			#data = self.stream.read(4800)
			#rames.append(data)
			#self.stream.stop_stream()
			#audio_bytes = b''.join(frames)
			# conversion to bas64:
			#audio_b64bytes = base64.b64encode(audio_bytes)
			#audio_string = audio_b64bytes.decode()


if __name__=="__main__":
	client_1 = makeClient('client_1')
	client_1.run()
	client_1.myMqttClient.mySubscribe("/287787/audio")
	
	#dht_device = adafruit_dht.DHT11(D23)
	
	twenty=True
	while True:
		# GET DATE AND TIME
		now= datetime.now()
		timestamp = int(now.timestamp())
		temperature = []
		humidity = []
		for in in range(6):
			#humidity = dht_device.humidity
			humidity.append(random.randrange(30,50))
			# PACK INFO INTO A JSON

			humidity_senML_json = json.dumps(humidity_senML_json)
			# SEND INFO THROUGH MQTT
			client_1.myMqttClient.myPublish("/287787/weather/humidity",humidity_senML_json)

			#temperature = dht_device.
			temperature.append(random.randrange(0,25))
			# PACK INFO INTO A JSON
		temp_hum_senML_json={
					"bn": "raspberrypi.local",
					"bt": timestamp,
					"e": [{"n": "temperature", "u": "Cel", "t": 0, "v": temperature},{"n": "humidity", "u": "Cel", "t": 0, "v": humidity}]}
			# SEND INFO THROUGH MQTT
		client_1.myMqttClient.myPublish("/287787/weather/temp_hum",json.dumps(temp_hum_senML_json))
		time.sleep(60)

while True:
		time.sleep(1)
			
