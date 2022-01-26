from DoSomething import DoSomething
import time
import json
from datetime import datetime


if __name__=="__main__":
	client_3 = DoSomething('client_3')
	client_3.run()
	client_3.myMqttClient.mySubscribe("/287787/audio")
	# PACK INFO INTO A JSON
	timestamp_json=json.dumps({'audio_duration':1})
	# SEND INFO THROUGH MQTT
	client_3.myMqttClient.myPublish("/287787/audio",timestamp_json)
			
