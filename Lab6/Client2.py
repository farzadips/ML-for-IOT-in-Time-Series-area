from DoSomething import DoSomething
import time
import json
from datetime import datetime as dt

# DEFINE THE BEHAVIOUR ON MESSAGE RECEIVE
# --> Redefine notify method of DoSomething (using inheritance)

class Subsciber(DoSomething):
	def notify(self,topic, msg):
	# Define the actions to be performed after client receives a message
		
		#Conversion from JSON to dict
		input_json = json.loads(msg)
		#print(input_json)
		timestamp = input_json['bt']
		events = input_json['e'][0]
		now = dt.fromtimestamp(float(timestamp))
		datetime_str = now.strftime('%d-%m-%y %H:%M:%S')
		if events['n'] == 'temperature':
			temperature = events['v']
			t_unit = events['u']
			
			print(f'The topic is: {topic}, date:{datetime_str}, temperature:{temperature}')
		elif events['n'] == 'humidity':
			humidity = events['v']
			h_unit = events['u']
			print(f'The topic is: {topic}, date:{datetime_str}, humidity:{humidity}')
			
        	
		
if __name__ == "__main__":
	client_2 = Subsciber('client_2')
	client_2.run()
	client_2.myMqttClient.mySubscribe("/287787/weather/#")
	
	while True:
		time.sleep(1)
