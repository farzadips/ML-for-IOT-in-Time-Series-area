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
		events = input_json['e']
		now = dt.fromtimestamp(float(timestamp))
		datetime_str = now.strftime('%d-%m-%y %H:%M:%S')
		
		temperature = events[0]['v']
		t_unit = events[0]['u']
		humidity = events[1]['v']
		h_unit = events[1]['u']
			

			
		model_path = 'E:/Github/Machine-learning-for-IOT/Homework3/ex1/part 1/models/' + model_name
		interpreter = tf.lite.Interpreter(model_path)
		interpreter.allocate_tensors()
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()
		input_shape = input_details[0]['shape']
		first = True
		window = np.zeros([1, 6, 2], dtype=np.float32)
		expected = np.zeros(2, dtype=np.float32)
		MEAN = np.array([9.107597, 75.904076], dtype=np.float32)
		STD = np.array([ 8.654227, 16.557089], dtype=np.float32)
		
        	
		
if __name__ == "__main__":
	client_2 = Subsciber('client_2')
	client_2.run()
	client_2.myMqttClient.mySubscribe("/287787/weather/#")
	
	while True:
		time.sleep(1)
