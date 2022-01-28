import datetime
from DoSomething import DoSomething
import time
import json

import tensorflow as tf
import numpy as np



class Subscriber_and_publish(DoSomething):
    def notify(self, topic, msg):

        #dht_device = adafruit_dht.DHT11(D4)
        input_json = json.loads(msg)
        
        model_name = input_json['model_name']
        tt = input_json['tth'] 
        th = input_json['hth']
        model_path = './' + model_name
        interpreter = tf.lite.Interpreter(model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']
        temp_hum_set = np.array([[]],dtype=np.float32)
        first = True
        window = np.zeros([1, 6, 2], dtype=np.float32)
        expected = np.zeros(2, dtype=np.float32)

        MEAN = np.array([9.107597, 75.904076], dtype=np.float32)
        STD = np.array([ 8.654227, 16.557089], dtype=np.float32)
        #while(True):
        for r in range(50):
            if(first):
                temp_hum_set = np.array([[-8.2, 93.3]],dtype=np.float32)
                first = False
            else:
                time.sleep(1)
            #np.array([[[-8.2, 93.3],[-8.41, 93.4],[-8.51, 93.90],[-8.31, 94.2],[-8.27, 94.1],[-8.05, 94.4],[-7.62, 94.8]]],dtype=np.float32)
                temp_hum_set = np.append(temp_hum_set,[[-8.2, 93.3]], axis = 0)
            
            if(temp_hum_set.shape[0]>6):
                print("predicting")
                now = datetime.datetime.now()
                datetime_str = str(now.strftime('%d-%m-%y %H:%M:%S'))
                for i in range(7):
                    temperature = temp_hum_set[i][0]
                    humidity = temp_hum_set[i][1]
                    if i < 6:
                        window[0, i, 0] = np.float32(temperature)
                        window[0, i, 1] = np.float32(humidity)
                    if i == 6:
                        expected[0] = np.float32(temperature)
                        expected[1] = np.float32(humidity)
                        window = (window - MEAN) / STD
                        interpreter.set_tensor(input_details[0]['index'], window)
                        interpreter.invoke()
                        predicted = interpreter.get_tensor(output_details[0]['index'])
                        temp_hum_set = temp_hum_set[1:]
                        if(abs(predicted[0][0][0]-expected[0])>tt):
                            error_temp =  '(' + str(datetime_str) + ')' +'  Temperature Alert: Predicted=' + str(predicted[0][0][0]) + '°C Actual=' + str(expected[0]) + '°C'
                            
                            error_temp = {"n": "temperature", "u":"/", "t":0, "v":error_temp} 
                            
                            body = {
                                    'e': error_temp
                            }
                            body_json = json.dumps(body)
                            
                            print("Alert published for Temperature")
                            test2.myMqttClient.myPublish('/1362341525/alerts', body_json)
                            #return output_json
                        if(abs(predicted[0][0][1]-expected[1])>th):
                            error_humd = '(' + str(datetime_str) + ')' + '  Humidity Alert: Predicted=' + str(predicted[0][0][1]) + '% Actual=' + str(expected[1]) + '%'
                            
                            error_humd = {"n": "humidity", "u":"/", "t":0, "v":error_humd}
                            body = {
                                    'e': error_humd
                            }
                            body_json = json.dumps(body)
                            print("Alert published for Humidity")
                            test2.myMqttClient.myPublish('/1362341525/alerts', body_json)
                            

                      

        print("\n")

        test.end()
        test2.end()
                    


if __name__ == "__main__":
    test = Subscriber_and_publish("subscriber 9")
    test.run()
    test.myMqttClient.mySubscribe("/1362341525/tt")

    test2 = Subscriber_and_publish("publisher 9")
    test2.run()
    

    while True:
        time.sleep(1)
